"""Launch one of our worker processes for an engine, after installing deps and downloading weights."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import httpx

from ..capabilities import local_profile
from ..catalog import hf
from ..proc import ManagedProcess, run_cmd
from ..schemas import DeploymentSpec
from ..util import free_port, human_bytes
from .base import Backend, OpenAIUpstream, WorkerBackend
from .context import DeployError, EngineContext
from .download import hf_download, repo_files
from .install import (
    apt_install,
    ensure_venv,
    nvidia_lib_path,
    pip_install,
    pip_uninstall,
    pip_upgrade,
    worker_pythonpath,
)
from .model_support import audio_adapter, audio_environment, model_config

WORKER_BASE_REQS = ["fastapi", "uvicorn", "python-multipart", "pydantic", "pyyaml", "httpx"]

WORKERS: dict[str, dict] = {
    "diffusers": {"module": "image_diffusers", "modality": "image", "download": "diffusers",
                  "reqs": ["diffusers>=0.35", "transformers>=4.51", "accelerate>=1.2", "sentencepiece",
                           "protobuf", "ftfy"], "label": "Diffusers"},
    "diffusers_video": {"module": "video_diffusers", "modality": "video", "download": "diffusers",
                        "reqs": ["diffusers>=0.35", "transformers>=4.51", "accelerate>=1.2", "sentencepiece",
                                 "protobuf", "ftfy", "imageio>=2.31", "imageio-ffmpeg"], "label": "Diffusers video"},
    "diffusers_world": {"module": "video_diffusers", "modality": "world", "download": "diffusers",
                        "reqs": ["diffusers>=0.41", "transformers>=4.57", "accelerate>=1.2", "sentencepiece",
                                 "protobuf", "ftfy", "imageio>=2.31", "imageio-ffmpeg", "av", "cosmos_guardrail"], "label": "Diffusers world"},
    "kokoro": {"module": "tts", "modality": "tts", "download": "repo",
               "allow": ["*.json", "*.pth", "voices/*.pt"],
               "reqs": ["kokoro>=0.9.4", "soundfile", "misaki[en]"], "apt": ["espeak-ng"], "label": "Kokoro"},
    "chatterbox": {"module": "tts", "modality": "tts", "download": "repo",
                   "allow": ["ve.*", "t3_cfg.*", "s3gen.*", "tokenizer.json", "conds.pt"],
                   "venv": "chatterbox", "reqs": ["chatterbox-tts", "soundfile"], "label": "Chatterbox"},
    "transformers_tts": {"module": "tts", "modality": "tts", "download": "transformers",
                         "reqs": ["transformers>=4.51", "accelerate>=1.2", "soundfile", "sentencepiece"],
                         "label": "Transformers TTS"},
    "transformers_audio": {"module": "tts", "modality": "audio", "download": "transformers",
                           "reqs": ["transformers>=4.51", "accelerate>=1.2", "soundfile", "sentencepiece"],
                           "label": "Transformers audio"},
    "faster_whisper": {"module": "stt", "modality": "stt", "download": "repo",
                       "reqs": ["faster-whisper>=1.1"], "label": "faster-whisper", "nvidia_libs": True},
    "transformers_asr": {"module": "stt", "modality": "stt", "download": "transformers",
                         "reqs": ["transformers>=4.51", "accelerate>=1.2", "soundfile", "librosa"],
                         "label": "Transformers ASR"},
    "transformers": {"module": "text_transformers", "modality": "text", "download": "transformers",
                     "reqs": ["transformers>=4.51", "accelerate>=1.2"], "label": "Transformers"},
}

PROGRESS_PREFIX = "KUI_PROGRESS "


async def download_model(spec: DeploymentSpec, ctx: EngineContext, mode: str,
                         allow: list[str] | None = None) -> tuple[str, str | None]:
    """Returns (local path, diffusers variant)."""
    if spec.model.source == "attached":
        return spec.model.id, spec.model.variant
    repo = spec.model.id
    revision = spec.options.get("revision")
    siblings = await repo_files(repo, revision=revision)
    if mode == "diffusers":
        if not hf.is_pipeline(siblings):
            async with httpx.AsyncClient() as client:
                twin = await hf.diffusers_twin(client, repo)
            if not twin:
                raise DeployError(f"{repo} is a single-file or original-format checkpoint, not a diffusers pipeline "
                                  "(it has no model_index.json). Deploy its diffusers version instead, usually a "
                                  "repo ending in -Diffusers.")
            ctx.log(f"{repo} isn't in diffusers format; using its diffusers version {twin['id']}")
            repo, siblings, revision = twin["id"], twin.get("siblings") or [], None
        sel = hf.diffusers_files(siblings)
        if not sel["files"]:
            raise DeployError(f"{repo} isn't a diffusers pipeline repo")
        plan = None
        if spec.options.get("weights") != "original":
            try:
                async with httpx.AsyncClient() as client:
                    plan = await hf.bf16_plan(client, repo, siblings, revision)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                ctx.log(f"Couldn't read weight precision ({exc}); downloading the weights as published")
        if plan:
            ctx.log(f"Storing fp32 weights as BF16 saves {human_bytes(plan['saved'])} of disk")
            folder = (repo + (f"@{revision}" if revision else "")).replace("/", "--")
            # v0.2.11-v0.2.15 also converted bitsandbytes quantization state; never reuse those copies.
            shutil.rmtree(ctx.paths.models / "bf16" / folder, ignore_errors=True)
            out = ctx.paths.models / "bf16-v2" / folder
            path = await hf_download(ctx, repo, sel["files"], plan["size"], revision=revision,
                                     convert=plan["files"], out_dir=out)
            return path, spec.model.variant or sel["variant"]
        path = await hf_download(ctx, repo, sel["files"], sel["size"], revision=revision)
        return path, spec.model.variant or sel["variant"]
    if mode == "transformers":
        sel = hf.transformers_files(siblings)
        return await hf_download(ctx, repo, sel["files"], sel["size"], revision=revision), None
    import fnmatch

    expected = sum(int(s.get("size") or 0) for s in siblings
                   if not allow or any(fnmatch.fnmatch(s["rfilename"], p) for p in allow))
    return await hf_download(ctx, repo, allow, expected or None, revision=revision), None


QUANT_PACKAGES = {"bitsandbytes": "bitsandbytes>=0.45", "gguf": "gguf>=0.10", "torchao": "torchao",
                  "quanto": "optimum-quanto", "gptq": "gptqmodel", "awq": "autoawq"}


def quantization_requirements(model_path: str, options: dict) -> list[str]:
    """Packages a checkpoint needs for weights it ships already quantized, or that the user asked to quantize.

    Pre-quantized pipelines (e.g. ``*_nf4_*`` uploads) declare ``quantization_config`` in a component's
    config.json; loading them without the backend fails deep inside diffusers/transformers.
    """
    methods = {"bitsandbytes"} if options.get("quantize") == "nf4" else set()
    root = Path(model_path)
    for config in [root / "config.json", *root.glob("*/config.json")]:
        try:
            if config.stat().st_size > 2 * 1024**2:
                continue
            quant = json.loads(config.read_text()).get("quantization_config")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(quant, dict) and quant.get("quant_method"):
            methods.add(str(quant["quant_method"]).lower())
    return [QUANT_PACKAGES[m] for m in sorted(methods) if m in QUANT_PACKAGES]


def pipeline_class(model_path: str) -> str | None:
    """The diffusers pipeline class a checkpoint names in its model_index.json."""
    try:
        name = json.loads((Path(model_path) / "model_index.json").read_text()).get("_class_name")
    except (OSError, ValueError, AttributeError):
        return None
    return name if isinstance(name, str) and name.isidentifier() else None


DIFFUSERS_SOURCES = [("the latest diffusers release", "diffusers"),
                     ("diffusers from GitHub", "diffusers @ git+https://github.com/huggingface/diffusers")]


async def import_check(python: str, module: str, name: str) -> tuple[bool, str]:
    """Import ``name`` from ``module`` in a fresh interpreter; returns (ok, output)."""
    lines: list[str] = []
    rc = await run_cmd([python, "-c", f"from {module} import {name}"], on_line=lines.append)
    return rc == 0, "\n".join(lines)


async def ensure_pipeline_class(ctx: EngineContext, python: str, model_path: str) -> None:
    """Make sure diffusers can import the checkpoint's pipeline before a worker tries to load it.

    Repairs the two ways a Kaggle environment breaks: a diffusers older than the pipeline
    (``QwenImage21Pipeline``), and a recent diffusers next to Kaggle's older torchao, which makes every
    pipeline fail with ``cannot import name 'FqnToConfig' from 'torchao.quantization'``.
    """
    name = pipeline_class(model_path) or "DiffusionPipeline"
    sources = list(DIFFUSERS_SOURCES)
    torchao_steps = ["upgrade", "remove"]
    output = ""
    for _ in range(len(sources) + len(torchao_steps) + 1):
        ok, output = await import_check(python, "diffusers", name)
        if ok:
            return
        if "torchao" in output and torchao_steps:
            step = torchao_steps.pop(0)
            ctx.log(f"diffusers can't import with the installed torchao; {step} torchao")
            if step == "upgrade":
                try:
                    await pip_upgrade(ctx, ["torchao"], python=python, label="a torchao that matches diffusers")
                except DeployError as exc:
                    ctx.log(str(exc))
            else:
                # KaggleUI never uses torchao; diffusers only imports it when it's installed.
                await pip_uninstall(ctx, ["torchao"], python=python)
        elif f"cannot import name '{name}'" in output and sources:
            label, requirement = sources.pop(0)
            ctx.log(f"{name} isn't in the installed diffusers; installing {label}")
            await pip_upgrade(ctx, [requirement], python=python, label=f"{label} for {name}")
        else:
            break
    if f"cannot import name '{name}'" in output:
        raise DeployError(f"This checkpoint needs diffusers' {name}, which isn't in any diffusers version yet.")
    last = next((line for line in reversed(output.splitlines()) if line.strip()), "unknown error")
    raise DeployError(f"diffusers can't load {name}: {last}")


async def start_worker(ctx: EngineContext, *, python: str, module: str, model: str, engine: str,
                       options: dict, variant: str | None = None, extra_env: dict | None = None,
                       label: str = "model") -> tuple[ManagedProcess, str]:
    port = free_port()
    cmd = [python, "-m", f"kaggleui.engines.workers.{module}", "--model", model, "--name", ctx.served_name,
           "--engine", engine, "--port", str(port), "--options", json.dumps(options),
           "--outputs", str(ctx.paths.outputs)]
    if variant:
        cmd += ["--variant", variant]

    def on_line(line: str) -> None:
        if line.startswith(PROGRESS_PREFIX):
            try:
                ev = json.loads(line[len(PROGRESS_PREFIX):])
                ctx.progress(ev.get("stage", "loading"), ev.get("progress"), ev.get("message", ""))
            except ValueError:
                pass
            return
        ctx.log(line)

    proc = ManagedProcess(f"{module}-{ctx.dep_id}", cmd, env=ctx.env(extra_env),
                          log_path=ctx.paths.logs / f"{ctx.dep_id}-{module}.log", on_line=on_line)
    ctx.processes.append(proc)
    ctx.progress("loading", None, f"Loading {label} into memory")
    await proc.start()
    base = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient(timeout=5) as c:
        for _ in range(3600):
            if not proc.running:
                raise DeployError(f"The {label} worker exited: " + " | ".join(list(proc.tail)[-5:]))
            try:
                r = await c.get(f"{base}/health")
                if r.status_code == 200:
                    h = r.json()
                    if h.get("stage") == "ready":
                        return proc, base
                    if h.get("stage") == "error":
                        raise DeployError(f"Loading failed: {h.get('error')}")
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise DeployError(f"The {label} worker didn't become ready within an hour")


async def deploy(spec: DeploymentSpec, ctx: EngineContext) -> tuple[Backend, dict]:
    engine = spec.model.engine
    cfg = WORKERS.get(engine)
    if not cfg:
        raise DeployError(f"Unknown engine '{engine}'")
    if engine == "diffusers_world" and (ctx.hw.min_sm or 0) < 80:
        raise DeployError("This Cosmos runtime requires an Ampere or newer GPU with bf16 support. "
                          "Kaggle T4 and P100 GPUs aren't supported by this checkpoint.")
    model_path, variant = await download_model(spec, ctx, cfg["download"], cfg.get("allow"))
    if engine == "kokoro" and "vocab" not in model_config(model_path):
        raise DeployError(f"{spec.model.id} holds the old Kokoro v0.19 weights, which the kokoro library no longer "
                          "loads. Deploy hexgrad/Kokoro-82M (v1.0) instead.")
    adapter = audio_adapter(model_config(model_path)) if engine in ("transformers_tts", "transformers_audio") else None
    if adapter == "midasheng" and spec.options.get("trust_remote_code") is not True:
        raise DeployError("MiDashengLM-Gen requires its repository's Python code. In Customize → Settings, "
                          "enable 'Allow model custom code' for this checkpoint and deploy again.")
    extra_env: dict[str, str] = {}
    python = sys.executable
    if adapter or cfg.get("venv"):
        env_name, reqs = audio_environment(adapter) if adapter else (cfg["venv"], cfg["reqs"])
        python = await ensure_venv(ctx, env_name, reqs + WORKER_BASE_REQS, label=adapter or cfg["label"])
        extra_env["PYTHONPATH"] = worker_pythonpath(ctx)
    else:
        await pip_install(ctx, cfg["reqs"], label=cfg["label"])
    quant_reqs = quantization_requirements(model_path, spec.options)
    if quant_reqs:
        if python == sys.executable:
            await pip_install(ctx, quant_reqs, label="quantization support")
        else:
            await pip_install(ctx, quant_reqs, python=python, label="quantization support")
    if cfg["download"] == "diffusers":
        await ensure_pipeline_class(ctx, python, model_path)
    if cfg.get("apt"):
        await apt_install(ctx, cfg["apt"])
    if cfg.get("nvidia_libs"):
        lib = nvidia_lib_path()
        if lib:
            extra_env["LD_LIBRARY_PATH"] = lib

    proc, base = await start_worker(ctx, python=python, module=cfg["module"], model=model_path, engine=engine,
                                    options=spec.options, variant=variant, extra_env=extra_env,
                                    label=cfg["label"])
    if cfg["modality"] == "text":
        detected = local_profile(model_path, spec.model.id, text_only=True)
        backend: Backend = OpenAIUpstream(base, ctx.served_name, template_thinking=True, profile=detected)
    else:
        detected = {}
        backend = WorkerBackend(cfg["modality"], base)
    return backend, {"worker": cfg["module"], "path": model_path, "variant": variant, "adapter": adapter, **detected}
