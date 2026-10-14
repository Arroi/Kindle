"""Launch one of our worker processes for an engine, after installing deps and downloading weights."""

from __future__ import annotations

import asyncio
import json
import sys

import httpx

from ..capabilities import local_profile
from ..catalog import hf
from ..proc import ManagedProcess
from ..schemas import DeploymentSpec
from ..util import free_port
from .base import Backend, OpenAIUpstream, WorkerBackend
from .context import DeployError, EngineContext
from .download import hf_download, repo_files
from .install import apt_install, ensure_venv, nvidia_lib_path, pip_install, worker_pythonpath
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
        sel = hf.diffusers_files(siblings)
        if not sel["files"]:
            raise DeployError(f"{repo} isn't a diffusers pipeline repo")
        path = await hf_download(ctx, repo, sel["files"], sel["size"], revision=revision)
        return path, spec.model.variant or sel["variant"]
    if mode == "transformers":
        sel = hf.transformers_files(siblings)
        return await hf_download(ctx, repo, sel["files"], sel["size"], revision=revision), None
    import fnmatch

    expected = sum(int(s.get("size") or 0) for s in siblings
                   if not allow or any(fnmatch.fnmatch(s["rfilename"], p) for p in allow))
    return await hf_download(ctx, repo, allow, expected or None, revision=revision), None


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
    adapter = audio_adapter(model_config(model_path)) if engine == "transformers_tts" else None
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
