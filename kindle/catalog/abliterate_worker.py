"""Abliteration worker: remove a text model's refusal direction and save the result as an imported model.

Two methods:

* ``quick``: classic directional ablation. Mean last-token residuals of harmful and harmless prompts give a
  refusal direction per layer; a few candidate layers are tried with an inference-time hook and the one
  that refuses least is projected out of every matrix that writes to the residual stream
  (attention output, MLP down projections, token embeddings). The weight edit streams the safetensors
  shards on the CPU/GPU one tensor at a time, so it works for models larger than GPU memory.
* ``heretic``: runs Heretic (https://heretic-project.org), which searches ablation parameters with
  Optuna, unattended in its own environment, keeping the trial with the fewest refusals.

Runs as a subprocess of ``ModelImports``; progress is reported with ``KUI_IMPORT`` lines.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ..util import dir_size
from . import attached, hf
from .import_worker import check_space, report

HARMFUL = ("mlabonne/harmful_behaviors", "text")
HARMLESS = ("mlabonne/harmless_alpaca", "text")
HERETIC = "heretic-llm==1.4.0"
REFUSAL_MARKERS = ("i can't", "i cannot", "i can’t", "i won't", "i will not", "i'm not able", "i am not able",
                   "i'm unable", "i am unable", "sorry", "i apologize", "as an ai", "not appropriate",
                   "i must decline", "cannot assist", "can't assist", "can't help", "cannot help",
                   "against my", "i'm sorry", "illegal", "unethical", "harmful")
# Matrices that write into the residual stream (any architecture that names them like Llama/Qwen/Gemma/Mistral).
WRITERS = re.compile(r"(?:^|\.)layers\.\d+\.(?:self_attn\.o_proj|attn\.o_proj|mlp\.(?:.*\.)?down_proj)\.weight$")
EMBED = re.compile(r"(?:^|\.)embed_tokens\.weight$")


def refused(text: str) -> bool:
    head = text.strip().lower()[:160]
    return any(marker in head for marker in REFUSAL_MARKERS)


def source_files(repo: str, revision: str | None) -> tuple[Path, list[str]]:
    """Download the unquantized safetensors checkpoint (into the shared HF cache)."""
    import httpx
    from huggingface_hub import snapshot_download

    with httpx.Client(timeout=30) as client:
        r = client.get(f"{hf.API}/models/{repo}", params={"blobs": "true"}, headers=hf._headers(),
                       follow_redirects=True)
        r.raise_for_status()
        siblings = r.json().get("siblings") or []
    names = [s["rfilename"] for s in siblings]
    if not any(n.endswith(".safetensors") for n in names):
        raise ValueError("Abliteration needs a Hugging Face checkpoint in safetensors format "
                         "(GGUF, Ollama and .bin-only models can't be edited). Pick the original model repo.")
    files = hf.transformers_files(siblings)["files"]
    report(status="downloading", progress=None, message=f"Downloading {repo}")
    path = snapshot_download(repo, revision=revision, allow_patterns=files, token=os.environ.get("HF_TOKEN") or None)
    config = json.loads((Path(path) / "config.json").read_text())
    if config.get("quantization_config"):
        raise ValueError("This checkpoint is already quantized (GPTQ/AWQ/bitsandbytes/FP8). "
                         "Abliterate the original full-precision repo instead.")
    return Path(path), files


def load_prompts(dataset: tuple[str, str], count: int, offset: int = 0) -> list[str]:
    from datasets import load_dataset

    rows = load_dataset(dataset[0], split="train")
    return [str(r) for r in rows[dataset[1]][offset:offset + count]]


def quick(job: dict, out: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src, files = source_files(job["repo"], job.get("revision"))
    weights = sum(f.stat().st_size for f in src.glob("**/*.safetensors"))
    check_space(out.parent, int(weights * 1.05))
    report(status="abliterating", progress=0.05, message="Loading the model for analysis")

    tok = AutoTokenizer.from_pretrained(src)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) \
        if torch.cuda.is_available() else 0
    kwargs: dict = {"torch_dtype": torch.float16, "device_map": "auto", "low_cpu_mem_usage": True}
    if job.get("quantize") or (vram and weights > vram * 0.8):
        from transformers import BitsAndBytesConfig

        # 4-bit is only for measuring the direction; the saved weights are edited at full precision.
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                           bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(src, **kwargs)
    model.eval()
    layers = model.config.get_text_config().num_hidden_layers if hasattr(model.config, "get_text_config") \
        else model.config.num_hidden_layers
    device = next(model.parameters()).device

    def chat(prompt: str) -> str:
        try:
            return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True)
        except Exception:  # noqa: BLE001  (base models without a chat template)
            return prompt

    count = int(job.get("samples") or 128)
    report(status="abliterating", progress=0.1, message="Measuring the refusal direction")
    harmful, harmless = load_prompts(HARMFUL, count), load_prompts(HARMLESS, count)
    held_out = load_prompts(HARMFUL, 12, offset=count)

    @torch.no_grad()
    def means(prompts: list[str]) -> torch.Tensor:
        total = None
        for i in range(0, len(prompts), 8):
            enc = tok([chat(p) for p in prompts[i:i + 8]], return_tensors="pt", padding=True).to(device)
            states = model(**enc, output_hidden_states=True).hidden_states  # embeddings + one per layer
            batch = torch.stack([h[:, -1, :].float().cpu() for h in states[1:]]).sum(dim=1)
            total = batch if total is None else total + batch
        return total / len(prompts)

    directions = means(harmful) - means(harmless)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    blocks = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) == layers and name.endswith("layers"):
            blocks = module
            break
    if blocks is None:
        raise ValueError("Couldn't find this model's decoder layers; try the Heretic method instead.")

    @torch.no_grad()
    def refusals(direction: torch.Tensor | None) -> int:
        hooks = []
        if direction is not None:
            def strip(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                r = direction.to(hidden.device, hidden.dtype)
                hidden.sub_((hidden @ r).unsqueeze(-1) * r)
                return output
            hooks = [block.register_forward_hook(strip) for block in blocks]
        try:
            enc = tok([chat(p) for p in held_out], return_tensors="pt", padding=True).to(device)
            outputs = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(outputs[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            return sum(refused(t) for t in texts)
        finally:
            for hook in hooks:
                hook.remove()

    baseline = refusals(None)
    candidates = sorted({min(layers - 1, int(layers * f)) for f in (0.4, 0.5, 0.6, 0.7, 0.8)})
    scored = []
    for n, layer in enumerate(candidates):
        report(status="abliterating", progress=0.2 + 0.4 * n / len(candidates),
               message=f"Testing layer {layer} ({n + 1}/{len(candidates)}): baseline {baseline}/{len(held_out)} refusals")
        scored.append((refusals(directions[layer]), abs(layer - layers * 0.6), layer))
    best_refusals, _, layer = min(scored)
    direction = directions[layer].clone()
    model = None  # free the analysis copy before editing weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    report(status="abliterating", progress=0.65,
           message=f"Layer {layer}: {baseline} → {best_refusals} of {len(held_out)} test prompts refused. Editing weights")
    hidden = direction.numel()
    work = "cuda" if torch.cuda.is_available() else "cpu"
    r = direction.to(work)
    shards = sorted(src.glob("**/*.safetensors"))
    from safetensors import safe_open
    from safetensors.torch import save_file

    edited = 0
    for i, shard in enumerate(shards):
        tensors = {}
        with safe_open(str(shard), framework="pt") as f:
            metadata = f.metadata()
            for key in f.keys():
                t = f.get_tensor(key)
                if t.ndim == 2 and (WRITERS.search(key) or EMBED.search(key)) and hidden in t.shape:
                    w = t.to(work, torch.float32)
                    if EMBED.search(key) or w.shape[0] != hidden:
                        w -= torch.outer(w @ r, r)  # rows are hidden vectors
                    else:
                        w -= torch.outer(r, r @ w)  # outputs are hidden vectors
                    t = w.to(t.dtype).cpu()
                    edited += 1
                tensors[key] = t
        target = out / shard.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(target), metadata=metadata or {"format": "pt"})
        report(status="abliterating", progress=0.65 + 0.3 * (i + 1) / len(shards),
               message=f"Saved shard {i + 1}/{len(shards)}")
    if not edited:
        raise ValueError("No residual-stream weights matched this architecture; try the Heretic method instead.")
    for name in files:
        if not name.endswith(".safetensors"):
            dest = out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / name, dest)
    (out / "abliteration.json").write_text(json.dumps({
        "method": "quick", "source": job["repo"], "layer": layer, "edited_tensors": edited,
        "test_refusals": {"before": baseline, "after": best_refusals, "prompts": len(held_out)}}, indent=2))


def heretic(job: dict, out: Path) -> None:
    env = Path(job["envs"]) / "heretic"
    exe = env / "bin" / "heretic"
    if not (env / ".kindle-ready").exists():
        report(status="abliterating", progress=None, message="Installing Heretic (first run takes a few minutes)")
        uv = shutil.which("uv")
        if not uv:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
            uv = shutil.which("uv") or str(Path(sys.executable).with_name("uv"))
        subprocess.run([uv, "venv", str(env), "--python", "3.11", "--seed"], check=True)
        subprocess.run([uv, "pip", "install", "--python", str(env / "bin" / "python"), HERETIC], check=True)
        (env / ".kindle-ready").touch()
    trials = int(job.get("trials") or 40)
    work = out.parent / "heretic-work"
    work.mkdir(parents=True, exist_ok=True)
    command = [str(exe), "--model", job["repo"], "--n-trials", str(trials), "--checkpoint-action", "restart",
               "--trial-index", "0", "--model-action", "save", "--save-directory", str(out),
               "--export-strategy", "merge", "--study-checkpoint-dir", str(work / "checkpoints")]
    if job.get("quantize"):
        command += ["--quantization", "bnb_4bit"]
    report(status="abliterating", progress=0.0, message=f"Heretic: preparing {trials} optimization trials")
    progress = re.compile(r"Running trial (\d+) of (\d+)")
    proc = subprocess.Popen(command, cwd=work, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1",
                                                                       "TERM": "dumb", "NO_COLOR": "1"})
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
        if m := progress.search(line):
            done, total = int(m.group(1)), int(m.group(2))
            report(status="abliterating", progress=min(0.95, done / total * 0.9),
                   message=f"Heretic: trial {done} of {total}")
        elif "Saving merged model" in line:
            report(status="abliterating", progress=0.95, message="Heretic: saving the best trial")
    if proc.wait() != 0 or not (out / "config.json").exists():
        raise ValueError(f"Heretic exited without saving a model (exit {proc.returncode}). See the job's log.")
    shutil.rmtree(work, ignore_errors=True)
    (out / "abliteration.json").write_text(json.dumps({"method": "heretic", "source": job["repo"],
                                                        "trials": trials}, indent=2))


def run(job: dict, root: Path) -> dict:
    out = root / (job["repo"].split("/")[-1] + "-abliterated")
    out.mkdir(parents=True)
    (heretic if job.get("method") == "heretic" else quick)(job, out)
    report(status="scanning", progress=None, message="Registering the abliterated model")
    models = attached.scan(root)
    for model in models:
        model.update(url=f"https://huggingface.co/{job['repo']}", imported=True, name=job["name"],
                     size=dir_size(Path(model["path"])))
    info = json.loads((out / "abliteration.json").read_text())
    tested = info.get("test_refusals")
    detail = f" Test prompts refused: {tested['before']} → {tested['after']} of {tested['prompts']}." if tested else ""
    return {"models": models, "message": f"Abliterated with {'Heretic' if info['method'] == 'heretic' else 'quick ablation'}.{detail}"}


if __name__ == "__main__":
    job, root, result = json.loads(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    try:
        answer = run(job, root)
    except Exception as exc:
        answer = {"error": f"{type(exc).__name__}: {exc}"}
    result.write_text(json.dumps(answer))
