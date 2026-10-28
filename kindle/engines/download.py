"""Hugging Face downloads with live progress (polls the cache folder while huggingface_hub downloads)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from ..catalog import hf
from ..proc import ManagedProcess
from ..util import dir_size, human_bytes
from .context import DeployError, EngineContext


def repo_cache_dir(repo_id: str) -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "hub" / ("models--" + repo_id.replace("/", "--"))


async def repo_files(repo_id: str, revision: str | None = None) -> list[dict]:
    import httpx

    async with httpx.AsyncClient() as client:
        info = await hf.model_info(client, repo_id, revision=revision)
    return info.get("siblings") or []


async def hf_download(ctx: EngineContext, repo_id: str, allow_patterns: list[str] | None = None,
                      expected_size: int | None = None, revision: str | None = None,
                      convert: list[str] | None = None, out_dir: Path | None = None) -> str:
    """Download (part of) a repo into the shared HF cache and return the local snapshot path.

    With ``convert``/``out_dir`` the files go to ``out_dir`` instead, and the listed .safetensors files
    are stored as BF16 one at a time (see :func:`_convert_worker`), so disk never holds the fp32 copy.
    """
    cache = out_dir if convert else repo_cache_dir(repo_id)
    start_size = dir_size(cache)
    ctx.progress("downloading", 0.0 if expected_size else None,
                 f"Downloading {repo_id}" + (f" ({human_bytes(expected_size)})" if expected_size else ""))
    ctx.log(f"Downloading {repo_id} → {cache}" + (" (fp32 weights stored as BF16)" if convert else ""))

    # A thread keeps downloading after its asyncio task is cancelled and blocks
    # Python's executor shutdown. A supervised child can be stopped immediately.
    with tempfile.TemporaryDirectory(prefix=f"kui-download-{ctx.dep_id}-") as work:
        result_file = Path(work) / "result.json"
        spec = {"repo_id": repo_id, "allow_patterns": allow_patterns, "revision": revision}
        if convert:
            spec.update(convert=convert, out=str(out_dir))
        proc = ManagedProcess(f"download-{ctx.dep_id}", [
            sys.executable, "-m", "kindle.engines.download", json.dumps(spec), str(result_file),
        ], env=ctx.env(), on_line=ctx.log)
        ctx.processes.append(proc)
        try:
            await proc.start()
            last_report = 0.0
            got = 0
            while proc.running:
                await asyncio.sleep(0.5)
                if expected_size and time.monotonic() - last_report > 1.0:
                    # Converted files shrink as they land, so report the high-water mark.
                    got = max(dir_size(cache) - start_size, got)
                    frac = min(got / expected_size, 0.99)
                    ctx.progress("downloading", frac,
                                 f"Downloading {repo_id}: {human_bytes(got)} / {human_bytes(expected_size)}")
                    last_report = time.monotonic()
            await proc.wait()
            result = json.loads(result_file.read_text()) if result_file.exists() else {
                "error": " | ".join(list(proc.tail)[-5:]) or f"Download process exited ({proc.returncode})",
                "type": "DownloadError",
            }
        finally:
            await proc.stop(timeout=2)
        if result.get("error"):
            name, message = result.get("type", "DownloadError"), result["error"]
            if "Gated" in name or "401" in message or "403" in message:
                raise DeployError(
                    f"{repo_id} is gated. Accept its license at https://huggingface.co/{repo_id}, add your Hugging "
                    "Face token in Kindle (Access & API → Keys & tokens), then restart this deployment.")
            if "RepositoryNotFound" in name or "404" in message:
                raise DeployError(f"Model {repo_id} wasn't found on Hugging Face")
            raise DeployError(f"Download failed: {name}: {message}")
        path = result["path"]
    ctx.progress("downloading", 1.0, f"Downloaded {repo_id}")
    return path


def to_bf16(src: Path, dst: Path) -> None:
    """Rewrite one .safetensors file with float32 tensors as bfloat16, a tensor at a time."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    with safe_open(str(src), framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            tensor = f.get_tensor(key)
            keep = tensor.dtype != torch.float32 or hf.QUANT_STATE.search(key)
            tensors[key] = tensor if keep else tensor.to(torch.bfloat16)
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(dst.name + ".partial")
    save_file(tensors, str(partial), metadata=metadata)
    partial.replace(dst)


def _convert_worker(spec: dict) -> str:
    """Download a pipeline into ``out``, storing the ``convert`` files as BF16.

    Each fp32 file is fetched, rewritten and deleted before the next one starts (two at a time), so peak
    disk is the BF16 pipeline plus two fp32 shards. A finished folder is reused on restart.
    """
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download, snapshot_download

    out = Path(spec["out"])
    done = out / ".kindle-complete"
    if done.exists():
        return str(out)
    token = os.environ.get("HF_TOKEN") or None
    convert = set(spec["convert"])
    rest = [f for f in spec.get("allow_patterns") or [] if f not in convert]
    if rest:
        snapshot_download(repo_id=spec["repo_id"], revision=spec.get("revision"), allow_patterns=rest,
                          local_dir=str(out), max_workers=8, token=token)
    incoming = out / ".incoming"

    def one(name: str) -> None:
        target = out / name
        if target.exists():
            return
        src = Path(hf_hub_download(spec["repo_id"], name, revision=spec.get("revision"), local_dir=str(incoming),
                                   token=token))
        to_bf16(src, target)
        src.unlink(missing_ok=True)
        print(f"Stored {name} as BF16", flush=True)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(one, sorted(convert)))
    shutil.rmtree(incoming, ignore_errors=True)
    done.touch()
    return str(out)


def _worker(spec: dict, result_file: Path) -> None:
    try:
        if spec.get("convert"):
            result_file.write_text(json.dumps({"path": _convert_worker(spec)}))
            return
        from huggingface_hub import snapshot_download

        kwargs = {"repo_id": spec["repo_id"], "max_workers": 8}
        if os.environ.get("HF_TOKEN"):
            kwargs["token"] = os.environ["HF_TOKEN"]
        if spec.get("allow_patterns"):
            kwargs["allow_patterns"] = spec["allow_patterns"]
        if spec.get("revision"):
            kwargs["revision"] = spec["revision"]
        result = {"path": snapshot_download(**kwargs)}
    except Exception as exc:
        result = {"error": str(exc), "type": type(exc).__name__}
    result_file.write_text(json.dumps(result))


if __name__ == "__main__":
    _worker(json.loads(sys.argv[1]), Path(sys.argv[2]))
