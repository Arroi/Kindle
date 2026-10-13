"""Hugging Face downloads with live progress (polls the cache folder while huggingface_hub downloads)."""

from __future__ import annotations

import asyncio
import json
import os
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
                      expected_size: int | None = None, revision: str | None = None) -> str:
    """Download (part of) a repo into the shared HF cache and return the local snapshot path."""
    cache = repo_cache_dir(repo_id)
    start_size = dir_size(cache)
    ctx.progress("downloading", 0.0 if expected_size else None,
                 f"Downloading {repo_id}" + (f" ({human_bytes(expected_size)})" if expected_size else ""))
    ctx.log(f"Downloading {repo_id} → {cache}")

    # A thread keeps downloading after its asyncio task is cancelled and blocks
    # Python's executor shutdown. A supervised child can be stopped immediately.
    with tempfile.TemporaryDirectory(prefix=f"kui-download-{ctx.dep_id}-") as work:
        result_file = Path(work) / "result.json"
        spec = {"repo_id": repo_id, "allow_patterns": allow_patterns, "revision": revision}
        proc = ManagedProcess(f"download-{ctx.dep_id}", [
            sys.executable, "-m", "kaggleui.engines.download", json.dumps(spec), str(result_file),
        ], env=ctx.env(), on_line=ctx.log)
        ctx.processes.append(proc)
        try:
            await proc.start()
            last_report = 0.0
            while proc.running:
                await asyncio.sleep(0.5)
                if expected_size and time.monotonic() - last_report > 1.0:
                    got = max(dir_size(cache) - start_size, 0)
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
                    "Face token in KaggleUI (Access & API → Keys & tokens), then restart this deployment.")
            if "RepositoryNotFound" in name or "404" in message:
                raise DeployError(f"Model {repo_id} wasn't found on Hugging Face")
            raise DeployError(f"Download failed: {name}: {message}")
        path = result["path"]
    ctx.progress("downloading", 1.0, f"Downloaded {repo_id}")
    return path


def _worker(spec: dict, result_file: Path) -> None:
    try:
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
