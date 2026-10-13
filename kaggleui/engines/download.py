"""Hugging Face downloads with live progress (polls the cache folder while huggingface_hub downloads)."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from ..catalog import hf
from ..util import dir_size, human_bytes
from .context import DeployError, EngineContext


def repo_cache_dir(repo_id: str) -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "hub" / ("models--" + repo_id.replace("/", "--"))


async def repo_files(repo_id: str) -> list[dict]:
    import httpx

    async with httpx.AsyncClient() as client:
        info = await hf.model_info(client, repo_id)
    return info.get("siblings") or []


async def hf_download(ctx: EngineContext, repo_id: str, allow_patterns: list[str] | None = None,
                      expected_size: int | None = None) -> str:
    """Download (part of) a repo into the shared HF cache and return the local snapshot path."""
    from huggingface_hub import snapshot_download

    cache = repo_cache_dir(repo_id)
    start_size = dir_size(cache)
    ctx.progress("downloading", 0.0 if expected_size else None,
                 f"Downloading {repo_id}" + (f" ({human_bytes(expected_size)})" if expected_size else ""))
    ctx.log(f"Downloading {repo_id} → {cache}")

    def run() -> str:
        kwargs = {"repo_id": repo_id, "max_workers": 8}
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
        return snapshot_download(**kwargs)

    task = asyncio.create_task(asyncio.to_thread(run))
    last_report = 0.0
    while not task.done():
        await asyncio.sleep(1.0)
        if expected_size and time.monotonic() - last_report > 1.0:
            got = max(dir_size(cache) - start_size, 0)
            frac = min(got / expected_size, 0.99) if expected_size else None
            ctx.progress("downloading", frac, f"Downloading {repo_id}: {human_bytes(got)} / {human_bytes(expected_size)}")
            last_report = time.monotonic()
    try:
        path = task.result()
    except Exception as e:
        name = type(e).__name__
        if "Gated" in name or "401" in str(e) or "403" in str(e):
            raise DeployError(
                f"{repo_id} is gated. Accept its license at https://huggingface.co/{repo_id} and add an "
                "HF_TOKEN secret (Add-ons → Secrets), then restart KaggleUI.") from e
        if "RepositoryNotFound" in name or "404" in str(e):
            raise DeployError(f"Model {repo_id} wasn't found on Hugging Face") from e
        raise DeployError(f"Download failed: {name}: {e}") from e
    ctx.progress("downloading", 1.0, f"Downloaded {repo_id}")
    return path
