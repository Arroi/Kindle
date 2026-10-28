"""Download tunnel binaries (cloudflared, ngrok, Gradio's frpc) on demand."""

from __future__ import annotations

import io
import os
import platform
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path

import httpx


def _arch() -> str:
    m = platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64") else "amd64"


def _os() -> str:
    return "darwin" if platform.system() == "Darwin" else "linux"


def _make_exec(p: Path) -> None:
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


async def ensure_cloudflared(bin_dir: Path) -> str:
    local = bin_dir / "cloudflared"
    if local.exists():
        return str(local)
    found = shutil.which("cloudflared")
    if found:
        return found
    bin_dir.mkdir(parents=True, exist_ok=True)
    base = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    if _os() == "linux":
        data = await _download(base + f"cloudflared-linux-{_arch()}")
        local.write_bytes(data)
    else:
        data = await _download(base + f"cloudflared-darwin-{_arch()}.tgz")
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("cloudflared"))
            f = tf.extractfile(member)
            assert f
            local.write_bytes(f.read())
    _make_exec(local)
    return str(local)


async def ensure_ngrok(bin_dir: Path) -> str:
    local = bin_dir / "ngrok"
    if local.exists():
        return str(local)
    found = shutil.which("ngrok")
    if found:
        return found
    bin_dir.mkdir(parents=True, exist_ok=True)
    base = "https://bin.equinox.io/c/bNyj1mQVY4c/"
    if _os() == "linux":
        data = await _download(base + f"ngrok-v3-stable-linux-{_arch()}.tgz")
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            f = tf.extractfile("ngrok")
            assert f
            local.write_bytes(f.read())
    else:
        data = await _download(base + f"ngrok-v3-stable-darwin-{_arch()}.zip")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            local.write_bytes(zf.read("ngrok"))
    _make_exec(local)
    os.environ.setdefault("NGROK_CONFIG", str(bin_dir / "ngrok.yml"))
    return str(local)


FRPC_VERSION = "0.3"


async def ensure_frpc(bin_dir: Path) -> str:
    """Gradio's patched frpc client, the same binary ``gradio share=True`` uses (no pip install needed)."""
    local = bin_dir / f"frpc_gradio_{FRPC_VERSION}"
    if local.exists():
        return str(local)
    bin_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://cdn-media.huggingface.co/frpc-gradio-{FRPC_VERSION}/frpc_{_os()}_{_arch()}"
    try:
        data = await _download(base)
    except httpx.HTTPStatusError:
        data = await _download(f"{base}_v{FRPC_VERSION}")  # older naming
    tmp = local.with_suffix(".part")
    tmp.write_bytes(data)
    _make_exec(tmp)
    tmp.replace(local)
    return str(local)
