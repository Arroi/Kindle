"""Download tunnel binaries (cloudflared, ngrok, burrow, Gradio's frpc) and the shayef client on demand."""

from __future__ import annotations

import asyncio
import io
import os
import platform
import shutil
import stat
import sys
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


SHAYEF_SOURCE = "git+https://github.com/Arroi/shayefrelay@main"
_shayef_fetched: set[Path] = set()


async def ensure_shayef(cache_dir: Path) -> Path | None:
    """The newest shayef client, from its repo (SHAYEF_SOURCE overrides), once per server start, so its
    fixes reach notebooks without a Kindle release. Installed into its own folder, not site-packages.
    None when it can't be fetched: Kindle then runs the copy it bundles (kindle/_vendor/shayef)."""
    target = cache_dir / "shayef-client"
    if target in _shayef_fetched:
        return target
    source = os.environ.get("SHAYEF_SOURCE") or SHAYEF_SOURCE
    staging = target.with_name("shayef-client.part")
    shutil.rmtree(staging, ignore_errors=True)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(staging), f"shayef @ {source}",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        proc.kill()
        out = b"timed out"
    if proc.returncode != 0 or not (staging / "shayef" / "__main__.py").is_file():
        tail = " | ".join(out.decode(errors="replace").strip().splitlines()[-2:])
        print(f"[kindle] Couldn't fetch the shayef client from {source} ({tail}); using the bundled copy.", flush=True)
        return None
    shutil.rmtree(target, ignore_errors=True)  # a running shayef already has its modules loaded
    staging.rename(target)
    _shayef_fetched.add(target)
    return target


async def ensure_burrow(bin_dir: Path) -> str:
    """burrow's tunnel agent (burroww.lovable.app), which serves a local port on your own domain."""
    local = bin_dir / "burrow-core"
    if local.exists():
        return str(local)
    bin_dir.mkdir(parents=True, exist_ok=True)
    data = await _download(f"https://burroww.lovable.app/api/public/agent/{_os()}-{_arch()}")
    if _os() == "darwin":  # a .tar.gz there, a bare binary on Linux
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            f = tf.extractfile(next(m for m in tf.getmembers() if m.isfile()))
            assert f
            data = f.read()
    tmp = local.with_suffix(".part")
    tmp.write_bytes(data)
    _make_exec(tmp)
    tmp.replace(local)
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
