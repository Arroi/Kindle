"""Prebuilt llama.cpp (llama-server + the RPC server) from the official GitHub releases.

Every session in a pool must run the *same* build: the RPC protocol changes between releases. The
head picks the tag and passes it to its helpers.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path

import httpx

# Verified to publish ubuntu-cuda-12.8 / 13.4 builds. Override with KINDLE_LLAMACPP_TAG.
DEFAULT_TAG = "b11960"
RELEASES = "https://github.com/ggml-org/llama.cpp/releases/download"
RPC_NAMES = ("ggml-rpc-server", "rpc-server", "llama-rpc-server")


class BinaryError(RuntimeError):
    pass


def tag() -> str:
    return os.environ.get("KINDLE_LLAMACPP_TAG", DEFAULT_TAG)


def cuda_flavor() -> str:
    """CUDA runtime build the installed driver can run (Blackwell needs 12.8 or newer)."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if m and int(m.group(1)) >= 13:
        return "13.4"
    return "12.8"


def _extract(archive: Path, root: Path) -> None:
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(root, filter="data")
        except TypeError:  # Pythons before the extraction-filter backport
            tar.extractall(root)  # noqa: S202 - official release archive


def _find(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        hit = next((p for p in root.rglob(name) if p.is_file()), None)
        if hit:
            return hit
    return None


class Build:
    def __init__(self, root: Path):
        self.root = root
        server = _find(root, ("llama-server",))
        if not server:
            raise BinaryError(f"llama-server is missing from {root}")
        self.server = server
        self.rpc = _find(root, RPC_NAMES)

    def env(self) -> dict[str, str]:
        """The release ships its ggml/llama shared libraries (and the CUDA runtime) next to the binaries."""
        dirs = sorted({str(p.parent) for p in self.root.rglob("*.so*")} | {str(self.server.parent)})
        current = os.environ.get("LD_LIBRARY_PATH", "")
        return {"LD_LIBRARY_PATH": ":".join([*dirs, *([current] if current else [])])}


async def ensure(bin_dir: Path, want: str, log: Callable[[str], None]) -> Build:
    """Download and unpack ``want`` once per session; later deployments reuse it."""
    root = bin_dir / f"llama.cpp-{want}"
    if (root / ".complete").exists():
        return Build(root)
    flavor = cuda_flavor()
    names = [f"llama-{want}-bin-ubuntu-cuda-{flavor}-x64.tar.gz", f"cudart-llama-{want}-bin-ubuntu-cuda-{flavor}-x64.tar.gz"]
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(connect=20, read=300, write=60, pool=20)) as client:
        for name in names:
            url = f"{RELEASES}/{want}/{name}"
            log(f"Downloading {url}")
            archive = root / name
            async with client.stream("GET", url) as r:
                if r.status_code == 404:
                    raise BinaryError(f"llama.cpp {want} has no {name}. Set KINDLE_LLAMACPP_TAG to a release that does.")
                r.raise_for_status()
                with archive.open("wb") as out:
                    async for block in r.aiter_bytes(1 << 20):
                        out.write(block)
            await asyncio.to_thread(_extract, archive, root)
            archive.unlink()
    for path in root.rglob("*"):
        if path.is_file() and (path.name.startswith(("llama-", *RPC_NAMES)) or path.suffix == ""):
            path.chmod(0o755)
    build = Build(root)
    (root / ".complete").write_text(want)
    log(f"llama.cpp {want} (CUDA {flavor}) ready: {build.server.parent}")
    return build
