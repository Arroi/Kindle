from __future__ import annotations

import asyncio
import os
import re
import secrets
import shutil
import socket
import string
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

GB = 1000**3
GiB = 1024**3
MiB = 1024**2


def free_port(start: int | None = None) -> int:
    """Return a free localhost TCP port (``start`` first, if given and free)."""
    if start:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", start))
                return start
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def short_id(n: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


_WORDS = (
    "amber birch cedar delta ember fjord grove harbor indigo juniper kelp lunar maple nova onyx "
    "pebble quartz river sable tundra umber velvet willow xenon yarrow zephyr cobalt dune falcon "
    "glacier heron iris jade koala lotus meadow nectar orchid prism quill raven summit tidal "
    "breeze canyon"
).split()


def generate_password() -> str:
    """Human-friendly but strong password, e.g. ``maple-onyx-river-4821``."""
    words = [secrets.choice(_WORDS) for _ in range(3)]
    return "-".join(words) + f"-{secrets.randbelow(9000) + 1000}"


def generate_api_key() -> str:
    return "kui-" + secrets.token_urlsafe(32)


def human_bytes(n: float | int | None) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


_SIZE_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([KMGT]B)\b")


def parse_size(text: str) -> int | None:
    """Parse '5.2GB' / '815MB' into bytes (decimal units, as Ollama displays them)."""
    m = _SIZE_RE.search(text or "")
    if not m:
        return None
    mult = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4}[m.group(2).upper()]
    return int(float(m.group(1)) * mult)


_PARAM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([mbt])(?![a-z])", re.I)


def parse_params(text: str) -> float | None:
    """Parse '8b', '0.6B', '270m', 'x7b' style parameter counts from a model name/tag."""
    best = None
    for m in _PARAM_RE.finditer(text or ""):
        val = float(m.group(1)) * {"m": 1e6, "b": 1e9, "t": 1e12}[m.group(2).lower()]
        best = val if best is None else max(best, val)
    return best


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def which(name: str, extra_dirs: list[Path] | None = None) -> str | None:
    for d in extra_dirs or []:
        p = d / name
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return shutil.which(name)


async def wait_for(check: Callable[[], Awaitable[bool]], timeout: float, interval: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if await check():
                return True
        except Exception:
            pass
        await asyncio.sleep(interval)
    return False


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")
