"""64-bit FNV-1a, the hash llama.cpp's RPC server names its cached tensors by.

Pure Python manages a few MB/s, far too slow for 100 GB of weights, so a tiny C version is compiled
once with the system compiler. ctypes releases the GIL during the call, so tensors hash in parallel.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

OFFSET = 0xCBF29CE484222325
PRIME = 0x100000001B3
MASK = (1 << 64) - 1

SOURCE = r"""
#include <stdint.h>
#include <stddef.h>
uint64_t kui_fnv1a(const unsigned char *data, size_t len, uint64_t hash) {
    for (size_t i = 0; i < len; i++) { hash ^= data[i]; hash *= 0x100000001b3ULL; }
    return hash;
}
"""


def fnv1a_py(data: bytes, value: int = OFFSET) -> int:
    for byte in data:
        value = ((value ^ byte) * PRIME) & MASK
    return value


class Hasher:
    """Incremental FNV-1a: feed chunks, read ``.value``."""

    def __init__(self, fn):
        self._fn = fn
        self.value = OFFSET

    def update(self, data: bytes) -> None:
        self.value = self._fn(data, len(data), self.value) if self._fn else fnv1a_py(data, self.value)


def load(bin_dir: Path):
    """The compiled hash function, or None when no C compiler is available."""
    lib = bin_dir / "libkuifnv.so"
    if not lib.exists():
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if not cc:
            return None
        src = bin_dir / "kuifnv.c"
        src.write_text(SOURCE)
        tmp = lib.with_suffix(".tmp.so")
        result = subprocess.run([cc, "-O2", "-shared", "-fPIC", "-o", str(tmp), str(src)], capture_output=True)
        if result.returncode != 0:
            return None
        tmp.replace(lib)
    fn = ctypes.CDLL(str(lib)).kui_fnv1a
    fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
    fn.restype = ctypes.c_uint64
    return fn


def cache_name(value: int) -> str:
    return f"{value:016x}"
