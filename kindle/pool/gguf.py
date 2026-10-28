"""Just enough GGUF to plan a pooled load: tensor names, byte ranges and the layer count.

Reads the header only (never tensor data), so it is cheap even for 200 GB checkpoints.
"""

from __future__ import annotations

import bisect
import re
import struct
from dataclasses import dataclass
from pathlib import Path

# ggml type id → (elements per block, bytes per block), from ggml.h / ggml.c type traits.
TYPE_SIZES: dict[int, tuple[int, int]] = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
    10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
    23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2),
    34: (256, 54), 35: (256, 66), 39: (32, 17),
}
# GGUF metadata value types → struct format (None: variable length)
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9
LAYER_RE = re.compile(r"^blk\.(\d+)\.")


class GGUFError(ValueError):
    pass


@dataclass
class Tensor:
    name: str
    offset: int  # absolute offset of the tensor's data in the file
    size: int | None  # bytes; None for a type this reader doesn't know

    @property
    def layer(self) -> int | None:
        m = LAYER_RE.match(self.name)
        return int(m.group(1)) if m else None


@dataclass
class Header:
    metadata: dict
    tensors: list[Tensor]

    @property
    def arch(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    @property
    def block_count(self) -> int | None:
        value = self.metadata.get(f"{self.arch}.block_count")
        return int(value) if isinstance(value, int) else None


class _Reader:
    def __init__(self, stream):
        self.stream = stream
        self.pos = 0

    def take(self, n: int) -> bytes:
        data = self.stream.read(n)
        if len(data) != n:
            raise GGUFError("Truncated GGUF header")
        self.pos += n
        return data

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.take(self.unpack("<Q")).decode("utf-8", errors="replace")

    def value(self, kind: int, keep: bool = True):
        if kind in _SCALAR:
            return self.unpack(_SCALAR[kind])
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            inner, count = self.unpack("<I"), self.unpack("<Q")
            if inner in _SCALAR:  # skip big arrays (vocab scores, merges) in one read
                data = self.take(struct.calcsize(_SCALAR[inner]) * count)
                return list(struct.unpack(f"<{count}{_SCALAR[inner][1]}", data)) if keep and count <= 64 else None
            items = [self.value(inner, keep and count <= 64) for _ in range(count)]
            return items if keep and count <= 64 else None
        raise GGUFError(f"Unknown GGUF value type {kind}")


def read_header(path: str | Path) -> Header:
    with open(path, "rb") as f:
        r = _Reader(f)
        if r.take(4) != b"GGUF":
            raise GGUFError(f"{path} is not a GGUF file")
        version = r.unpack("<I")
        if version < 2:
            raise GGUFError(f"GGUF version {version} is too old")
        n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
        metadata = {}
        for _ in range(n_kv):
            key = r.string()
            metadata[key] = r.value(r.unpack("<I"))
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.unpack("<Q") for _ in range(r.unpack("<I"))]
            kind, offset = r.unpack("<I"), r.unpack("<Q")
            infos.append((name, dims, kind, offset))
        align = int(metadata.get("general.alignment") or 32)
        data_start = (r.pos + align - 1) // align * align
    tensors = []
    for name, dims, kind, offset in infos:
        size = None
        if kind in TYPE_SIZES:
            block, nbytes = TYPE_SIZES[kind]
            count = 1
            for d in dims:
                count *= d
            size = count // block * nbytes
        tensors.append(Tensor(name, data_start + offset, size))
    return Header(metadata, tensors)


def layer_devices(block_count: int, split: list[float]) -> list[int]:
    """Device index of each layer, mirroring llama.cpp's layer split with every layer offloaded.

    llama.cpp normalises ``--tensor-split`` into cumulative fractions and puts layer ``il`` on the
    first device whose fraction exceeds ``il / (n_layer + 1)`` (the +1 is the output layer).
    """
    total = sum(split)
    if total <= 0:
        raise ValueError("tensor split must have a positive total")
    cumulative, running = [], 0.0
    for part in split:
        running += part / total
        cumulative.append(running)
    act = block_count + 1
    return [min(bisect.bisect_right(cumulative, il / act), len(split) - 1) for il in range(block_count)]
