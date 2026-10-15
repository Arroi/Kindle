"""Mock engine: realistic install/download/load progress and fake outputs, for developing the UI off-GPU."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import math
import struct
import time
import uuid
import wave
import zlib
from collections.abc import AsyncIterator
from pathlib import Path

from ..schemas import DeploymentSpec
from .base import Backend
from .context import EngineContext

ASSETS = Path(__file__).with_name("assets")


def _png(width: int, height: int, seed: int) -> bytes:
    width, height = max(16, min(width, 512)), max(16, min(height, 512))
    r0, g0, b0 = (seed >> 16) & 255, (seed >> 8) & 255, seed & 255
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row += bytes(((r0 + x) % 256, (g0 + y) % 256, (b0 + (x + y) // 2) % 256))
        rows.append(bytes(row))
    raw = zlib.compress(b"".join(rows), 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


def _wav(seconds: float, freq: float = 220.0) -> bytes:
    sr = 16000
    n = int(sr * max(0.4, min(seconds, 20)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            env = min(1.0, i / 800, (n - i) / 800)
            v = 0.25 * env * (math.sin(2 * math.pi * freq * i / sr) + 0.4 * math.sin(2 * math.pi * freq * 1.5 * i / sr))
            frames += struct.pack("<h", int(v * 32767))
        w.writeframes(bytes(frames))
    return buf.getvalue()


class MockBackend(Backend):
    def __init__(self, modality: str, name: str):
        self.modality = modality
        self.name = name
        self.jobs: dict[str, dict] = {}

    def _reply(self, body: dict) -> str:
        msgs = body.get("messages") or []
        last = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                c = m.get("content")
                last = c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))
                break
        return (f"This is a mock reply from {self.name}. You said: \"{last[:200]}\". "
                "Real models answer here once KaggleUI runs on a GPU.")

    async def chat(self, body: dict) -> dict:
        text = self._reply(body)
        return {"id": "chatcmpl-mock-" + uuid.uuid4().hex[:12], "object": "chat.completion", "created": int(time.time()),
                "model": self.name, "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                                 "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": len(text.split()), "total_tokens": 12 + len(text.split())}}

    async def chat_stream(self, body: dict) -> AsyncIterator[dict]:
        cid = "chatcmpl-mock-" + uuid.uuid4().hex[:12]
        base = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": self.name}
        yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
        words = self._reply(body).split(" ")
        for i, w in enumerate(words):
            await asyncio.sleep(0.03)
            yield {**base, "choices": [{"index": 0, "delta": {"content": (" " if i else "") + w}, "finish_reason": None}]}
        yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 12, "completion_tokens": len(words), "total_tokens": 12 + len(words)}}

    async def completions(self, body: dict) -> dict:
        return {"id": "cmpl-mock", "object": "text_completion", "created": int(time.time()), "model": self.name,
                "choices": [{"index": 0, "text": " (mock completion)", "finish_reason": "stop"}]}

    async def embeddings(self, body: dict) -> dict:
        inputs = body.get("input")
        inputs = [inputs] if isinstance(inputs, str) else list(inputs or [])
        data = []
        for i, text in enumerate(inputs):
            h = hashlib.sha256(str(text).encode()).digest()
            vec = [(b - 128) / 128 for b in h]
            data.append({"object": "embedding", "index": i, "embedding": vec})
        return {"object": "list", "data": data, "model": self.name, "usage": {"prompt_tokens": 4, "total_tokens": 4}}

    async def images(self, body: dict) -> dict:
        await asyncio.sleep(0.8)
        w, h = int(body.get("width") or 512), int(body.get("height") or 512)
        if body.get("size") and "x" in str(body["size"]):
            w, h = (int(v) for v in str(body["size"]).split("x", 1))
        n = max(1, min(int(body.get("n") or 1), 4))
        seed = int(hashlib.md5(str(body.get("prompt", "")).encode()).hexdigest()[:6], 16)
        return {"images": [base64.b64encode(_png(w, h, seed + i * 9973)).decode() for i in range(n)]}

    async def speech(self, body: dict) -> tuple[bytes, str]:
        await asyncio.sleep(0.3)
        text = str(body.get("input") or "")
        return _wav(len(text.split()) * 0.3, 200 + (len(text) % 7) * 30), "audio/wav"

    async def voices(self) -> list[str]:
        return ["af_heart", "af_bella", "am_adam", "bf_emma", "bm_george"]

    async def transcribe(self, data: bytes, filename: str, params: dict) -> dict:
        await asyncio.sleep(0.5)
        text = f"This is a mock transcription of {filename or 'audio'} ({len(data)} bytes)."
        return {"text": text, "language": params.get("language") or "en", "duration": 3.0, "task": "transcribe",
                "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": text}]}

    async def job_submit(self, body: dict, image: bytes | None = None) -> str:
        job_id = "job_mock_" + uuid.uuid4().hex[:12]
        self.jobs[job_id] = {"id": job_id, "status": "in_progress", "progress": 0.0, "message": "Generating",
                             "error": None, "created": time.time(), "has_result": False}

        async def run() -> None:
            for i in range(10):
                await asyncio.sleep(0.4)
                self.jobs[job_id]["progress"] = (i + 1) / 10
                self.jobs[job_id]["message"] = f"Step {i + 1}/10"
            self.jobs[job_id].update(status="completed", message="Done", has_result=True)

        asyncio.create_task(run())
        return job_id

    async def job_status(self, job_id: str) -> dict:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    async def job_result(self, job_id: str) -> tuple[bytes, str]:
        return (ASSETS / "mock.mp4").read_bytes(), "video/mp4"


async def deploy(spec: DeploymentSpec, ctx: EngineContext) -> tuple[Backend, dict]:
    ctx.progress("installing", None, f"Installing {spec.model.engine} (simulated)")
    ctx.log(f"[mock] would install engine {spec.model.engine}")
    await asyncio.sleep(0.8)
    for i in range(11):
        ctx.progress("downloading", i / 10, f"Downloading {spec.model.id}: {i * 10}% (simulated)")
        await asyncio.sleep(0.25)
    ctx.progress("loading", None, "Loading into accelerator memory (simulated)")
    where = f"TPU ({len(ctx.hw.tpus)} chips)" if ctx.hw.accelerator == "tpu" and ctx.hw.tpus else f"GPUs {ctx.gpus or 'none'}"
    ctx.log(f"[mock] loading {spec.model.id} on {where}")
    await asyncio.sleep(0.8)
    caps = ["completion", "tools"] if spec.modality == "text" else []
    info: dict = {"mock": True, "capabilities": caps}
    if spec.modality == "world":
        from .world_engine import recipe_for

        try:
            recipe = recipe_for(spec)
            info["controls"] = recipe.get("controls")
            info["start_frame"] = recipe.get("start_frame")
        except Exception:
            pass
    return MockBackend(spec.modality, ctx.served_name), info
