"""Backend = what the API gateway talks to once a model is running.

Every engine (Ollama, vLLM, our worker processes, the mock engine) hands the deployment manager a
Backend. The gateway speaks OpenAI-style dicts to it and translates other API flavours on top.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class Unsupported(Exception):
    """The model behind this backend can't do the requested operation."""


class Backend:
    modality: str = "text"
    native: frozenset[str] = frozenset()
    """API flavours the upstream speaks natively ("anthropic", "ollama"), proxied without translation."""
    base_url: str | None = None
    upstream_model: str | None = None

    async def chat(self, body: dict) -> dict:
        raise Unsupported("chat")

    def chat_stream(self, body: dict) -> AsyncIterator[dict]:
        raise Unsupported("chat")

    async def completions(self, body: dict) -> dict:
        raise Unsupported("completions")

    async def embeddings(self, body: dict) -> dict:
        raise Unsupported("embeddings")

    async def images(self, body: dict) -> dict:
        raise Unsupported("image generation")

    async def speech(self, body: dict) -> tuple[bytes, str]:
        raise Unsupported("speech")

    async def voices(self) -> list[str]:
        return []

    async def transcribe(self, data: bytes, filename: str, params: dict) -> dict:
        raise Unsupported("transcription")

    async def job_submit(self, body: dict, image: bytes | None = None) -> str:
        raise Unsupported("video generation")

    async def job_status(self, job_id: str) -> dict:
        raise Unsupported("video generation")

    async def job_result(self, job_id: str) -> tuple[bytes, str]:
        raise Unsupported("video generation")

    async def close(self) -> None:
        pass


async def iter_sse(response: httpx.Response) -> AsyncIterator[dict]:
    """Parse an OpenAI-style SSE stream into JSON chunks."""
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            if data == "[DONE]":
                return
            continue
        try:
            yield json.loads(data)
        except ValueError:
            continue


class UpstreamError(Exception):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"upstream error {status}: {detail}")
        self.status = status
        self.detail = detail


async def _raise_for(r: httpx.Response) -> None:
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = (await r.aread()).decode(errors="replace")[:500] if not r.is_closed else r.text[:500]
        raise UpstreamError(r.status_code, detail)


class OpenAIUpstream(Backend):
    """Text backend for anything serving /v1/chat/completions (Ollama, vLLM, our transformers worker)."""

    def __init__(self, base_url: str, upstream_model: str, native: frozenset[str] = frozenset()):
        self.modality = "text"
        self.base_url = base_url.rstrip("/")
        self.upstream_model = upstream_model
        self.native = native
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=600, write=60, pool=10))

    def _body(self, body: dict) -> dict:
        return {**body, "model": self.upstream_model}

    async def chat(self, body: dict) -> dict:
        r = await self.client.post(f"{self.base_url}/v1/chat/completions", json=self._body({**body, "stream": False}))
        await _raise_for(r)
        return r.json()

    async def chat_stream(self, body: dict) -> AsyncIterator[dict]:
        payload = self._body({**body, "stream": True})
        payload.setdefault("stream_options", {"include_usage": True})
        async with self.client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as r:
            await _raise_for(r)
            async for chunk in iter_sse(r):
                yield chunk

    async def completions(self, body: dict) -> dict:
        r = await self.client.post(f"{self.base_url}/v1/completions", json=self._body({**body, "stream": False}))
        await _raise_for(r)
        return r.json()

    async def embeddings(self, body: dict) -> dict:
        r = await self.client.post(f"{self.base_url}/v1/embeddings", json=self._body(body))
        await _raise_for(r)
        return r.json()

    async def close(self) -> None:
        await self.client.aclose()


class WorkerBackend(Backend):
    """Talks to one of our worker processes (image, video, speech, transcription, world)."""

    def __init__(self, modality: str, base_url: str):
        self.modality = modality
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=1800, write=120, pool=10))

    async def images(self, body: dict) -> dict:
        r = await self.client.post(f"{self.base_url}/images", json=body)
        await _raise_for(r)
        return r.json()

    async def speech(self, body: dict) -> tuple[bytes, str]:
        r = await self.client.post(f"{self.base_url}/speech", json=body)
        await _raise_for(r)
        return r.content, r.headers.get("content-type", "audio/wav")

    async def voices(self) -> list[str]:
        try:
            r = await self.client.get(f"{self.base_url}/voices")
            return r.json().get("voices", []) if r.status_code == 200 else []
        except httpx.HTTPError:
            return []

    async def transcribe(self, data: bytes, filename: str, params: dict) -> dict:
        files = {"file": (filename or "audio", data)}
        r = await self.client.post(f"{self.base_url}/transcribe", files=files,
                                   data={k: str(v) for k, v in params.items() if v is not None})
        await _raise_for(r)
        return r.json()

    async def job_submit(self, body: dict, image: bytes | None = None) -> str:
        payload = dict(body)
        if image is not None:
            import base64

            payload["image_b64"] = base64.b64encode(image).decode()
        r = await self.client.post(f"{self.base_url}/jobs", json=payload)
        await _raise_for(r)
        return r.json()["id"]

    async def job_status(self, job_id: str) -> dict:
        r = await self.client.get(f"{self.base_url}/jobs/{job_id}")
        await _raise_for(r)
        return r.json()

    async def job_result(self, job_id: str) -> tuple[bytes, str]:
        r = await self.client.get(f"{self.base_url}/jobs/{job_id}/result")
        await _raise_for(r)
        return r.content, r.headers.get("content-type", "video/mp4")

    async def close(self) -> None:
        await self.client.aclose()
