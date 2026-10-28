"""Linked sessions: pool another Kindle's running models behind this one's API.

The hub polls each linked session's ``/v1/kindle/node`` with that session's API key. Its ready
models become extra deployments here, so ``model`` names resolve to local and linked copies alike
and requests alternate between them. Only sessions with the same GPUs as this one can be linked.

Forwarded requests carry ``x-kui-hop`` so a session never forwards them again (no A → B → A loops).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

import httpx

from . import __version__
from .deploy.manager import Deployment
from .engines.base import Backend, Unreachable, Unsupported, UpstreamError, _raise_for, iter_sse
from .schemas import DeploymentSpec, ModelRef
from .util import short_id

if TYPE_CHECKING:
    from .server.state import AppState
    from .system.probe import HardwareProfile

HOP_HEADER = "x-kui-hop"
POLL_SECONDS = 10
# Video and world jobs keep their IDs on the session that runs them, so they aren't pooled.
LINKABLE = ("text", "image", "tts", "audio", "stt")


class LinkError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def gpu_signature(hw: HardwareProfile) -> list[str]:
    """What must match between linked sessions: the accelerator model of every device."""
    return sorted(d.name for d in (*hw.gpus, *hw.tpus))


def describe(signature: list[str]) -> str:
    if not signature:
        return "no GPU"
    return ", ".join(f"{n}× {name}" for name, n in Counter(signature).items())


def normalize_url(url: str) -> str:
    """Accept a pasted panel link (with ``?token=…`` or a page path) and keep only the origin."""
    url = url.strip()
    if "://" not in url:
        url = "https://" + url
    parts = urlsplit(url)
    if not parts.hostname:
        raise LinkError(400, "That doesn't look like a Kindle link")
    return urlunsplit((parts.scheme, parts.netloc, "", "", "")).rstrip("/")


def node_info(state: AppState) -> dict:
    """This session as a linked peer sees it: its GPUs and its own (never linked) ready models."""
    hw = state.probe.snapshot()
    sig = gpu_signature(hw)
    return {
        "node_id": state.links.node_id, "version": __version__, "gpus": sig, "gpu_label": describe(sig),
        "models": [{"name": d.served_name, "modality": d.spec.modality,
                    "capabilities": d.info.get("capabilities", []), "reasoning": d.info.get("reasoning")}
                   for d in state.manager.ready(linked=False) if d.spec.modality in LINKABLE],
        "pool": {k: v for k, v in state.pool_helper.info().items() if k in ("enabled", "status", "tag", "gpus", "error")},
    }


class RemoteBackend(Backend):
    """Forwards OpenAI-style calls to the same model on a linked session."""

    def __init__(self, modality: str, base_url: str, model: str, client: httpx.AsyncClient, link: Link | None = None):
        self.modality = modality
        self.remote_url = base_url
        self.upstream_model = model
        self.client = client
        self.link = link

    def _body(self, body: dict) -> dict:
        return {**body, "model": self.upstream_model}

    @contextmanager
    def _reach(self):
        """A session that stops answering leaves the pool now, not at the next 10 s poll."""
        try:
            yield
        except httpx.TransportError as exc:
            if self.link is not None:
                self.link.status, self.link.error = "down", f"Stopped answering ({type(exc).__name__})"
            raise Unreachable("The linked session", exc) from exc

    async def chat(self, body: dict) -> dict:
        with self._reach():
            r = await self.client.post(f"{self.remote_url}/v1/chat/completions", json=self._body({**body, "stream": False}))
        await _raise_for(r)
        return r.json()

    async def chat_stream(self, body: dict) -> AsyncIterator[dict]:
        with self._reach():
            async with self.client.stream("POST", f"{self.remote_url}/v1/chat/completions",
                                          json=self._body({**body, "stream": True})) as r:
                await _raise_for(r)
                async for chunk in iter_sse(r):
                    if chunk.get("error"):
                        error = chunk["error"]
                        raise UpstreamError(502, error.get("message", str(error)) if isinstance(error, dict) else str(error))
                    yield chunk

    async def completions(self, body: dict) -> dict:
        with self._reach():
            r = await self.client.post(f"{self.remote_url}/v1/completions", json=self._body({**body, "stream": False}))
        await _raise_for(r)
        return r.json()

    async def embeddings(self, body: dict) -> dict:
        with self._reach():
            r = await self.client.post(f"{self.remote_url}/v1/embeddings", json=self._body(body))
        await _raise_for(r)
        return r.json()

    async def images(self, body: dict) -> dict:
        r = await self.client.post(f"{self.remote_url}/v1/images/generations",
                                   json=self._body({**body, "response_format": "b64_json"}))
        await _raise_for(r)
        return {"images": [item["b64_json"] for item in r.json().get("data") or [] if item.get("b64_json")]}

    async def speech(self, body: dict) -> tuple[bytes, str]:
        path = "/v1/audio/generations" if self.modality == "audio" else "/v1/audio/speech"
        r = await self.client.post(f"{self.remote_url}{path}", json=self._body(body))
        await _raise_for(r)
        return r.content, r.headers.get("content-type", "audio/wav")

    async def voices(self) -> list[str]:
        return (await self.voice_info()).get("voices", [])

    async def voice_info(self) -> dict:
        r = await self.client.get(f"{self.remote_url}/v1/audio/voices", params={"model": self.upstream_model})
        await _raise_for(r)
        return r.json()

    async def transcribe(self, data: bytes, filename: str, params: dict) -> dict:
        form = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in params.items()
                if v is not None and k not in ("task", "word_timestamps")}
        form.update(model=self.upstream_model, response_format="verbose_json")
        if params.get("word_timestamps"):
            form["timestamp_granularities[]"] = "word"
        path = "/v1/audio/translations" if params.get("task") == "translate" else "/v1/audio/transcriptions"
        r = await self.client.post(f"{self.remote_url}{path}", files={"file": (filename or "audio", data)}, data=form)
        await _raise_for(r)
        return r.json()

    async def job_submit(self, body: dict, image: bytes | None = None) -> str:
        raise Unsupported("video generation on a linked session")


@dataclass
class Link:
    id: str
    url: str
    api_key: str
    status: str = "connecting"  # connecting | up | down | mismatch
    error: str | None = None
    node_id: str | None = None
    version: str | None = None
    gpus: list[str] = field(default_factory=list)
    models: list[dict] = field(default_factory=list)
    last_seen: float | None = None
    pool: dict = field(default_factory=dict)
    deployments: dict[str, Deployment] = field(default_factory=dict)
    client: httpx.AsyncClient | None = None

    def public(self) -> dict:
        return {"id": self.id, "url": self.url, "status": self.status, "error": self.error, "version": self.version,
                "gpu_label": describe(self.gpus) if self.gpus else None, "last_seen": self.last_seen,
                "pool": self.pool,
                "models": [{"name": m["name"], "modality": m["modality"]} for m in self.models]}


class LinkManager:
    def __init__(self, state: AppState, transport: httpx.AsyncBaseTransport | None = None):
        self.state = state
        self.node_id = uuid.uuid4().hex
        self.links: dict[str, Link] = {}
        self._transport = transport
        self._turn: Counter[str] = Counter()
        self._load()

    # -- persistence (URLs and API keys survive app restarts and updates) ----------------------
    @property
    def _file(self):
        return self.state.paths.home / "links.json"

    def _load(self) -> None:
        try:
            saved = json.loads(self._file.read_text())
        except (OSError, ValueError):
            return
        for item in saved if isinstance(saved, list) else []:
            if isinstance(item, dict) and item.get("url"):
                link = Link(id=str(item.get("id") or short_id(6)), url=item["url"], api_key=item.get("api_key", ""))
                self.links[link.id] = link

    def _save(self) -> None:
        data = [{"id": link.id, "url": link.url, "api_key": link.api_key} for link in self.links.values()]
        tmp = self._file.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream)
        tmp.replace(self._file)

    # -- queries ---------------------------------------------------------------------------------
    def list(self) -> dict:
        sig = gpu_signature(self.state.probe.snapshot())
        return {"gpu_label": describe(sig), "can_link": bool(sig), "links": [link.public() for link in self.links.values()]}

    def deployments(self, modality: str | None = None) -> list[Deployment]:
        return [d for link in self.links.values() if link.status == "up" for d in link.deployments.values()
                if modality is None or d.spec.modality == modality]

    def pick(self, matches: list[Deployment]) -> Deployment:
        """Alternate between copies of the same model (local and linked)."""
        key = matches[0].served_name
        choice = matches[self._turn[key] % len(matches)]
        self._turn[key] += 1
        return choice

    # -- changes ---------------------------------------------------------------------------------
    def _client(self, api_key: str) -> httpx.AsyncClient:
        headers = {HOP_HEADER: "1"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return httpx.AsyncClient(headers=headers, transport=self._transport, follow_redirects=True,
                                 timeout=httpx.Timeout(connect=15, read=1800, write=120, pool=15))

    async def _fetch_node(self, client: httpx.AsyncClient, url: str) -> dict:
        try:
            r = await client.get(f"{url}/v1/kindle/node", timeout=20)
        except httpx.HTTPError as e:
            raise LinkError(502, f"Couldn't reach {url}: {type(e).__name__}") from e
        if r.status_code == 401:
            raise LinkError(401, "That session rejected the API key. Copy the API key from its card or panel.")
        if r.status_code == 403:
            raise LinkError(403, "That session has its OpenAI-compatible API turned off.")
        if r.status_code == 404 or "json" not in r.headers.get("content-type", ""):
            raise LinkError(400, "That isn't a Kindle link, or its Kindle is too old to link. Update it first.")
        if r.status_code >= 400:
            raise LinkError(502, f"That session returned HTTP {r.status_code}")
        return r.json()

    def _check(self, node: dict) -> None:
        if node.get("node_id") == self.node_id:
            raise LinkError(400, "That's this session's own link.")
        mine = gpu_signature(self.state.probe.snapshot())
        if not mine:
            raise LinkError(409, "This session has no GPU. Linking pools sessions with matching GPUs.")
        theirs = sorted(node.get("gpus") or [])
        if theirs != mine:
            raise LinkError(409, f"GPUs don't match: this session has {describe(mine)}, "
                                 f"that one has {describe(theirs)}. Use the same accelerator in both.")

    async def add(self, url: str, api_key: str = "") -> dict:
        url = normalize_url(url)
        api_key = api_key.strip()
        if any(link.url == url for link in self.links.values()):
            raise LinkError(409, "That session is already linked.")
        client = self._client(api_key)
        try:
            node = await self._fetch_node(client, url)
            self._check(node)
            if any(link.node_id == node.get("node_id") for link in self.links.values()):
                raise LinkError(409, "That session is already linked under another address.")
        except BaseException:
            await client.aclose()
            raise
        link = Link(id=short_id(6), url=url, api_key=api_key, client=client)
        self._apply(link, node)
        self.links[link.id] = link
        self._save()
        self._changed()
        return link.public()

    async def remove(self, link_id: str) -> None:
        link = self.links.pop(link_id, None)
        if not link:
            raise LinkError(404, "No such linked session")
        self._save()
        if link.client:
            await link.client.aclose()
        self._changed()

    def _apply(self, link: Link, node: dict) -> None:
        link.node_id = node.get("node_id")
        link.version = node.get("version")
        link.gpus = sorted(node.get("gpus") or [])
        link.models = [m for m in node.get("models") or [] if m.get("name") and m.get("modality") in LINKABLE]
        link.last_seen = time.time()
        link.pool = node.get("pool") or {}
        link.status, link.error = "up", None
        keep: dict[str, Deployment] = {}
        for m in link.models:
            key = f"{m['modality']}:{m['name']}"
            dep = link.deployments.get(key)
            if dep is None:
                spec = DeploymentSpec(modality=m["modality"], name=m["name"],
                                      model=ModelRef(source="hf", id=m["name"], engine="linked"))
                dep = Deployment(id=f"{link.id}-{short_id(4)}", spec=spec, served_name=m["name"], status="ready",
                                 message="Ready", progress=1.0, ready_at=time.time(),
                                 backend=RemoteBackend(spec.modality, link.url, m["name"], link.client, link))
            dep.info = {"capabilities": m.get("capabilities") or [], "reasoning": m.get("reasoning"),
                        "runtime": "linked", "linked_url": link.url}
            keep[key] = dep
        link.deployments = keep

    async def refresh(self, link: Link) -> None:
        if link.client is None:
            link.client = self._client(link.api_key)
        def shape() -> dict:
            return {k: v for k, v in link.public().items() if k != "last_seen"}

        before = shape()
        try:
            node = await self._fetch_node(link.client, link.url)
            self._check(node)
            self._apply(link, node)
        except LinkError as e:
            link.status = "mismatch" if e.status == 409 and "GPUs" in e.message else "down"
            link.error = e.message
            link.deployments = {}
        if shape() != before:
            self._changed()

    async def run(self) -> None:
        while True:
            await asyncio.gather(*(self.refresh(link) for link in list(self.links.values())), return_exceptions=True)
            await asyncio.sleep(POLL_SECONDS)

    async def close(self) -> None:
        await asyncio.gather(*(link.client.aclose() for link in self.links.values() if link.client),
                             return_exceptions=True)

    def _changed(self) -> None:
        self.state.bus.publish("links", self.list())

