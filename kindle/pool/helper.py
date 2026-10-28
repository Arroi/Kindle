"""Helper side of a GPU pool: run llama.cpp's RPC server for a head session and pre-seed its cache.

Without seeding, the head streams every weight it places here through the public relay, which
takes hours for a 100+ GB model. With it, this session downloads exactly those tensors from Hugging
Face itself (byte ranges, so no full checkpoint lands on disk) and stores them the way the RPC
server's ``-c`` cache expects. A tensor seeded by mistake only costs disk; a missing one only costs
time, since the head then sends it over the relay as usual.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from ..proc import ManagedProcess
from ..util import free_port
from . import binaries, fnv

if TYPE_CHECKING:
    from ..server.state import AppState

SEED_CHUNK = 8 << 20
SEED_PARALLEL = 4


class PoolHelper:
    def __init__(self, state: AppState):
        self.state = state
        self.status = "idle"  # idle | installing | starting | seeding | ready | error
        self.message = "Not sharing yet"
        self.error: str | None = None
        self.progress: float | None = None
        self.tag: str | None = None
        self.port: int | None = None
        self.proc: ManagedProcess | None = None
        self.task: asyncio.Task | None = None
        self.seeded_bytes = 0
        self.log_lines: list[str] = []

    @property
    def cache_root(self) -> Path:
        return self.state.paths.cache / "llamacpp-rpc"

    @property
    def cache_dir(self) -> Path:
        # XDG_CACHE_HOME is set below, so the RPC server's cache is <root>/llama.cpp/rpc on every build.
        return self.cache_root / "llama.cpp" / "rpc"

    def _log(self, line: str) -> None:
        self.log_lines = [*self.log_lines[-199:], line]
        print(f"[kindle pool] {line}", flush=True)

    def info(self) -> dict:
        hw = self.state.probe.snapshot()
        return {"enabled": self.state.settings.pool.share_gpu, "status": self.status, "message": self.message,
                "error": self.error, "progress": self.progress, "tag": self.tag,
                "gpus": [{"name": g.name, "vram_total": g.vram_total, "vram_free": g.vram_free} for g in hw.gpus],
                "seeded_bytes": self.seeded_bytes, "log": self.log_lines[-30:]}

    def _set(self, status: str, message: str, progress: float | None = None) -> None:
        self.status, self.message, self.progress = status, message, progress
        self.state.bus.publish("pool", self.info())

    # -- requests from the head ------------------------------------------------------------------
    def prepare(self, tag: str, seed: dict | None) -> dict:
        if not self.state.settings.pool.share_gpu:
            raise PermissionError("This session isn't sharing its GPU. Turn on 'Share this GPU' in its Linked sessions tab.")
        if self.task and not self.task.done():
            if self.tag == tag:
                return self.info()  # already preparing for this head
            self.task.cancel()
        self.task = asyncio.create_task(self._prepare(tag, seed))
        return self.info()

    async def release(self) -> dict:
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.proc:
            await self.proc.stop()
            self.proc = None
        self.port = None
        self._set("idle", "Released by the head session")
        return self.info()

    async def _prepare(self, tag: str, seed: dict | None) -> None:
        try:
            self.error = None
            if not (self.proc and self.proc.running and self.tag == tag):
                if self.proc:
                    await self.proc.stop()
                self._set("installing", f"Installing llama.cpp {tag}")
                build = await binaries.ensure(self.state.paths.bin, tag, self._log)
                if not build.rpc:
                    raise binaries.BinaryError(f"llama.cpp {tag} doesn't include the RPC server")
                self._set("starting", "Starting the RPC server")
                self.port = free_port()
                self.cache_root.mkdir(parents=True, exist_ok=True)
                env = {**build.env(), "XDG_CACHE_HOME": str(self.cache_root)}
                env.pop("LLAMA_CACHE", None)
                os.environ.pop("LLAMA_CACHE", None)
                # 127.0.0.1 only: the RPC server has no authentication. The head reaches it through /_kui/pool/rpc.
                self.proc = ManagedProcess("ggml-rpc", [str(build.rpc), "-H", "127.0.0.1", "-p", str(self.port), "-c"],
                                           env=env, log_path=self.state.paths.logs / "pool-rpc.log", on_line=self._log)
                await self.proc.start()
                await self._wait_port()
                self.tag = tag
            if seed and seed.get("tensors"):
                try:
                    await self._seed(seed)
                except (httpx.HTTPError, OSError, KeyError, ValueError) as exc:  # seeding only saves time
                    self._log(f"Seeding stopped ({type(exc).__name__}: {exc}); the head will send weights over the link.")
            self._set("ready", f"Sharing {self._gpu_label()} with the pool", 1.0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - shown to the head and in this session's panel
            self.error = f"{type(exc).__name__}: {exc}"
            self._log(f"ERROR: {self.error}")
            self._set("error", "Couldn't prepare this GPU for the pool")

    def _gpu_label(self) -> str:
        gpus = self.state.probe.snapshot().gpus
        return ", ".join(g.name for g in gpus) or "this session"

    async def _wait_port(self) -> None:
        for _ in range(120):
            if self.proc and not self.proc.running:
                raise RuntimeError("RPC server exited: " + " | ".join(list(self.proc.tail)[-4:]))
            try:
                _, writer = await asyncio.open_connection("127.0.0.1", self.port)
                writer.close()
                return
            except OSError:
                await asyncio.sleep(0.5)
        raise RuntimeError("RPC server didn't start listening within 60 s")

    # -- cache seeding ---------------------------------------------------------------------------
    async def _seed(self, seed: dict) -> None:
        fn = await asyncio.to_thread(fnv.load, self.state.paths.bin)
        if fn is None:
            self._log("No C compiler here, so weights can't be pre-seeded; the head will send them over the link instead.")
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = self.cache_dir / "kindle-manifest.json"
        try:
            manifest = json.loads(manifest_file.read_text())
        except (OSError, ValueError):
            manifest = {}
        repo, revision = seed["repo"], seed.get("revision") or "main"
        todo = []
        for t in seed["tensors"]:
            key = f"{repo}@{revision}:{t['file']}:{t['offset']}:{t['size']}"
            if not (manifest.get(key) and (self.cache_dir / manifest[key]).exists()):
                todo.append((key, t))
        total = sum(t["size"] for _, t in todo) or 1
        self.seeded_bytes = 0
        self._log(f"Seeding {len(todo)} tensors ({total / 1e9:.1f} GB) from {repo}; "
                  f"{len(seed['tensors']) - len(todo)} already cached")
        headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
        started = time.monotonic()
        queue: asyncio.Queue = asyncio.Queue()
        for item in todo:
            queue.put_nowait(item)

        async with httpx.AsyncClient(follow_redirects=True, headers=headers,
                                     timeout=httpx.Timeout(connect=20, read=120, write=60, pool=60)) as client:
            failures: list[str] = []

            async def worker() -> None:
                while len(failures) < 3:
                    try:
                        key, t = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    url = f"https://huggingface.co/{repo}/resolve/{revision}/{t['file']}"
                    try:
                        manifest[key] = await self._fetch(client, url, t, fn)
                    except (httpx.HTTPError, RuntimeError, OSError) as exc:
                        failures.append(f"{t['name']}: {exc}")
                        self._log(f"Seeding {t['name']} failed ({exc}); the head will send it over the link.")
                        continue
                    rate = self.seeded_bytes / max(time.monotonic() - started, 1e-3)
                    self._set("seeding", f"Downloading this GPU's share of the weights: {self.seeded_bytes / 1e9:.1f} / "
                                         f"{total / 1e9:.1f} GB ({rate / 1e6:.0f} MB/s)", self.seeded_bytes / total)

            try:
                await asyncio.gather(*(worker() for _ in range(SEED_PARALLEL)))
            finally:
                manifest_file.write_text(json.dumps(manifest))
            if len(failures) >= 3:
                self._log("Stopped seeding after repeated failures; the remaining weights will come over the link.")

    async def _fetch(self, client: httpx.AsyncClient, url: str, t: dict, fn) -> str:
        start, size = int(t["offset"]), int(t["size"])
        hasher = fnv.Hasher(fn)
        tmp = self.cache_dir / f".seed-{start}-{os.getpid()}-{id(t)}"
        got = 0
        try:
            async with client.stream("GET", url, headers={"Range": f"bytes={start}-{start + size - 1}"}) as r:
                if r.status_code != 206:
                    raise RuntimeError(f"Hugging Face didn't serve a byte range for {t['file']} (HTTP {r.status_code})")
                with tmp.open("wb") as out:
                    async for block in r.aiter_bytes(SEED_CHUNK):
                        await asyncio.to_thread(hasher.update, block)
                        out.write(block)
                        got += len(block)
                        self.seeded_bytes += len(block)
            if got != size:
                raise RuntimeError(f"Short read for {t['name']}: {got} of {size} bytes")
            name = fnv.cache_name(hasher.value)
            tmp.replace(self.cache_dir / name)
            return name
        finally:
            tmp.unlink(missing_ok=True)
