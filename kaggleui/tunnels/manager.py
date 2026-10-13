"""Expose local ports publicly: Cloudflare quick tunnels (default), named tunnels, or ngrok.

Kaggle has no inbound ports, so every URL the user opens goes through one of these.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from ..config import Paths
from ..proc import ManagedProcess
from .binaries import ensure_cloudflared, ensure_ngrok

QUICK_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

PROVIDER_LABELS = {
    "cloudflare": "Cloudflare quick tunnel",
    "cloudflare_named": "Cloudflare named tunnel",
    "ngrok": "ngrok",
    "none": "Local only",
}


@dataclass
class Tunnel:
    name: str
    provider: str
    port: int
    url: str | None = None
    status: str = "starting"  # starting | up | restarting | error | stopped
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    restarts: int = 0
    supports_sse: bool = True
    proc: ManagedProcess | None = None
    task: asyncio.Task | None = None

    def info(self) -> dict:
        return {
            "name": self.name, "provider": self.provider, "label": PROVIDER_LABELS.get(self.provider, self.provider),
            "port": self.port, "url": self.url, "status": self.status, "error": self.error,
            "restarts": self.restarts, "supports_sse": self.supports_sse,
        }


class TunnelManager:
    def __init__(self, paths: Paths, on_change: Callable[[Tunnel], None] | None = None):
        self.paths = paths
        self.on_change = on_change
        self.tunnels: dict[str, Tunnel] = {}

    def _changed(self, t: Tunnel) -> None:
        if self.on_change:
            self.on_change(t)

    def get(self, name: str) -> Tunnel | None:
        return self.tunnels.get(name)

    def list(self) -> list[dict]:
        return [t.info() for t in self.tunnels.values()]

    async def open(self, name: str, port: int, provider: str) -> Tunnel:
        await self.close(name)
        if provider == "cloudflare_named" and name != "main":
            provider = "cloudflare"  # a named tunnel's hostname maps to one fixed port
        if provider == "ngrok" and name != "main":
            provider = "cloudflare"  # free ngrok accounts get one agent session
        t = Tunnel(name=name, provider=provider, port=port)
        t.supports_sse = provider != "cloudflare"
        self.tunnels[name] = t
        if provider == "none":
            t.url = f"http://127.0.0.1:{port}"
            t.status = "up"
            self._changed(t)
            return t
        t.task = asyncio.create_task(self._supervise(t))
        self._changed(t)
        return t

    async def close(self, name: str) -> None:
        t = self.tunnels.pop(name, None)
        if not t:
            return
        t.status = "stopped"
        if t.task:
            t.task.cancel()
            await asyncio.gather(t.task, return_exceptions=True)
        if t.proc:
            await t.proc.stop(timeout=5)
        self._changed(t)

    async def close_all(self) -> None:
        await asyncio.gather(*(self.close(name) for name in list(self.tunnels)))

    async def wait_url(self, name: str, timeout: float = 60) -> str | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            t = self.tunnels.get(name)
            if t and t.url and t.status == "up":
                if t.provider == "cloudflare":
                    try:
                        await asyncio.wait_for(asyncio.get_running_loop().getaddrinfo(
                            urlsplit(t.url).hostname, 443, type=socket.SOCK_STREAM), timeout=3)
                    except (OSError, asyncio.TimeoutError):
                        await asyncio.sleep(0.5)
                        continue
                return t.url
            if t and t.status == "error" and t.restarts >= 3:
                return None
            await asyncio.sleep(0.5)
        return None

    # -- supervision --------------------------------------------------------------------------
    async def _supervise(self, t: Tunnel) -> None:
        backoff = 2.0
        while t.status != "stopped":
            try:
                await self._run_once(t)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # binary download failure, bad token, ...
                t.error = str(e)
            if t.status == "stopped":
                return
            t.status = "restarting" if t.restarts < 50 else "error"
            t.restarts += 1
            self._changed(t)
            await asyncio.sleep(min(backoff, 30))
            backoff *= 1.6

    async def _run_once(self, t: Tunnel) -> None:
        if t.provider == "cloudflare":
            await self._run_cloudflare_quick(t)
        elif t.provider == "cloudflare_named":
            await self._run_cloudflare_named(t)
        elif t.provider == "ngrok":
            await self._run_ngrok(t)
        else:
            raise ValueError(f"unknown tunnel provider {t.provider}")

    async def _run_cloudflare_quick(self, t: Tunnel) -> None:
        exe = await ensure_cloudflared(self.paths.bin)
        # A user-level ~/.cloudflared/config.yml breaks quick tunnels; give it a clean HOME.
        home = self.paths.cache / "cloudflared-home"
        home.mkdir(parents=True, exist_ok=True)
        cmd = [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{t.port}"]
        if t.restarts % 2 == 1:
            cmd[2:2] = ["--protocol", "http2"]  # QUIC can be blocked; alternate on retries

        def on_line(line: str) -> None:
            m = QUICK_URL_RE.search(line)
            if m and t.url != m.group(0):
                t.url = m.group(0)
                t.status = "starting"
                t.error = None
                self._changed(t)
            elif "Registered tunnel connection" in line and t.url:
                t.status = "up"
                self._changed(t)

        t.proc = ManagedProcess(f"tunnel-{t.name}", cmd, env={"HOME": str(home)},
                                log_path=self.paths.logs / f"tunnel-{t.name}.log", on_line=on_line)
        await t.proc.start()
        rc = await t.proc.wait()
        if t.status != "stopped":
            t.error = f"cloudflared exited ({rc}): " + " | ".join(list(t.proc.tail)[-3:])
            t.url = None

    async def _run_cloudflare_named(self, t: Tunnel) -> None:
        token = os.environ.get("CF_TUNNEL_TOKEN")
        host = os.environ.get("CF_TUNNEL_HOSTNAME")
        if not token or not host:
            raise RuntimeError("Named tunnel needs CF_TUNNEL_TOKEN and CF_TUNNEL_HOSTNAME secrets")
        exe = await ensure_cloudflared(self.paths.bin)
        t.url = host if host.startswith("http") else f"https://{host}"

        def on_line(line: str) -> None:
            if "Registered tunnel connection" in line:
                t.status = "up"
                t.error = None
                self._changed(t)

        t.proc = ManagedProcess(f"tunnel-{t.name}", [exe, "tunnel", "--no-autoupdate", "run", "--token", token],
                                log_path=self.paths.logs / f"tunnel-{t.name}.log", on_line=on_line)
        await t.proc.start()
        rc = await t.proc.wait()
        if t.status != "stopped":
            t.error = f"cloudflared exited ({rc}): " + " | ".join(list(t.proc.tail)[-3:])

    async def _run_ngrok(self, t: Tunnel) -> None:
        token = os.environ.get("NGROK_AUTHTOKEN")
        if not token:
            raise RuntimeError("ngrok needs the NGROK_AUTHTOKEN secret")
        exe = await ensure_ngrok(self.paths.bin)
        cmd = [exe, "http", str(t.port), "--authtoken", token, "--log", "stdout", "--log-format", "json"]
        domain = os.environ.get("NGROK_DOMAIN")
        if domain:
            cmd += ["--url", domain if domain.startswith("http") else f"https://{domain}"]

        def on_line(line: str) -> None:
            try:
                data = json.loads(line)
            except ValueError:
                return
            url = data.get("url")
            if url and str(url).startswith("https://"):
                t.url = url
                t.status = "up"
                t.error = None
                self._changed(t)
            elif data.get("lvl") in ("eror", "crit") and data.get("err"):
                t.error = str(data.get("err"))[:300]

        t.proc = ManagedProcess(f"tunnel-{t.name}", cmd, log_path=self.paths.logs / f"tunnel-{t.name}.log",
                                on_line=on_line)
        await t.proc.start()
        rc = await t.proc.wait()
        if t.status != "stopped":
            t.error = t.error or f"ngrok exited ({rc})"
            t.url = None
