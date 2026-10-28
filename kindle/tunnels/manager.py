"""Expose local ports publicly: Cloudflare quick tunnels (default), named tunnels, burrow, ngrok, or Gradio share links.

Kaggle has no inbound ports, so every URL the user opens goes through one of these.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from ..config import Paths
from ..proc import ManagedProcess
from .binaries import ensure_burrow, ensure_cloudflared, ensure_frpc, ensure_ngrok
from .persistent import PersistentProcess, binding, read_record, stop_saved_tunnel

QUICK_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
GRADIO_SUCCESS_RE = re.compile(r"start proxy success: (\S+)")
GRADIO_URL_RE = re.compile(r"https://[a-zA-Z0-9.-]+\.gradio\.live")
BURROW_SERVICE_RE = re.compile(r'"service"\s*:\s*"(https?://[^"]+)"')
GRADIO_TUNNEL_APIS =("https://api.gradio.app/v3/tunnel-request", "https://api.gradio.live/v3/tunnel-request")

PROVIDER_LABELS = {
    "cloudflare": "Cloudflare quick tunnel",
    "cloudflare_named": "Cloudflare named tunnel",
    "burrow": "Burrow tunnel",
    "ngrok": "ngrok",
    "gradio": "Gradio share link",
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
    proc: ManagedProcess | PersistentProcess | None = None
    task: asyncio.Task | None = None
    persistent: bool = False

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
        if isinstance(t.proc, PersistentProcess):
            t.proc.save(t.info())
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
        if provider == "burrow" and name != "main":
            provider = "gradio"  # so does burrow's; and it's used where quick tunnels don't work (molab)
        if provider == "ngrok" and name != "main":
            provider = "cloudflare"  # free ngrok accounts get one agent session
        if name == "main":
            record = read_record(self.paths)
            if record and (provider == "none" or record.get("binding") != binding(provider, port)):
                await asyncio.to_thread(stop_saved_tunnel, self.paths)
        t = Tunnel(name=name, provider=provider, port=port, persistent=name == "main")
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

    async def close(self, name: str, *, preserve: bool = False) -> None:
        t = self.tunnels.pop(name, None)
        if not t:
            return
        # Keep the last connected state on disk for the replacement server.
        keep = preserve and isinstance(t.proc, PersistentProcess) and t.proc.running
        if keep:
            t.proc.save(t.info())
        t.status = "stopped"
        if t.task:
            t.task.cancel()
            await asyncio.gather(t.task, return_exceptions=True)
        if t.proc and not keep:
            await t.proc.stop(timeout=5)
        elif name == "main" and not preserve:
            await asyncio.to_thread(stop_saved_tunnel, self.paths)
        if self.on_change:
            self.on_change(t)

    async def close_all(self, *, preserve_main: bool = False) -> None:
        await asyncio.gather(*(self.close(name, preserve=preserve_main and name == "main")
                               for name in list(self.tunnels)))

    async def _start_process(self, t: Tunnel, cmd: list[str], on_line: Callable[[str], None],
                             env: dict[str, str] | None = None) -> None:
        if t.persistent:
            proc = PersistentProcess(self.paths, cmd, signature=binding(t.provider, t.port),
                                     env=env, on_line=on_line)
            record = proc.adopt()
            if record:
                t.url = record.get("url")
                t.status = record.get("status", "starting")
                t.error = record.get("error")
                t.restarts = record.get("restarts", 0)
                print("[kindle] Reusing the existing public tunnel and URL.", flush=True)
            else:
                t.status, t.error = "starting", None
                if t.provider not in ("cloudflare_named", "burrow"):  # their URL is the fixed hostname
                    t.url = None
            t.proc = proc
        else:
            t.proc = ManagedProcess(f"tunnel-{t.name}", cmd, env=env,
                                    log_path=self.paths.logs / f"tunnel-{t.name}.log", on_line=on_line)
        await t.proc.start()
        self._changed(t)

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
                if t.provider == "burrow":
                    problem = await self._reaches_kindle(t)
                    if problem:
                        if not (t.error or "").startswith("burrow forwards"):  # the dashboard's port says more
                            t.error = (f"burrow is connected, but {t.url} doesn't reach Kindle ({problem}). "
                                       f"Point the tunnel at http://127.0.0.1:{t.port} on burrow's dashboard.")
                            self._changed(t)
                        await asyncio.sleep(2)
                        continue
                    if t.error:
                        t.error = None
                        self._changed(t)
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
        elif t.provider == "burrow":
            await self._run_burrow(t)
        elif t.provider == "ngrok":
            await self._run_ngrok(t)
        elif t.provider == "gradio":
            await self._run_gradio(t)
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

        await self._start_process(t, cmd, on_line, env={"HOME": str(home)})
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

        await self._start_process(t, [exe, "tunnel", "--no-autoupdate", "run", "--token", token], on_line)
        rc = await t.proc.wait()
        if t.status != "stopped":
            t.error = f"cloudflared exited ({rc}): " + " | ".join(list(t.proc.tail)[-3:])

    async def _run_burrow(self, t: Tunnel) -> None:
        """burrow serves BURROW_HOSTNAME through its agent. The local port it forwards to is set on
        burrow's dashboard and has to be Kindle's port (8890 unless KINDLE_PORT says otherwise)."""
        token = os.environ.get("BURROW_TOKEN")
        host = os.environ.get("BURROW_HOSTNAME")
        if not token or not host:
            raise RuntimeError("burrow needs the BURROW_TOKEN and BURROW_HOSTNAME secrets")
        exe = await ensure_burrow(self.paths.bin)
        t.url = (host if host.startswith("http") else f"https://{host}").rstrip("/")

        def on_line(line: str) -> None:
            low = line.lower()
            if "registered tunnel connection" in low:
                t.status = "up"
                t.error = None
                self._changed(t)
            elif "token is not valid" in low:
                t.error = "burrow rejected BURROW_TOKEN. Copy the key again from burrow's dashboard."
            elif "updated to new configuration" in low:
                services = BURROW_SERVICE_RE.findall(line.replace('\\"', '"'))
                ports = {urlsplit(s).port for s in services if urlsplit(s).hostname in ("localhost", "127.0.0.1")}
                if ports and t.port not in ports:
                    t.error = (f"burrow forwards {host} to {', '.join(services)}, but Kindle listens on port "
                               f"{t.port}. Point the tunnel at http://127.0.0.1:{t.port} on burrow's dashboard.")
                    self._changed(t)

        # The agent reads its key from TUNNEL_TOKEN, as burrow's own `burrow connect` wrapper passes it.
        await self._start_process(t, [exe, "tunnel", "--no-autoupdate", "run"], on_line, env={"TUNNEL_TOKEN": token})
        rc = await t.proc.wait()
        if t.status != "stopped":
            t.error = t.error or f"burrow exited ({rc}): " + " | ".join(list(t.proc.tail)[-3:])

    async def _reaches_kindle(self, t: Tunnel) -> str | None:
        """None when the hostname reaches Kindle, else what it reached instead. A connected agent isn't
        enough: the hostname must forward to this port, not to nothing (a gateway error) or another app."""
        try:
            async with httpx.AsyncClient(timeout=8, follow_redirects=False) as client:
                r = await client.get(f"{t.url}/_kui/api/session")
        except httpx.HTTPError as e:
            return f"{type(e).__name__}: {e}"[:200]
        try:
            if r.status_code == 200 and "admin" in r.json():
                return None
        except (ValueError, TypeError):
            pass
        return f"HTTP {r.status_code} from another app" if r.status_code < 500 else f"HTTP {r.status_code}"

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

        await self._start_process(t, cmd, on_line)
        rc = await t.proc.wait()
        if t.status != "stopped":
            t.error = t.error or f"ngrok exited ({rc})"
            t.url = None

    async def _gradio_relay(self, t: Tunnel) -> tuple[str, list[str]]:
        """Ask Gradio's API for a relay server; returns ``host:port`` and frpc TLS arguments."""
        override = os.environ.get("GRADIO_SHARE_SERVER_ADDRESS")
        if override:
            return override, []
        relay, error = None, None
        async with httpx.AsyncClient(timeout=30) as client:
            for api in GRADIO_TUNNEL_APIS:
                try:
                    r = await client.get(api)
                    r.raise_for_status()
                    relay = r.json()[0]
                    break
                except (httpx.HTTPError, ValueError, LookupError) as e:
                    error = e
        if relay is None:
            raise RuntimeError(f"Gradio's share server is unavailable: {error}")
        tls: list[str] = []
        # Gradio verifies the relay with its root CA. Alternate plain connections on retries in
        # case the TLS handshake is what fails in this container.
        if relay.get("root_ca") and t.restarts % 2 == 0:
            ca = self.paths.cache / f"gradio-share-ca-{t.name}.pem"  # one per tunnel: they start concurrently
            ca.write_text(relay["root_ca"])
            tls = ["--tls_enable", "--tls_trusted_ca_file", str(ca)]
        return f"{relay['host']}:{int(relay['port'])}", tls

    async def _run_gradio(self, t: Tunnel) -> None:
        exe = await ensure_frpc(self.paths.bin)
        server, tls = await self._gradio_relay(t)
        cmd = [exe, "http", "-n", secrets.token_urlsafe(32), "-l", str(t.port), "-i", "127.0.0.1",
               "--uc", "--sd", "random", "--ue", "--server_addr", server, "--disable_log_color", *tls]

        def on_line(line: str) -> None:
            m = GRADIO_SUCCESS_RE.search(line) or GRADIO_URL_RE.search(line)
            if m:
                url = m.group(1) if m.re is GRADIO_SUCCESS_RE else m.group(0)
                t.url = url if url.startswith("http") else f"https://{url}"
                t.status = "up"
                t.error = None
                self._changed(t)
            elif "login to server failed" in line:
                t.error = line.strip()[:300]

        await self._start_process(t, cmd, on_line)
        rc = await t.proc.wait()
        if t.status != "stopped":
            # Share links also expire after about 72 hours; the supervisor opens a new one.
            t.error = t.error or f"frpc exited ({rc}): " + " | ".join(list(t.proc.tail)[-3:])
            t.url = None
