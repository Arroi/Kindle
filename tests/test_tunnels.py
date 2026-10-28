"""A printed hostname must have a connected tunnel and resolve before it is shared."""

import asyncio
import json
from unittest.mock import AsyncMock

from kindle.schemas import Settings
from kindle.server.app import _open_main_tunnel
from kindle.server.state import AppState
from kindle.tunnels import manager


async def test_quick_tunnel_waits_for_registration(paths, monkeypatch):
    started = asyncio.Event()
    finish = asyncio.Event()

    class Process:
        def __init__(self, *args, on_line, **kwargs):
            self.on_line = on_line

        async def start(self):
            self.on_line("Your quick Tunnel has been created! https://example.trycloudflare.com")
            started.set()

        async def wait(self):
            await finish.wait()
            return 0

    monkeypatch.setattr(manager, "ensure_cloudflared", AsyncMock(return_value="cloudflared"))
    monkeypatch.setattr(manager, "ManagedProcess", Process)
    tunnels = manager.TunnelManager(paths)
    tunnel = manager.Tunnel("main", "cloudflare", 8999)
    task = asyncio.create_task(tunnels._run_cloudflare_quick(tunnel))
    await asyncio.wait_for(started.wait(), timeout=1)
    try:
        assert tunnel.url == "https://example.trycloudflare.com"
        assert tunnel.status == "starting"
        tunnel.proc.on_line("Registered tunnel connection connIndex=0")
        assert tunnel.status == "up"
    finally:
        tunnel.status = "stopped"
        finish.set()
        await task


async def test_gradio_tunnel_reports_share_url(paths, monkeypatch):
    started = asyncio.Event()
    finish = asyncio.Event()
    seen = {}

    class Process:
        def __init__(self, name, cmd, *args, on_line, **kwargs):
            seen["cmd"] = cmd
            self.on_line = on_line

        async def start(self):
            started.set()

        async def wait(self):
            await finish.wait()
            return 0

    monkeypatch.setattr(manager, "ensure_frpc", AsyncMock(return_value="frpc"))
    monkeypatch.setattr(manager.TunnelManager, "_gradio_relay",
                        AsyncMock(return_value=("relay.gradio.live:7000", ["--tls_enable"])))
    monkeypatch.setattr(manager, "ManagedProcess", Process)
    tunnels = manager.TunnelManager(paths)
    tunnel = manager.Tunnel("ui-1", "gradio", 8999)
    task = asyncio.create_task(tunnels._run_gradio(tunnel))
    await asyncio.wait_for(started.wait(), timeout=1)
    try:
        cmd = seen["cmd"]
        assert cmd[cmd.index("-l") + 1] == "8999"
        assert cmd[cmd.index("--server_addr") + 1] == "relay.gradio.live:7000"
        assert "--tls_enable" in cmd
        assert tunnel.url is None
        tunnel.proc.on_line("[I] [service.go:301] [abc] start proxy success: https://0123abcd.gradio.live")
        assert tunnel.url == "https://0123abcd.gradio.live"
        assert tunnel.status == "up"
    finally:
        tunnel.status = "stopped"
        finish.set()
        await task


def test_molab_defaults_to_gradio(monkeypatch):
    from kindle import config

    monkeypatch.delenv("KINDLE_TUNNEL", raising=False)
    monkeypatch.setattr(config, "ON_MOLAB", True)
    assert config.default_tunnel() == "gradio"


async def test_quick_tunnel_waits_for_dns(paths, monkeypatch):
    tunnels = manager.TunnelManager(paths)
    tunnels.tunnels["main"] = manager.Tunnel("main", "cloudflare", 8999,
                                            url="https://example.trycloudflare.com", status="up")
    lookup = AsyncMock(side_effect=[OSError("DNS isn't ready"), [(2, 1, 6, "", ("104.16.230.132", 443))]])
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", lookup)
    assert await tunnels.wait_url("main", timeout=2) == "https://example.trycloudflare.com"
    assert lookup.await_count == 2
    tunnels.tunnels["main"].status = "starting"
    assert await tunnels.wait_url("main", timeout=0.01) is None


async def test_failed_tunnel_does_not_publish_localhost(paths):
    settings = Settings()
    settings.exposure.provider = "none"
    state = AppState(paths, settings, port=8999, mock=True, fake_hw="t4x2")
    state.tunnels.wait_url = AsyncMock(return_value=None)
    try:
        assert await _open_main_tunnel(state) is False
        recorded = json.loads(paths.state_file.read_text())
        assert recorded["status"] == "tunnel_error"
        assert recorded["url"] is None
    finally:
        await state.catalog.close()
        await state.tunnels.close_all()
