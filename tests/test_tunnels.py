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
    monkeypatch.delenv("BURROW_TOKEN", raising=False)
    monkeypatch.delenv("SHAYEF_TOKEN", raising=False)
    monkeypatch.setattr(config, "ON_MOLAB", True)
    assert config.default_tunnel() == "gradio"
    monkeypatch.setenv("BURROW_TOKEN", "eyJ-key")
    assert config.default_tunnel() == "gradio"  # the hostname is needed too
    monkeypatch.setenv("BURROW_HOSTNAME", "ai.example.com")
    assert config.default_tunnel() == "burrow"
    monkeypatch.setenv("SHAYEF_TOKEN", "shy-token")
    assert config.default_tunnel() == "shayef"  # preferred over burrow when both are set


async def test_shayef_is_gated_by_kindles_api_key(paths, monkeypatch):
    started = asyncio.Event()
    finish = asyncio.Event()
    seen = {}

    class Process:
        def __init__(self, paths, cmd, *, on_line, env=None, **kwargs):
            seen["cmd"], seen["env"] = cmd, env
            self.on_line = on_line
            self.tail = []

        def adopt(self):
            return None

        async def start(self):
            started.set()

        async def wait(self):
            await finish.wait()
            return 0

    monkeypatch.setenv("SHAYEF_TOKEN", "shy-token")
    monkeypatch.setenv("SHAYEF_NAME", "my-kindle")
    monkeypatch.setattr(manager, "ensure_cloudflared", AsyncMock(return_value="/k/bin/cloudflared"))
    monkeypatch.setattr(manager, "PersistentProcess", Process)
    tunnels = manager.TunnelManager(paths, shayef_key=lambda: "kui-key")
    tunnel = manager.Tunnel("main", "shayef", 8890, persistent=True)
    task = asyncio.create_task(tunnels._run_shayef(tunnel))
    await asyncio.wait_for(started.wait(), timeout=1)
    try:
        cmd = seen["cmd"]
        assert cmd[1:7] == ["-u", "-m", "kindle._vendor.shayef", "up", "8890", "my-kindle"]
        assert cmd[cmd.index("--access") + 1] == "key"
        assert seen["env"]["SHAYEF_KEY"] == "kui-key"  # API clients pass the gate with the key they already send
        assert "shy-token" not in " ".join(cmd) and "kui-key" not in " ".join(cmd)  # never where ps shows it
        assert seen["env"]["SHAYEF_RELAY"] == "/k/bin/cloudflared"
        assert tunnel.info()["link_key"] == "kui-key"
        assert tunnel.url is None and tunnel.status == "starting"
        tunnel.proc.on_line("  - hostname ready reused=False url=https://my-kindle.shayef.me")
        assert tunnel.url == "https://my-kindle.shayef.me" and tunnel.status == "starting"
        tunnel.proc.on_line("  - tunnel is live url=https://my-kindle.shayef.me")
        assert tunnel.status == "up"
        tunnel.proc.on_line("  error: my-kindle.shayef.me belongs to another client.")
        tunnel.proc.on_line("  hint:  Pick another SHAYEF_NAME.")
        assert tunnel.error == "my-kindle.shayef.me belongs to another client. Pick another SHAYEF_NAME."
        tunnel.proc.on_line("  x the edge rejected the relay detail=Unauthorized: invalid tunnel token")
        assert tunnel.error == "Unauthorized: invalid tunnel token"
    finally:
        tunnel.status = "stopped"
        finish.set()
        await task


async def test_shayef_link_is_shared_only_once_it_reaches_kindle(paths, monkeypatch):
    tunnels = manager.TunnelManager(paths)
    tunnels.tunnels["main"] = manager.Tunnel("main", "shayef", 8890, url="https://kindle.shayef.me", status="up")
    errors = []
    reaches = AsyncMock(side_effect=lambda t: errors.append(t.error) or (None if len(errors) > 1 else "ConnectError"))
    monkeypatch.setattr(tunnels, "_reaches_kindle", reaches)
    assert await tunnels.wait_url("main", timeout=5) == "https://kindle.shayef.me"
    assert errors[1] == "shayef is connected, but https://kindle.shayef.me doesn't reach Kindle (ConnectError)."
    assert tunnels.tunnels["main"].error is None


async def test_burrow_runs_its_agent_with_the_key(paths, monkeypatch):
    started = asyncio.Event()
    finish = asyncio.Event()
    seen = {}

    class Process:
        def __init__(self, paths, cmd, *, on_line, env=None, **kwargs):
            seen["cmd"], seen["env"] = cmd, env
            self.on_line = on_line

        def adopt(self):
            return None

        async def start(self):
            started.set()

        async def wait(self):
            await finish.wait()
            return 0

    monkeypatch.setenv("BURROW_TOKEN", "eyJ-key")
    monkeypatch.setenv("BURROW_HOSTNAME", "ai.example.com")
    monkeypatch.setattr(manager, "ensure_burrow", AsyncMock(return_value="burrow-core"))
    monkeypatch.setattr(manager, "PersistentProcess", Process)
    tunnels = manager.TunnelManager(paths)
    tunnel = manager.Tunnel("main", "burrow", 8890, persistent=True)
    task = asyncio.create_task(tunnels._run_burrow(tunnel))
    await asyncio.wait_for(started.wait(), timeout=1)
    try:
        assert seen["cmd"] == ["burrow-core", "tunnel", "--no-autoupdate", "run"]
        assert seen["env"] == {"TUNNEL_TOKEN": "eyJ-key"}  # never on the command line, where ps shows it
        assert tunnel.url == "https://ai.example.com" and tunnel.status == "starting"
        tunnel.proc.on_line('INF Updated to new configuration config="{\\"ingress\\":[{\\"hostname\\":'
                            '\\"ai.example.com\\", \\"service\\":\\"http://127.0.0.1:3000\\"}]}" version=1')
        assert "http://127.0.0.1:8890" in tunnel.error
        tunnel.proc.on_line("INF Registered tunnel connection connIndex=0 location=fra08 protocol=quic")
        assert tunnel.status == "up"
    finally:
        tunnel.status = "stopped"
        finish.set()
        await task


async def test_burrow_link_is_shared_only_once_it_reaches_kindle(paths, monkeypatch):
    tunnels = manager.TunnelManager(paths)
    tunnels.tunnels["main"] = manager.Tunnel("main", "burrow", 8890, url="https://ai.example.com", status="up")
    errors = []
    reaches = AsyncMock(side_effect=lambda t: errors.append(t.error) or (None if len(errors) > 1 else "HTTP 502"))
    monkeypatch.setattr(tunnels, "_reaches_kindle", reaches)
    assert await tunnels.wait_url("main", timeout=5) == "https://ai.example.com"
    assert reaches.await_count == 2
    assert "doesn't reach Kindle (HTTP 502)" in errors[1]  # what the notebook prints if it never does
    assert tunnels.tunnels["main"].error is None


def test_a_new_api_key_is_a_new_shayef_binding(monkeypatch):
    from kindle.tunnels.persistent import binding

    monkeypatch.setenv("SHAYEF_TOKEN", "shy-token")
    assert binding("shayef", 8890, "old-key") != binding("shayef", 8890, "new-key")
    assert binding("gradio", 8890) == binding("gradio", 8890, "")


def test_panel_link_carries_a_gated_tunnels_key():
    from kindle.notify import panel_link

    assert panel_link("https://kindle.shayef.me/", "adm") == "https://kindle.shayef.me/?token=adm"
    assert panel_link("https://kindle.shayef.me", "adm", "kui-key") == "https://kindle.shayef.me/?k=kui-key&token=adm"


async def test_burrow_side_tunnels_use_gradio(paths, monkeypatch):
    async def idle(self, t):
        return None

    monkeypatch.setattr(manager.TunnelManager, "_supervise", idle)
    tunnels = manager.TunnelManager(paths)
    assert (await tunnels.open("ui-1", 9001, "burrow")).provider == "gradio"  # burrow's domain maps to one port
    assert (await tunnels.open("ui-2", 9002, "shayef")).provider == "gradio"  # and so does a shayef name
    await tunnels.close_all()


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
