"""GPU pool: GGUF planning, the FNV cache names, the RPC bridge and the helper endpoints."""

import asyncio
import struct
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from kindle.engines import llamacpp
from kindle.engines.context import DeployError
from kindle.pool import fnv
from kindle.pool.bridge import Bridge, serve_rpc, ws_url
from kindle.pool.gguf import layer_devices, read_header
from kindle.schemas import Settings
from kindle.server.app import create_app
from kindle.server.state import AppState

MiB = 1024 * 1024


def _s(text: str) -> bytes:
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def write_gguf(path, layers: int, rows: int = 4096, cols: int = 4096, kind: int = 1):
    """A header-only GGUF: real layout and offsets, no tensor data."""
    names = ["token_embd.weight"] + [f"blk.{i}.ffn_up.weight" for i in range(layers)] + ["output.weight"]
    kv = [(_s("general.architecture"), 8, _s("llama")), (_s("llama.block_count"), 4, struct.pack("<I", layers)),
          (_s("tokenizer.ggml.tokens"), 9, struct.pack("<IQ", 8, 3) + _s("a") + _s("b") + _s("c"))]
    out = b"GGUF" + struct.pack("<IQQ", 3, len(names), len(kv))
    for key, kind_id, value in kv:
        out += key + struct.pack("<I", kind_id) + value
    size = rows * cols * 2
    for i, name in enumerate(names):
        out += _s(name) + struct.pack("<I", 2) + struct.pack("<QQ", cols, rows) + struct.pack("<I", kind) + \
            struct.pack("<Q", i * size)
    path.write_bytes(out)
    return size


def test_gguf_header_and_layer_plan(tmp_path):
    path = tmp_path / "m.gguf"
    size = write_gguf(path, layers=4)
    h = read_header(path)
    assert h.arch == "llama" and h.block_count == 4
    blocks = [t for t in h.tensors if t.layer is not None]
    assert [t.layer for t in blocks] == [0, 1, 2, 3] and all(t.size == size for t in blocks)
    data_start = h.tensors[0].offset
    assert data_start % 32 == 0 and blocks[1].offset == data_start + 2 * size
    # llama.cpp's layer split: 5 slots (4 layers + output) over two equal GPUs
    assert layer_devices(4, [1, 1]) == [0, 0, 0, 1]
    assert layer_devices(4, [3, 1]) == [0, 0, 0, 0]


def test_seed_plan_sends_each_helper_its_layers(tmp_path):
    path = tmp_path / "m-00001-of-00001.gguf"
    write_gguf(path, layers=8)
    devices = [llamacpp.Device("linkA", "RTX", 90 << 30), llamacpp.Device("local", "RTX", 90 << 30)]
    split = llamacpp.plan_split(devices)
    assert split[0] > split[1]  # the head keeps more headroom
    plan = llamacpp.seed_plan([("Q4/m-00001-of-00001.gguf", read_header(path).tensors)], 8, split, devices)
    names = [t["name"] for t in plan["linkA"]]
    assert names and all(n.startswith("blk.") for n in names) and "blk.7.ffn_up.weight" not in names
    assert all(t["file"] == "Q4/m-00001-of-00001.gguf" and t["size"] > 10 * MiB for t in plan["linkA"])


def test_fnv_matches_reference_vectors(tmp_path):
    assert fnv.fnv1a_py(b"") == 0xCBF29CE484222325
    assert fnv.fnv1a_py(b"a") == 0xAF63DC4C8601EC8C
    assert fnv.fnv1a_py(b"foobar") == 0x85944171F73967E8
    h = fnv.Hasher(None)
    h.update(b"foo")
    h.update(b"bar")  # streaming equals one shot
    assert fnv.cache_name(h.value) == "85944171f73967e8"
    compiled = fnv.load(tmp_path)
    if compiled is None:
        pytest.skip("no C compiler")
    c = fnv.Hasher(compiled)
    c.update(b"foo")
    c.update(b"bar")
    assert c.value == h.value


def test_ws_url():
    assert ws_url("https://a.gradio.live") == "wss://a.gradio.live/_kui/pool/rpc"
    assert ws_url("http://127.0.0.1:8890") == "ws://127.0.0.1:8890/_kui/pool/rpc"


async def test_bridge_carries_tcp_through_a_websocket():
    import uvicorn

    async def echo(reader, writer):
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
        writer.close()

    rpc = await asyncio.start_server(echo, "127.0.0.1", 0)
    rpc_port = rpc.sockets[0].getsockname()[1]
    seen = {}
    app = FastAPI()

    @app.websocket("/_kui/pool/rpc")
    async def route(ws: WebSocket):
        seen["auth"] = ws.headers.get("authorization")
        await ws.accept()
        await serve_rpc(ws, rpc_port)

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", ws="websockets"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    bridge = Bridge(f"http://127.0.0.1:{port}", "kui-key", print)
    local = await bridge.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", local)
        payload = bytes(range(256)) * 8192  # 2 MiB: crosses chunk boundaries
        writer.write(payload)
        await writer.drain()
        got = await asyncio.wait_for(reader.readexactly(len(payload)), 20)
        assert got == payload and seen["auth"] == "Bearer kui-key"
        writer.close()
    finally:
        await bridge.close()
        server.should_exit = True
        await task
        rpc.close()


@pytest.fixture
def state(paths):
    s = Settings()
    s.exposure.provider = "none"
    return AppState(paths=paths, settings=s, port=8999, mock=True, fake_hw="t4x2")


def test_pool_endpoints_need_key_and_opt_in(state):
    key = {"Authorization": f"Bearer {state.auth.api_key}"}
    with TestClient(create_app(state)) as c:
        assert c.get("/_kui/pool/status").status_code == 401
        r = c.get("/_kui/pool/status", headers=key)
        assert r.status_code == 403 and "Share this GPU" in r.json()["detail"]
        state.settings.pool.share_gpu = True
        info = c.get("/_kui/pool/status", headers=key).json()
        assert info["enabled"] is True and info["status"] == "idle" and info["gpus"]
        assert c.post("/_kui/pool/prepare", json={"tag": "b1; rm -rf /"}, headers=key).status_code == 400
        with pytest.raises(WebSocketDisconnect):  # not ready: the socket is refused
            with c.websocket_connect("/_kui/pool/rpc", headers=key) as ws:
                ws.receive_bytes()


def test_pick_helpers_requires_sharing_sessions():
    sharing = SimpleNamespace(id="a", url="https://a", status="up", pool={"enabled": True})
    idle = SimpleNamespace(id="b", url="https://b", status="up", pool={"enabled": False})
    down = SimpleNamespace(id="c", url="https://c", status="down", pool={"enabled": True})
    ctx = SimpleNamespace(app=SimpleNamespace(links=SimpleNamespace(links={"a": sharing, "b": idle, "c": down})))
    assert llamacpp.pick_helpers(ctx, None) == []
    assert llamacpp.pick_helpers(ctx, "all") == [sharing]
    with pytest.raises(DeployError, match="sharing a GPU"):
        llamacpp.pick_helpers(ctx, ["a", "b"])
    with pytest.raises(DeployError, match="aren't connected"):
        llamacpp.pick_helpers(ctx, ["c"])
