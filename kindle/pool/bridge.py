"""Carry llama.cpp RPC (plain TCP) over a session's public HTTPS link as a WebSocket.

Hosted notebooks can't accept inbound TCP, and the RPC server must never be exposed anyway (it
has no authentication). So on the head, ``llama-server --rpc 127.0.0.1:<port>`` connects to a local
listener here, and each connection becomes an authenticated WebSocket to the helper's
``/_kui/pool/rpc``, which the helper pipes into its RPC server on 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Callable

from fastapi import WebSocket, WebSocketDisconnect

CHUNK = 1 << 20


def _nodelay(writer: asyncio.StreamWriter) -> None:
    sock = writer.get_extra_info("socket")
    if sock is not None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass


def ws_url(base: str) -> str:
    return ("wss://" + base[len("https://"):] if base.startswith("https://")
            else "ws://" + base[len("http://"):] if base.startswith("http://") else base) + "/_kui/pool/rpc"


class Bridge:
    """Head side: a localhost TCP port that tunnels to one helper."""

    def __init__(self, base_url: str, api_key: str, log: Callable[[str], None]):
        self.url = ws_url(base_url.rstrip("/"))
        self.api_key = api_key
        self.log = log
        self.server: asyncio.base_events.Server | None = None
        self.port = 0
        self._conns: set[asyncio.Task] = set()

    async def start(self) -> int:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self.port

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        from websockets.asyncio.client import connect

        from ..links import HOP_HEADER

        task = asyncio.current_task()
        if task:
            self._conns.add(task)
        _nodelay(writer)
        headers = {"Authorization": f"Bearer {self.api_key}", HOP_HEADER: "1"}
        try:
            async with connect(self.url, additional_headers=headers, max_size=None, compression=None,
                               ping_interval=20, ping_timeout=120, open_timeout=30) as ws:

                async def up() -> None:
                    while data := await reader.read(CHUNK):
                        await ws.send(data)
                    await ws.close()

                async def down() -> None:
                    async for message in ws:
                        writer.write(message if isinstance(message, bytes) else message.encode())
                        await writer.drain()

                done, pending = await asyncio.wait([asyncio.create_task(up()), asyncio.create_task(down())],
                                                   return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                for t in done:
                    if t.exception() and not isinstance(t.exception(), ConnectionError):
                        self.log(f"Pool link {self.url}: {type(t.exception()).__name__}: {t.exception()}")
        except Exception as exc:  # noqa: BLE001 - llama-server sees the dropped connection and reports it
            self.log(f"Pool link {self.url} failed: {type(exc).__name__}: {exc}")
        finally:
            writer.close()
            if task:
                self._conns.discard(task)

    async def close(self) -> None:
        if self.server:
            self.server.close()
        for t in list(self._conns):
            t.cancel()
        await asyncio.gather(*self._conns, return_exceptions=True)


async def serve_rpc(ws: WebSocket, port: int) -> None:
    """Helper side: pipe one accepted WebSocket into the local RPC server."""
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        await ws.close(code=1011, reason="RPC server isn't running")
        return
    _nodelay(writer)

    async def up() -> None:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            data = message.get("bytes") or (message.get("text") or "").encode()
            writer.write(data)
            await writer.drain()

    async def down() -> None:
        while data := await reader.read(CHUNK):
            await ws.send_bytes(data)

    tasks = [asyncio.create_task(up()), asyncio.create_task(down())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.close()
        try:
            await ws.close()
        except RuntimeError:
            pass
