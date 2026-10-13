"""In-process pub/sub that fans events out to control-panel WebSockets.

WebSockets (not SSE) because Cloudflare quick tunnels don't support Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, type_: str, data: Any) -> None:
        event = {"type": type_, "data": data, "ts": time.time()}
        if self._loop is not None:
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is not self._loop:
                self._loop.call_soon_threadsafe(self._fanout, event)
                return
        self._fanout(event)

    def _fanout(self, event: dict) -> None:
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
