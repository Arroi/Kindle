"""Studio chat over WebSocket (token streaming that also works through Cloudflare quick tunnels)."""

from __future__ import annotations

import asyncio
import time

from fastapi import WebSocket, WebSocketDisconnect

from ..gateway.common import wrap
from .state import AppState


async def studio_ws(ws: WebSocket, state: AppState) -> None:
    if not state.auth.has_access(ws, state.settings):  # type: ignore[arg-type]
        await ws.close(code=4401)
        return
    await ws.accept()
    tasks: dict[str, asyncio.Task] = {}
    send_lock = asyncio.Lock()

    async def send(msg: dict) -> None:
        async with send_lock:
            await ws.send_json(msg)

    async def run_chat(req_id: str, msg: dict) -> None:
        started = time.time()
        first: float | None = None
        tokens = 0
        usage: dict = {}
        finish = None
        try:
            if state.updates.blocked:
                raise RuntimeError(state.updates.status()["message"])
            dep = state.manager.resolve(msg.get("model"), "text")
            if not dep or not dep.backend:
                raise RuntimeError("That model isn't running")
            body = {"messages": msg.get("messages") or [], **(msg.get("params") or {})}
            async for chunk in dep.backend.chat_stream(body):
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    reasoning = delta.get("reasoning") or delta.get("reasoning_content") or delta.get("thinking")
                    if content or reasoning:
                        if first is None:
                            first = time.time()
                        tokens += 1
                        out = {"op": "delta", "id": req_id}
                        if content:
                            out["content"] = content
                        if reasoning:
                            out["reasoning"] = reasoning
                        await send(out)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
            elapsed = time.time() - started
            gen_time = time.time() - (first or started)
            out_tokens = usage.get("completion_tokens") or tokens
            await send({"op": "done", "id": req_id, "finish_reason": finish or "stop", "usage": usage,
                        "elapsed": round(elapsed, 2), "ttft": round((first or time.time()) - started, 2),
                        "tps": round(out_tokens / gen_time, 1) if gen_time > 0 else None})
        except asyncio.CancelledError:
            await send({"op": "done", "id": req_id, "finish_reason": "cancelled"})
        except Exception as e:  # noqa: BLE001
            await send({"op": "error", "id": req_id, "message": wrap(e).message})
        finally:
            tasks.pop(req_id, None)

    try:
        while True:
            msg = await ws.receive_json()
            op, req_id = msg.get("op"), str(msg.get("id") or "")
            if op == "chat" and req_id:
                tasks[req_id] = asyncio.create_task(run_chat(req_id, msg))
            elif op == "cancel" and req_id in tasks:
                tasks[req_id].cancel()
            elif op == "ping":
                await send({"op": "pong"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        pending = list(tasks.values())
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
