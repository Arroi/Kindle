"""Authenticating reverse proxy (HTTP + WebSocket).

Used two ways:
* path-prefixed inside the main app (``/gradio/<id>/…`` → a Gradio process), and
* as a standalone root-level app on its own port + tunnel for UIs that can't live under a path
  prefix (Open WebUI, ComfyUI). That app shows a small login page when the access password is on.
"""

from __future__ import annotations

import asyncio
import html
from typing import TYPE_CHECKING

import httpx
import websockets
from fastapi import FastAPI, Form, Request, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

if TYPE_CHECKING:
    from .state import AppState

HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
               "transfer-encoding", "upgrade", "host", "content-length"}
_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=None, write=120, pool=30), follow_redirects=False)


async def proxy_http(request: Request, target_base: str, path: str) -> Response:
    url = f"{target_base.rstrip('/')}/{path.lstrip('/')}"
    if request.url.query:
        url += "?" + request.url.query
    headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP_HEADERS]
    headers.append(("x-forwarded-proto", request.headers.get("x-forwarded-proto", request.url.scheme)))
    req = _client.build_request(request.method, url, headers=headers, content=request.stream())
    try:
        resp = await _client.send(req, stream=True)
    except httpx.HTTPError as e:
        return Response(f"Upstream unavailable: {e}", status_code=502)
    out_headers = [(k, v) for k, v in resp.headers.multi_items() if k.lower() not in HOP_HEADERS]
    response = StreamingResponse(resp.aiter_raw(), status_code=resp.status_code, background=BackgroundTask(resp.aclose))
    response.raw_headers = [(k.lower().encode(), v.encode()) for k, v in out_headers]
    return response


async def proxy_ws(ws: WebSocket, target_url: str) -> None:
    protocols = [p.strip() for p in ws.headers.get("sec-websocket-protocol", "").split(",") if p.strip()]
    passthrough = {k: v for k, v in ws.headers.items() if k.lower() in ("cookie", "origin", "user-agent")}
    try:
        upstream = await websockets.connect(target_url, subprotocols=protocols or None, additional_headers=passthrough,
                                            max_size=None, open_timeout=20)
    except Exception:
        await ws.close(code=1011)
        return
    await ws.accept(subprotocol=upstream.subprotocol)

    async def client_to_upstream() -> None:
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    await upstream.send(msg["text"])
                elif msg.get("bytes") is not None:
                    await upstream.send(msg["bytes"])
        finally:
            await upstream.close()

    async def upstream_to_client() -> None:
        try:
            async for data in upstream:
                if isinstance(data, bytes):
                    await ws.send_bytes(data)
                else:
                    await ws.send_text(data)
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    await asyncio.gather(client_to_upstream(), upstream_to_client(), return_exceptions=True)


LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Kindle</title><style>
:root{{color-scheme:light dark;--bg:#f6f7f9;--card:#fff;--fg:#14171c;--muted:#5d6672;--line:#dfe3e8;--accent:#1f6feb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d1014;--card:#151a21;--fg:#e8ecf1;--muted:#98a2ae;--line:#26303b;--accent:#4c8dff}}}}
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
form{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:28px;width:min(360px,calc(100vw - 32px));box-sizing:border-box}}
h1{{font-size:18px;margin:0 0 4px}}p{{color:var(--muted);margin:0 0 18px;font-size:14px}}
input{{width:100%;box-sizing:border-box;padding:10px 12px;border-radius:8px;border:1px solid var(--line);background:transparent;color:inherit;font:inherit}}
button{{margin-top:12px;width:100%;padding:10px;border:0;border-radius:8px;background:var(--accent);color:#fff;font:inherit;font-weight:600;cursor:pointer}}
.err{{color:#d1242f;font-size:13px;margin-top:10px}}</style></head><body>
<form method="post" action="/_kui/login"><h1>{title}</h1><p>This interface is password-locked. Ask the host for the password.</p>
<input type="password" name="password" placeholder="Password" autofocus autocomplete="current-password">
<button type="submit">Unlock</button>{error}</form></body></html>"""


def make_root_proxy(state: AppState, target_port: int, title: str) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    target = f"http://127.0.0.1:{target_port}"

    @app.get("/_kui/login")
    async def login_page(error: str = "") -> HTMLResponse:
        err = f'<div class="err">{html.escape(error)}</div>' if error else ""
        return HTMLResponse(LOGIN_PAGE.format(title=html.escape(title), error=err))

    @app.post("/_kui/login")
    async def login(request: Request, password: str = Form("")) -> Response:
        ip = state.auth.client_ip(request)
        if state.auth.is_rate_limited(ip):
            return RedirectResponse("/_kui/login?error=Too+many+attempts", status_code=303)
        if not state.auth.check_password(password):
            state.auth.record_failure(ip)
            return RedirectResponse("/_kui/login?error=Wrong+password", status_code=303)
        resp = RedirectResponse("/", status_code=303)
        state.auth.set_access_cookie(request, resp)
        return resp

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def forward(path: str, request: Request) -> Response:
        if not state.auth.has_access(request, state.settings):
            if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
                return RedirectResponse("/_kui/login", status_code=303)
            return Response("Locked", status_code=401)
        return await proxy_http(request, target, path)

    @app.websocket("/{path:path}")
    async def forward_ws(ws: WebSocket, path: str) -> None:
        if not state.auth.has_access(ws, state.settings):  # type: ignore[arg-type]
            await ws.close(code=4401)
            return
        query = ("?" + ws.url.query) if ws.url.query else ""
        await proxy_ws(ws, f"ws://127.0.0.1:{target_port}/{path}{query}")

    return app
