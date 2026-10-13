"""The KaggleUI server: control panel API, Studio, API gateway, Gradio proxy and the static SPA."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError

from .. import __version__, notify, secretstore
from ..catalog.resolve import ENGINE_LABELS, MODALITY_ENGINES
from ..engines.world_engine import list_recipes
from ..gateway import anthropic as anthropic_gw
from ..gateway import ollama_compat as ollama_gw
from ..gateway import openai as openai_gw
from ..schemas import MODALITIES, DeploymentSpec, Preset, SettingsPatch
from ..secretstore import secret_status
from ..tunnels.manager import PROVIDER_LABELS
from .auth import ACCESS_COOKIE, ADMIN_COOKIE
from .proxy import proxy_http, proxy_ws
from .state import AppState
from .studio import studio_ws

STATIC = Path(__file__).resolve().parents[1] / "static"
RESERVED = ("/v1/", "/api/", "/_kui/", "/gradio/")
INSTALL_LINE = "!curl -fsSL https://raw.githubusercontent.com/UltraFEmotes/KaggleUI/main/install.sh | bash"


def create_app(state: AppState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.bus.bind_loop(asyncio.get_running_loop())
        tasks = [asyncio.create_task(_metrics_loop(state)), asyncio.create_task(_startup(state))]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await state.manager.shutdown()
            from ..engines.ollama import OllamaServer

            await OllamaServer.get().stop()  # only stops an Ollama we started ourselves
            await state.tunnels.close_all()
            await state.catalog.close()
            try:
                _write_state(state, "stopped")
            except Exception:
                pass

    app = FastAPI(title="KaggleUI", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.kui = state

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    app.include_router(openai_gw.router)
    app.include_router(anthropic_gw.router)
    app.include_router(ollama_gw.router)

    # =========================================================================================
    # Session / auth
    # =========================================================================================
    def require_admin(request: Request) -> None:
        if not state.auth.is_admin(request):
            raise HTTPException(401, "Admin token required")

    def require_access(request: Request) -> None:
        if not state.auth.has_access(request, state.settings):
            raise HTTPException(401, "Password required")

    @app.get("/_kui/api/session")
    async def session(request: Request):
        return {"admin": state.auth.is_admin(request), "access": state.auth.has_access(request, state.settings),
                "password_enabled": state.settings.access.password_enabled, "version": __version__,
                "mock": state.mock, "fake_hw": state.fake_hw}

    @app.post("/_kui/api/login")
    async def login(request: Request, body: dict = Body(...)):
        ip = state.auth.client_ip(request)
        if state.auth.is_rate_limited(ip):
            raise HTTPException(429, "Too many attempts. Wait a few minutes.")
        if not state.auth.check_admin_token(str(body.get("token", "")).strip()):
            state.auth.record_failure(ip)
            raise HTTPException(401, "Wrong admin token")
        resp = JSONResponse({"ok": True})
        state.auth.set_admin_cookie(request, resp)
        return resp

    @app.post("/_kui/api/access-login")
    async def access_login(request: Request, body: dict = Body(...)):
        ip = state.auth.client_ip(request)
        if state.auth.is_rate_limited(ip):
            raise HTTPException(429, "Too many attempts. Wait a few minutes.")
        if not state.auth.check_password(str(body.get("password", "")).strip()):
            state.auth.record_failure(ip)
            raise HTTPException(401, "Wrong password")
        resp = JSONResponse({"ok": True})
        state.auth.set_access_cookie(request, resp)
        return resp

    @app.post("/_kui/api/logout")
    async def logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(ADMIN_COOKIE, path="/")
        resp.delete_cookie(ACCESS_COOKIE, path="/")
        return resp

    # =========================================================================================
    # Panel (admin)
    # =========================================================================================
    def overview() -> dict:
        main = state.tunnels.get("main")
        return {
            "version": __version__, "mock": state.mock, "fake_hw": state.fake_hw,
            "system": state.probe.snapshot().to_dict(),
            "settings": state.settings.model_dump(),
            "credentials": {"admin_token": state.auth.admin_token, "password": state.auth.password,
                            "api_key": state.auth.api_key},
            "secrets": secret_status(),
            "tunnels": state.tunnels.list(),
            "main_url": state.main_url(),
            "streaming_ok": bool(main and main.supports_sse),
            "deployments": state.manager.list(),
            "started_at": state.started_at,
        }

    @app.get("/_kui/api/overview")
    async def get_overview(request: Request):
        require_admin(request)
        return overview()

    @app.get("/_kui/api/system")
    async def system(request: Request):
        require_admin(request)
        return state.probe.snapshot().to_dict()

    @app.put("/_kui/api/settings")
    async def update_settings(request: Request, patch: SettingsPatch):
        require_admin(request)
        before = state.settings.exposure.provider
        data = state.settings.model_dump()
        for section in ("access", "api", "exposure"):
            changes = getattr(patch, section)
            if changes:
                data[section].update(changes)
        try:
            from ..schemas import Settings

            state.settings = Settings.model_validate(data)
        except ValidationError as e:
            raise HTTPException(422, str(e)) from e
        if state.settings.exposure.provider != before:
            asyncio.create_task(_open_main_tunnel(state))
        state.bus.publish("settings", state.settings.model_dump())
        return state.settings.model_dump()

    @app.post("/_kui/api/credentials/regenerate")
    async def regenerate(request: Request, body: dict = Body(...)):
        require_admin(request)
        which = body.get("which")
        if which not in ("password", "api_key"):
            raise HTTPException(400, "which must be 'password' or 'api_key'")
        state.auth.regenerate(which)
        _write_state(state)
        return {"admin_token": state.auth.admin_token, "password": state.auth.password, "api_key": state.auth.api_key}

    # -- secrets (keys & tokens) ----------------------------------------------------------------
    async def hf_identity() -> dict:
        token = os.environ.get("HF_TOKEN")
        if not token:
            return {"ok": False, "user": None, "error": None}
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {token}"})
            if r.status_code == 200:
                return {"ok": True, "user": r.json().get("name"), "error": None}
            return {"ok": False, "user": None, "error": "Hugging Face rejected this token" if r.status_code == 401
                    else f"Hugging Face returned {r.status_code}"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "user": None, "error": f"Couldn't reach Hugging Face: {type(e).__name__}"}

    def secrets_payload(hf: dict | None = None) -> dict:
        data = secretstore.secret_details()
        if hf is not None:
            data["hf"] = hf
        return data

    def secrets_changed() -> None:
        state.bus.publish("secrets", secret_status())

    @app.get("/_kui/api/secrets")
    async def get_secrets(request: Request):
        require_admin(request)
        return secrets_payload(await hf_identity())

    @app.put("/_kui/api/secrets/{name}")
    async def put_secret(request: Request, name: str, body: dict = Body(...)):
        require_admin(request)
        value = str(body.get("value") or "").strip()
        if name not in secretstore.EDITABLE:
            raise HTTPException(400, "That secret can't be set here")
        if not value:
            raise HTTPException(400, "Empty value")
        secretstore.set_secret(name, value)
        secrets_changed()
        hf = await hf_identity() if name == "HF_TOKEN" else None
        return secrets_payload(hf)

    @app.delete("/_kui/api/secrets/{name}")
    async def delete_secret(request: Request, name: str):
        require_admin(request)
        if name not in secretstore.EDITABLE:
            raise HTTPException(400, "That secret can't be removed here")
        secretstore.clear_secret(name)
        secrets_changed()
        return secrets_payload()

    @app.post("/_kui/api/secrets/reload")
    async def reload_secrets(request: Request):
        require_admin(request)
        before = secret_status()
        await asyncio.to_thread(secretstore.load_kaggle_secrets, True)
        after = secret_status()
        secrets_changed()
        data = secrets_payload(await hf_identity())
        data["added"] = [n for n in after if after[n] and not before[n]]
        return data

    @app.post("/_kui/api/secrets/import")
    async def import_secrets(request: Request, body: dict = Body(...)):
        """Used by launch() to hand over Kaggle secrets read in the notebook kernel."""
        require_admin(request)
        updated = []
        for name, value in (body.get("values") or {}).items():
            if name in secretstore.EDITABLE and value and os.environ.get(name) != value:
                secretstore.set_secret(name, str(value), source="kaggle")
                updated.append(name)
        if updated:
            secrets_changed()
        return {"updated": updated}

    @app.get("/_kui/api/catalog/meta")
    async def catalog_meta(request: Request):
        require_admin(request)
        return {"modalities": list(MODALITIES), "engines": MODALITY_ENGINES, "engine_labels": ENGINE_LABELS,
                "recipes": [{"name": r["name"], "title": r["title"], "repo": r["repo"], "controls": r.get("controls"),
                             "unverified": bool(r.get("unverified"))} for r in list_recipes()],
                "tunnel_providers": PROVIDER_LABELS}

    @app.get("/_kui/api/catalog/search")
    async def catalog_search(request: Request, modality: str = "text", q: str = "", source: str = "all",
                             sort: str = "recommended", limit: int = 40, show_unsupported: bool = False,
                             fit: str = "all"):
        require_admin(request)
        if modality not in MODALITIES:
            raise HTTPException(400, "unknown modality")
        return await state.catalog.search(modality, q, source, sort, max(1, min(limit, 300)), show_unsupported,
                                          "fits" if fit == "fits" else "all")

    @app.get("/_kui/api/catalog/detail")
    async def catalog_detail(request: Request, key: str):
        require_admin(request)
        try:
            return await state.catalog.detail(key)
        except KeyError as e:
            raise HTTPException(404, "Unknown model") from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"Couldn't load model details: {e}") from e

    @app.get("/_kui/api/deployments")
    async def list_deployments(request: Request):
        require_admin(request)
        return state.manager.list()

    @app.post("/_kui/api/deployments")
    async def create_deployment(request: Request, spec: DeploymentSpec):
        require_admin(request)
        dep = state.manager.create(spec)
        return dep.public()

    @app.delete("/_kui/api/deployments/{dep_id}")
    async def delete_deployment(request: Request, dep_id: str):
        require_admin(request)
        if not state.manager.get(dep_id):
            raise HTTPException(404, "No such deployment")
        await state.manager.stop(dep_id)
        return {"ok": True}

    @app.post("/_kui/api/deployments/{dep_id}/restart")
    async def restart_deployment(request: Request, dep_id: str):
        require_admin(request)
        dep = await state.manager.restart(dep_id)
        if not dep:
            raise HTTPException(404, "No such deployment")
        return dep.public()

    @app.get("/_kui/api/deployments/{dep_id}/logs")
    async def deployment_logs(request: Request, dep_id: str):
        require_admin(request)
        dep = state.manager.get(dep_id)
        if not dep:
            raise HTTPException(404, "No such deployment")
        return {"lines": list(dep.logs)}

    @app.get("/_kui/api/tunnels")
    async def tunnels(request: Request):
        require_admin(request)
        return state.tunnels.list()

    @app.post("/_kui/api/tunnels/main/restart")
    async def restart_tunnel(request: Request):
        require_admin(request)
        asyncio.create_task(_open_main_tunnel(state))
        return {"ok": True}

    @app.get("/_kui/api/preset")
    async def preset(request: Request):
        require_admin(request)
        specs = []
        for d in state.manager.deployments.values():
            if d.status in ("ready", "queued", "installing", "downloading", "loading"):
                specs.append(d.spec)
        p = Preset(settings=state.settings, deployments=specs)
        data = p.model_dump(exclude_defaults=False)
        cell = (f"{INSTALL_LINE}\n\nimport kaggleui\nkaggleui.launch(preset={json.dumps(data)})\n")
        return {"preset": data, "cell": cell}

    @app.websocket("/_kui/ws")
    async def events(ws: WebSocket):
        if not state.auth.is_admin(ws):  # type: ignore[arg-type]
            await ws.close(code=4401)
            return
        await ws.accept()
        q = state.bus.subscribe()
        try:
            await ws.send_json({"type": "overview", "data": overview(), "ts": time.time()})
            while True:
                event = await q.get()
                await ws.send_json(event)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            state.bus.unsubscribe(q)

    # =========================================================================================
    # Studio (access password)
    # =========================================================================================
    @app.get("/_kui/api/studio/models")
    async def studio_models(request: Request):
        require_access(request)
        out = []
        for d in state.manager.deployments.values():
            out.append({"id": d.id, "name": d.served_name, "modality": d.spec.modality, "status": d.status,
                        "message": d.message, "progress": d.progress, "capabilities": d.info.get("capabilities", []),
                        "controls": d.info.get("controls"), "options": d.spec.options,
                        "interface": d.interface, "engine": d.spec.model.engine, "runtime": d.info.get("runtime")})
        return out

    @app.websocket("/_kui/studio/ws")
    async def studio(ws: WebSocket):
        await studio_ws(ws, state)

    # =========================================================================================
    # Gradio (path-prefixed proxy)
    # =========================================================================================
    def gradio_target(dep_id: str) -> int:
        dep = state.manager.get(dep_id)
        port = (dep.interface or {}).get("local_port") if dep else None
        if not port or dep.spec.interface != "gradio":
            raise HTTPException(404, "No Gradio app for that deployment")
        return int(port)

    @app.api_route("/gradio/{dep_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
    async def gradio_http(request: Request, dep_id: str, path: str):
        if not state.auth.has_access(request, state.settings):
            if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
                return RedirectResponse(f"/login?next=/gradio/{dep_id}/", status_code=303)
            raise HTTPException(401, "Password required")
        return await proxy_http(request, f"http://127.0.0.1:{gradio_target(dep_id)}", path)

    @app.get("/gradio/{dep_id}")
    async def gradio_root(dep_id: str):
        return RedirectResponse(f"/gradio/{dep_id}/")

    @app.websocket("/gradio/{dep_id}/{path:path}")
    async def gradio_ws(ws: WebSocket, dep_id: str, path: str):
        if not state.auth.has_access(ws, state.settings):  # type: ignore[arg-type]
            await ws.close(code=4401)
            return
        query = ("?" + ws.url.query) if ws.url.query else ""
        await proxy_ws(ws, f"ws://127.0.0.1:{gradio_target(dep_id)}/{path}{query}")

    @app.api_route("/_kui/{path:path}", methods=["POST", "PUT", "PATCH", "DELETE"])
    async def unknown_panel_call(path: str):
        # Without this, a page newer than the server gets a confusing "405 Method Not Allowed".
        raise HTTPException(404, f"This KaggleUI server ({__version__}) doesn't have /_kui/{path}. It's probably "
                                 "older than this page: in the notebook, re-run the KaggleUI cell (it restarts the server).")

    # =========================================================================================
    # Static SPA
    # =========================================================================================
    @app.get("/{path:path}")
    async def spa(request: Request, path: str):
        token = request.query_params.get("token")
        if token is not None:
            target = "/" + path
            if state.auth.check_admin_token(token):
                resp = RedirectResponse(target, status_code=303)
                state.auth.set_admin_cookie(request, resp)
                return resp
            state.auth.record_failure(state.auth.client_ip(request))
            return RedirectResponse(target + "?bad_token=1", status_code=303)
        if ("/" + path).startswith(RESERVED):
            raise HTTPException(404)
        candidate = (STATIC / path).resolve()
        if path and candidate.is_file() and STATIC.resolve() in candidate.parents:
            headers = {"Cache-Control": "public, max-age=31536000, immutable"} if path.startswith("assets/") else {}
            return FileResponse(candidate, headers=headers)
        index = STATIC / "index.html"
        if index.exists():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>KaggleUI</h1><p>The web UI isn't built. Run <code>pnpm build</code> in <code>web/</code>.</p>")

    return app


# =============================================================================================
# Background tasks
# =============================================================================================

async def _metrics_loop(state: AppState) -> None:
    while True:
        await asyncio.sleep(2)
        if not state.bus._subs:
            continue
        try:
            snap = await asyncio.to_thread(state.probe.snapshot)
            state.bus.publish("metrics", snap.to_dict())
        except Exception:
            pass


def _write_state(state: AppState, status: str = "running") -> None:
    data = {
        "pid": os.getpid(), "version": __version__, "port": state.port, "status": status, "url": state.main_url(),
        "tunnel": (state.tunnels.get("main").info() if state.tunnels.get("main") else None),
        "admin_token": state.auth.admin_token,
        "password": state.auth.password if state.settings.access.password_enabled else None,
        "api_key": state.auth.api_key if state.settings.api.enabled and state.settings.api.require_key else None,
        "updated": time.time(),
    }
    path = state.paths.state_file
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


async def _open_main_tunnel(state: AppState) -> None:
    provider = state.settings.exposure.provider
    await state.tunnels.open("main", state.port, provider)
    url = await state.tunnels.wait_url("main", timeout=90)
    _write_state(state)
    if url and provider != "none":
        result = await notify.push(url, state.auth.admin_token)
        if result:
            print(f"[kaggleui] notification {result}", flush=True)


async def _startup(state: AppState) -> None:
    _write_state(state, "starting")
    await _open_main_tunnel(state)

    last_url = state.main_url()

    def on_tunnel(t) -> None:
        nonlocal last_url
        state.bus.publish("tunnel", t.info())
        if t.name == "main" and t.url and t.url != last_url:
            last_url = t.url
            _write_state(state)
            print(f"[kaggleui] public URL changed: {t.url}", flush=True)
            asyncio.create_task(notify.push(t.url, state.auth.admin_token))

    state.tunnels.on_change = on_tunnel
    if state.preset:
        for spec in state.preset.deployments:
            state.manager.create(spec)
    print(notify.card_text(state.main_url(), state.auth.admin_token,
                           state.auth.password if state.settings.access.password_enabled else None,
                           state.auth.api_key if state.settings.api.enabled else None), flush=True)


def response_json(data) -> Response:
    return JSONResponse(data)
