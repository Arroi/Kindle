"""Gateway auth + model lookup shared by the OpenAI, Anthropic and Ollama-compatible routers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from ..engines.base import Unreachable, Unsupported, UpstreamError
from ..links import HOP_HEADER

if TYPE_CHECKING:
    from ..deploy.manager import Deployment
    from ..server.state import AppState

T = TypeVar("T")

MODALITY_LABEL = {"text": "text model", "image": "image model", "video": "video model",
                  "tts": "text-to-speech model", "audio": "audio generation model", "mesh": "3D model",
                  "stt": "speech-to-text model", "world": "world model"}


class GatewayError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error", code: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind
        self.code = code


def state_of(request: Request) -> AppState:
    return request.app.state.kui


def check_access(request: Request, surface: str) -> None:
    """Allow internal callers, panel admins, Studio users (access cookie), or valid API keys."""
    st = state_of(request)
    if st.updates.blocked:
        raise GatewayError(426, st.updates.status()["message"], "upgrade_required")
    auth = st.auth
    if auth.is_internal(request) or auth.is_admin(request):
        return
    if request.headers.get("x-kui-studio") == "1" and auth.has_access(request, st.settings):
        return
    api = st.settings.api
    if not api.enabled or not getattr(api, surface, False):
        raise GatewayError(403, f"The {surface} API is turned off in Kindle (API settings).", "permission_error")
    if api.require_key and not auth.check_api_key(request):
        ip = auth.client_ip(request)
        if auth.is_rate_limited(ip):
            raise GatewayError(429, "Too many failed attempts. Try again in a few minutes.", "rate_limit_error")
        auth.record_failure(ip)
        raise GatewayError(401, "Invalid or missing API key. Send it as 'Authorization: Bearer <key>' or 'x-api-key'.",
                           "authentication_error", "invalid_api_key")


def find(request: Request, model: str | None, modality: str, exclude: frozenset[str] = frozenset()) -> Deployment:
    st = state_of(request)
    # A request forwarded by a linked session is served here, never forwarded again.
    dep = st.manager.resolve(model, modality, linked=not request.headers.get(HOP_HEADER), exclude=exclude)
    if dep is None:
        running = [d.served_name for d in st.manager.ready(modality)]
        hint = f" Running: {', '.join(running)}." if running else f" No {MODALITY_LABEL[modality]} is running yet."
        raise GatewayError(404, f"Model '{model}' not found.{hint}", "not_found_error", "model_not_found")
    return dep


MAX_ATTEMPTS = 3


async def call_with_failover(request: Request, model: str | None, modality: str,
                             call: Callable[[Deployment], Awaitable[T]]) -> tuple[Deployment, T]:
    """Run ``call`` on a copy of the model; if that copy doesn't answer, try another one."""
    tried: set[str] = set()
    while True:
        dep = find(request, model, modality, frozenset(tried))
        try:
            return dep, await call(dep)
        except Unreachable:
            tried.add(dep.id)
            if len(tried) >= MAX_ATTEMPTS or _alternative(request, model, modality, tried) is None:
                raise


async def stream_with_failover(request: Request, model: str | None, modality: str,
                               open_stream: Callable[[Deployment], AsyncIterator[dict]]) -> tuple[Deployment, AsyncIterator[dict]]:
    """Like ``call_with_failover`` for streams: switch copies only before the first chunk was produced."""
    tried: set[str] = set()
    while True:
        dep = find(request, model, modality, frozenset(tried))
        stream = open_stream(dep)
        try:
            first = await anext(stream)
        except StopAsyncIteration:
            return dep, _empty()
        except Unreachable:
            tried.add(dep.id)
            if len(tried) >= MAX_ATTEMPTS or _alternative(request, model, modality, tried) is None:
                raise
            continue
        return dep, _prepend(first, stream)


def _alternative(request: Request, model: str | None, modality: str, tried: set[str]) -> Deployment | None:
    st = state_of(request)
    return st.manager.resolve(model, modality, linked=not request.headers.get(HOP_HEADER), exclude=frozenset(tried))


async def _prepend(first: dict, rest: AsyncIterator[dict]) -> AsyncIterator[dict]:
    yield first
    async for chunk in rest:
        yield chunk


async def _empty() -> AsyncIterator[dict]:
    return
    yield  # pragma: no cover - makes this an async generator


def openai_error(e: GatewayError) -> JSONResponse:
    return JSONResponse({"error": {"message": e.message, "type": e.kind, "code": e.code}}, status_code=e.status)


def anthropic_error(e: GatewayError) -> JSONResponse:
    kind = {"invalid_request_error": "invalid_request_error", "not_found_error": "not_found_error",
            "authentication_error": "authentication_error", "permission_error": "permission_error",
            "rate_limit_error": "rate_limit_error"}.get(e.kind, "api_error")
    return JSONResponse({"type": "error", "error": {"type": kind, "message": e.message}}, status_code=e.status)


def ollama_error(e: GatewayError) -> JSONResponse:
    return JSONResponse({"error": e.message}, status_code=e.status)


def wrap(exc: Exception) -> GatewayError:
    if isinstance(exc, GatewayError):
        return exc
    if isinstance(exc, Unsupported):
        return GatewayError(400, f"This model doesn't support {exc}.")
    if isinstance(exc, UpstreamError):
        detail = exc.detail
        if isinstance(detail, dict):
            detail = (detail.get("error") or {}).get("message") if isinstance(detail.get("error"), dict) else \
                detail.get("error") or detail.get("detail") or json.dumps(detail)
        return GatewayError(exc.status if 400 <= exc.status < 600 else 502, f"Model server error: {detail}", "api_error")
    if isinstance(exc, HTTPException):
        return GatewayError(exc.status_code, str(exc.detail))
    return GatewayError(500, f"{type(exc).__name__}: {exc}", "api_error")


async def sse_openai(chunks: AsyncIterator[dict], served: str) -> AsyncIterator[str]:
    try:
        async for chunk in chunks:
            chunk["model"] = served
            yield "data: " + json.dumps(chunk) + "\n\n"
    except Exception as e:  # mid-stream failure: surface it in-band
        err = wrap(e)
        yield "data: " + json.dumps({"error": {"message": err.message, "type": err.kind}}) + "\n\n"
    yield "data: [DONE]\n\n"


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
