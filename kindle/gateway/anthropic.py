"""Anthropic Messages API: proxied natively to Ollama, translated to OpenAI chat for other engines."""

from __future__ import annotations

import json

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .common import (
    SSE_HEADERS,
    GatewayError,
    anthropic_error,
    call_with_failover,
    check_access,
    find,
    stream_with_failover,
    wrap,
)
from .translate import anthropic_to_openai, openai_stream_to_anthropic, openai_to_anthropic, wants_thinking

router = APIRouter()
_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=600, write=60, pool=10))


async def _body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception as e:
        raise GatewayError(400, "Request body must be JSON") from e
    if not isinstance(body, dict) or not body.get("messages"):
        raise GatewayError(400, "'messages' is required")
    return body


@router.post("/v1/messages")
async def messages(request: Request):
    try:
        check_access(request, "anthropic")
        body = await _body(request)
        dep = find(request, body.get("model"), "text")
        backend = dep.backend
        assert backend
        served = dep.served_name

        if "anthropic" in backend.native and backend.base_url:
            upstream = {**body, "model": backend.upstream_model}
            url = f"{backend.base_url}/v1/messages"
            headers = {"content-type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": "kindle"}
            if body.get("stream"):
                async def relay():
                    async with _client.stream("POST", url, json=upstream, headers=headers) as r:
                        if r.status_code >= 400:
                            err = (await r.aread()).decode(errors="replace")[:500]
                            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': err}})}\n\n"
                            return
                        async for line in r.aiter_lines():
                            yield line.replace(backend.upstream_model or "\0", served) + "\n"

                return StreamingResponse(relay(), media_type="text/event-stream", headers=SSE_HEADERS)
            r = await _client.post(url, json=upstream, headers=headers)
            try:
                data = r.json()
            except ValueError:
                raise GatewayError(r.status_code if r.status_code >= 400 else 502,
                                   f"Model server error: {r.text[:300]}", "api_error") from None
            if isinstance(data, dict) and data.get("model"):
                data["model"] = served
            return JSONResponse(data, status_code=r.status_code)

        oai = anthropic_to_openai(body)
        thinking = wants_thinking(body)
        if body.get("stream"):
            dep, stream = await stream_with_failover(request, body.get("model"), "text",
                                                     lambda d: d.backend.chat_stream(oai))
            return StreamingResponse(openai_stream_to_anthropic(stream, dep.served_name, thinking),
                                     media_type="text/event-stream", headers=SSE_HEADERS)
        dep, resp = await call_with_failover(request, body.get("model"), "text", lambda d: d.backend.chat(oai))
        return openai_to_anthropic(resp, dep.served_name, thinking)
    except Exception as e:  # noqa: BLE001
        return anthropic_error(wrap(e))


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    try:
        check_access(request, "anthropic")
        body = await _body(request)
        counted = None
        try:
            dep = find(request, body.get("model"), "text")
            if dep.backend:
                counted = await dep.backend.count_tokens(anthropic_to_openai(body))
        except GatewayError:
            pass
        if counted is None:  # runners without a tokenizer endpoint: a rough estimate
            text = json.dumps(body.get("messages")) + json.dumps(body.get("system") or "") + json.dumps(body.get("tools") or [])
            counted = max(1, len(text) // 4)
        return {"input_tokens": counted}
    except Exception as e:  # noqa: BLE001
        return anthropic_error(wrap(e))
