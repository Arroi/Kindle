"""OpenAI-compatible API: chat, completions, embeddings, images, speech, transcriptions, videos, models."""

from __future__ import annotations

import base64
import time
import uuid

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from .common import SSE_HEADERS, GatewayError, check_access, find, openai_error, sse_openai, state_of, wrap

router = APIRouter()


async def _json(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception as e:
        raise GatewayError(400, "Request body must be JSON") from e
    if not isinstance(body, dict):
        raise GatewayError(400, "Request body must be a JSON object")
    return body


def _model_entry(d) -> dict:
    return {"id": d.served_name, "object": "model", "created": int(d.ready_at or d.created_at),
            "owned_by": "kaggleui", "modality": d.spec.modality}


@router.get("/v1/models")
async def list_models(request: Request):
    try:
        if request.headers.get("anthropic-version"):
            check_access(request, "anthropic")
        else:
            check_access(request, "openai")
    except GatewayError as e:
        return openai_error(e)
    deps = state_of(request).manager.ready()
    if request.headers.get("anthropic-version"):
        data = [{"type": "model", "id": d.served_name, "display_name": d.served_name,
                 "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(d.ready_at or d.created_at))}
                for d in deps if d.spec.modality == "text"]
        return {"data": data, "has_more": False, "first_id": data[0]["id"] if data else None,
                "last_id": data[-1]["id"] if data else None}
    return {"object": "list", "data": [_model_entry(d) for d in deps]}


@router.get("/v1/models/{model_id:path}")
async def get_model(model_id: str, request: Request):
    try:
        check_access(request, "openai")
        for d in state_of(request).manager.ready():
            if d.served_name == model_id:
                return _model_entry(d)
        raise GatewayError(404, f"Model '{model_id}' not found", "not_found_error", "model_not_found")
    except GatewayError as e:
        return openai_error(e)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        check_access(request, "openai")
        body = await _json(request)
        dep = find(request, body.get("model"), "text")
        backend = dep.backend
        assert backend
        if body.get("stream"):
            return StreamingResponse(sse_openai(backend.chat_stream(body), dep.served_name),
                                     media_type="text/event-stream", headers=SSE_HEADERS)
        resp = await backend.chat(body)
        resp["model"] = dep.served_name
        return resp
    except Exception as e:  # noqa: BLE001
        return openai_error(wrap(e))


@router.post("/v1/completions")
async def completions(request: Request):
    try:
        check_access(request, "openai")
        body = await _json(request)
        dep = find(request, body.get("model"), "text")
        assert dep.backend
        resp = await dep.backend.completions({**body, "stream": False})
        resp["model"] = dep.served_name
        return resp
    except Exception as e:  # noqa: BLE001
        return openai_error(wrap(e))


@router.post("/v1/embeddings")
async def embeddings(request: Request):
    try:
        check_access(request, "openai")
        body = await _json(request)
        dep = find(request, body.get("model"), "text")
        assert dep.backend
        resp = await dep.backend.embeddings(body)
        resp["model"] = dep.served_name
        return resp
    except Exception as e:  # noqa: BLE001
        return openai_error(wrap(e))


@router.post("/v1/images/generations")
async def images(request: Request):
    try:
        check_access(request, "openai")
        body = await _json(request)
        dep = find(request, body.get("model"), "image")
        assert dep.backend
        out = await dep.backend.images(body)
        imgs = out.get("images") or []
        as_url = body.get("response_format") == "url"
        data = [{"url": f"data:image/png;base64,{b}"} if as_url else {"b64_json": b, "revised_prompt": body.get("prompt")}
                for b in imgs]
        return {"created": int(time.time()), "data": data}
    except Exception as e:  # noqa: BLE001
        return openai_error(wrap(e))


@router.post("/v1/audio/speech")
async def speech(request: Request):
    try:
        check_access(request, "openai")
        body = await _json(request)
        dep = find(request, body.get("model"), "tts")
        assert dep.backend
        data, media = await dep.backend.speech(body)
        return Response(data, media_type=media)
    except Exception as e:  # noqa: BLE001
        return openai_error(wrap(e))


@router.get("/v1/audio/voices")
async def voices(request: Request, model: str | None = None):
    try:
        check_access(request, "openai")
        dep = find(request, model, "tts")
        assert dep.backend
        return {"voices": await dep.backend.voices()}
    except Exception as e:  # noqa: BLE001
        return openai_error(wrap(e))


def _fmt_ts(t: float | None, sep: str) -> str:
    t = t or 0.0
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)):03d}"


def _format_transcript(result: dict, fmt: str):
    segs = result.get("segments") or []
    if fmt == "text":
        return PlainTextResponse(result.get("text", ""))
    if fmt == "srt":
        lines = [f"{i + 1}\n{_fmt_ts(s.get('start'), ',')} --> {_fmt_ts(s.get('end'), ',')}\n{s.get('text', '')}\n"
                 for i, s in enumerate(segs)]
        return PlainTextResponse("\n".join(lines))
    if fmt == "vtt":
        lines = ["WEBVTT\n"] + [f"{_fmt_ts(s.get('start'), '.')} --> {_fmt_ts(s.get('end'), '.')}\n{s.get('text', '')}\n"
                                for s in segs]
        return PlainTextResponse("\n".join(lines), media_type="text/vtt")
    if fmt == "verbose_json":
        return result
    return {"text": result.get("text", "")}


async def _transcribe(request: Request, task: str):
    check_access(request, "openai")
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile) and not hasattr(upload, "read"):
        raise GatewayError(400, "Missing 'file' upload")
    dep = find(request, form.get("model"), "stt")
    assert dep.backend
    granularities = form.getlist("timestamp_granularities[]") or form.getlist("timestamp_granularities")
    params = {"language": form.get("language"), "prompt": form.get("prompt"), "task": task,
              "temperature": form.get("temperature") or 0, "word_timestamps": "word" in granularities}
    data = await upload.read()
    result = await dep.backend.transcribe(data, getattr(upload, "filename", "audio"), params)
    return _format_transcript(result, str(form.get("response_format") or "json"))


@router.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    try:
        return await _transcribe(request, "transcribe")
    except Exception as e:  # noqa: BLE001
        return openai_error(wrap(e))


@router.post("/v1/audio/translations")
async def translations(request: Request):
    try:
        return await _transcribe(request, "translate")
    except Exception as e:  # noqa: BLE001
        return openai_error(wrap(e))


# -- Videos (OpenAI-style async jobs; also used for world models) --------------------------------

def _video_object(job: dict, status: dict | None = None) -> dict:
    status = status or {}
    state = status.get("status", job.get("status", "queued"))
    obj = {"id": job["id"], "object": "video", "model": job["model"], "status": state,
           "progress": int(round(float(status.get("progress", 0)) * 100)), "created_at": int(job["created"]),
           "seconds": job.get("seconds"), "size": job.get("size"), "prompt": job.get("prompt")}
    if status.get("message"):
        obj["message"] = status["message"]
    if job.get("start_frame_model"):
        obj["start_frame_model"] = job["start_frame_model"]
    if state == "failed":
        obj["error"] = {"message": status.get("error") or "Generation failed"}
    return obj


async def paint_start_frame(request: Request, dep, body: dict) -> tuple[bytes, str]:
    """World models don't read text. Turn the prompt into a first frame with a running image model."""
    st = state_of(request)
    wanted = body.get("start_frame_model")
    painter = find(request, wanted, "image") if wanted else next(iter(st.manager.ready("image")), None)
    if painter is None:
        raise GatewayError(400, f"{dep.served_name} doesn't take text; it starts from an image. To start from a "
                                "description, also deploy an image model (SD Turbo is small and fast) and KaggleUI "
                                "paints the first frame with it. Or send an input_reference image.")
    frame = dep.info.get("start_frame") or {}
    w, _, h = str(frame.get("size") or "640x360").partition("x")
    prompt = f"{frame.get('prompt_prefix') or ''}{str(body.get('prompt')).strip()}"
    out = await painter.backend.images({"prompt": prompt, "width": int(w), "height": int(h), "n": 1,
                                        "seed": body.get("seed")})
    images = out.get("images") or []
    if not images:
        raise GatewayError(502, "The image model didn't return a starting frame", "api_error")
    return base64.b64decode(images[0]), painter.served_name


@router.post("/v1/start-frames")
async def start_frame(request: Request):
    """Preview the first frame a world model would start from (used by the Studio)."""
    try:
        check_access(request, "openai")
        body = await _json(request)
        dep = find(request, body.get("model"), "world")
        data, painter = await paint_start_frame(request, dep, body)
        return {"b64_json": base64.b64encode(data).decode(), "painted_with": painter}
    except Exception as e:  # noqa: BLE001
        return openai_error(wrap(e))


@router.post("/v1/videos")
async def create_video(request: Request):
    try:
        check_access(request, "openai")
        st = state_of(request)
        image: bytes | None = None
        if request.headers.get("content-type", "").startswith("multipart/"):
            form = await request.form()
            body = {k: v for k, v in form.items() if isinstance(v, str)}
            ref = form.get("input_reference")
            if ref is not None and hasattr(ref, "read"):
                image = await ref.read()
        else:
            body = await _json(request)
            if body.get("input_reference_b64"):
                image = base64.b64decode(body["input_reference_b64"])
        model = body.get("model")
        dep = None
        for modality in ("video", "world"):
            try:
                dep = find(request, model, modality)
                break
            except GatewayError:
                continue
        if dep is None:
            raise GatewayError(404, f"No video or world model named '{model}' is running", "not_found_error",
                               "model_not_found")
        assert dep.backend
        painted_with = None
        if dep.spec.modality == "world" and image is None and str(body.get("prompt") or "").strip():
            image, painted_with = await paint_start_frame(request, dep, body)
        for k in ("frames", "steps", "fps", "seed", "width", "height"):
            if k in body and isinstance(body[k], str) and body[k].strip():
                try:
                    body[k] = float(body[k]) if "." in body[k] else int(body[k])
                except ValueError:
                    pass
        worker_id = await dep.backend.job_submit(body, image)
        vid = "video_" + uuid.uuid4().hex[:20]
        job = {"id": vid, "dep": dep.id, "worker_id": worker_id, "model": dep.served_name, "created": time.time(),
               "prompt": body.get("prompt"), "seconds": body.get("seconds"), "size": body.get("size"), "status": "queued",
               "start_frame_model": painted_with}
        st.video_jobs[vid] = job
        return _video_object(job)
    except Exception as e:  # noqa: BLE001
        return openai_error(wrap(e))


async def _job(request: Request, video_id: str):
    st = state_of(request)
    job = st.video_jobs.get(video_id)
    if not job:
        raise GatewayError(404, f"Video '{video_id}' not found", "not_found_error")
    dep = st.manager.get(job["dep"])
    if not dep or not dep.backend:
        raise GatewayError(410, "The model that made this video is no longer running", "not_found_error")
    return job, dep


@router.get("/v1/videos")
async def list_videos(request: Request):
    try:
        check_access(request, "openai")
        st = state_of(request)
        jobs = sorted(st.video_jobs.values(), key=lambda j: -j["created"])
        return {"object": "list", "data": [_video_object(j) for j in jobs]}
    except Exception as e:  # noqa: BLE001
        return openai_error(wrap(e))


@router.get("/v1/videos/{video_id}")
async def get_video(video_id: str, request: Request):
    try:
        check_access(request, "openai")
        job, dep = await _job(request, video_id)
        status = await dep.backend.job_status(job["worker_id"])
        job["status"] = status.get("status", job["status"])
        return _video_object(job, status)
    except Exception as e:  # noqa: BLE001
        return openai_error(wrap(e))


@router.get("/v1/videos/{video_id}/content")
async def video_content(video_id: str, request: Request):
    try:
        check_access(request, "openai")
        job, dep = await _job(request, video_id)
        data, media = await dep.backend.job_result(job["worker_id"])
        return Response(data, media_type=media, headers={"Content-Disposition": f'inline; filename="{video_id}.mp4"'})
    except Exception as e:  # noqa: BLE001
        return openai_error(wrap(e))


@router.get("/v1/health")
async def health(request: Request):
    st = state_of(request)
    return JSONResponse({"ok": True, "models": [d.served_name for d in st.manager.ready()]})
