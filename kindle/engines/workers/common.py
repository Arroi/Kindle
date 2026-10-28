"""Shared scaffolding for worker processes.

A worker is a small FastAPI app on localhost that loads one model in a background thread and
serves it. Requests during loading get 503; ``GET /health`` reports the load stage.
Only fastapi/uvicorn/pydantic are needed here so workers can also run inside isolated venvs.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse


def parse_args(extra: Callable[[argparse.ArgumentParser], None] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="local path or HF repo id")
    p.add_argument("--name", default="model")
    p.add_argument("--engine", default="")
    p.add_argument("--variant", default=None)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--options", default="{}")
    p.add_argument("--outputs", default="/tmp/kindle/outputs")
    if extra:
        extra(p)
    args = p.parse_args()
    args.options = json.loads(args.options or "{}")
    if args.variant in ("", "None", "null"):
        args.variant = None
    Path(args.outputs).mkdir(parents=True, exist_ok=True)
    return args


class WorkerState:
    def __init__(self) -> None:
        self.stage = "loading"
        self.message = "Loading model"
        self.error: str | None = None
        self.obj: Any = None
        self.info: dict = {}
        self.gpu_lock = threading.Lock()

    def require_ready(self) -> Any:
        if self.stage == "error":
            raise HTTPException(500, f"Model failed to load: {self.error}")
        if self.stage != "ready":
            raise HTTPException(503, self.message)
        return self.obj


def create_app(state: WorkerState) -> FastAPI:
    app = FastAPI()

    @app.exception_handler(ValueError)
    async def invalid_request(request, exc: ValueError):
        return JSONResponse({"error": {"message": str(exc), "type": "invalid_request_error"}}, status_code=400)

    @app.exception_handler(Exception)
    async def inference_error(request, exc: Exception):
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        return JSONResponse({"error": {"message": f"{type(exc).__name__}: {exc}", "type": "worker_error"}}, status_code=500)

    @app.get("/health")
    def health() -> dict:
        return {"stage": state.stage, "message": state.message, "error": state.error, "info": state.info}

    return app


def start_loading(state: WorkerState, loader: Callable[[], Any]) -> None:
    def run() -> None:
        try:
            state.obj = loader()
            state.stage = "ready"
            state.message = "Ready"
            print("KUI_READY", flush=True)
        except Exception as e:  # noqa: BLE001 — reported to the panel
            state.stage = "error"
            detail = str(e)
            if "trust_remote_code" in detail:
                detail = "This model requires custom Python code. In Customize → Settings, enable 'Allow model custom code' for a repository you trust, then deploy again. " + detail
            elif "does not recognize this architecture" in detail:
                detail = "This checkpoint needs an architecture-specific runtime. Update Kindle and deploy again; if it still fails, include the model ID when reporting it. " + detail
            state.error = f"{type(e).__name__}: {detail}"
            state.message = "Load failed"
            traceback.print_exc()

    threading.Thread(target=run, daemon=True).start()


def serve(app: FastAPI, port: int) -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def torch_dtype(options: dict):
    import torch

    name = str(options.get("dtype") or "float16")
    return {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32}.get(name, torch.float16)


def device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def progress(stage: str, frac: float | None = None, message: str = "") -> None:
    """Report progress to the deployment manager via stdout."""
    print("KUI_PROGRESS " + json.dumps({"stage": stage, "progress": frac, "message": message}), flush=True)


class JobStore:
    """Sequential job queue for long generations (video, world). One GPU job at a time."""

    def __init__(self, outputs: str, run: Callable[[dict, Callable[[float, str], None]], str]):
        self.outputs = outputs
        self.run = run
        self.jobs: dict[str, dict] = {}
        self.q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, payload: dict) -> str:
        job_id = "job_" + uuid.uuid4().hex[:16]
        self.jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0.0, "message": "Queued",
                             "error": None, "result": None, "created": time.time(), "payload": payload}
        self.q.put(job_id)
        return job_id

    def get(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "No such job")
        return {k: v for k, v in job.items() if k not in ("payload", "result")} | {
            "has_result": bool(job.get("result"))}

    def result(self, job_id: str) -> FileResponse:
        job = self.jobs.get(job_id)
        if not job or not job.get("result"):
            raise HTTPException(404, "Result not ready")
        path = job["result"]
        media = "video/mp4" if path.endswith(".mp4") else "image/gif" if path.endswith(".gif") else "application/octet-stream"
        if path.endswith(".zip"):
            media = "application/zip"
        return FileResponse(path, media_type=media, filename=os.path.basename(path))

    def assets(self, job_id: str) -> list[dict]:
        from .world_assets import manifest

        job = self.jobs.get(job_id)
        if not job or not job.get("result"):
            raise HTTPException(404, "Result not ready")
        return manifest(job["result"])["assets"]

    def asset(self, job_id: str, name: str) -> FileResponse:
        from .world_assets import asset_path

        job = self.jobs.get(job_id)
        if not job or not job.get("result"):
            raise HTTPException(404, "Result not ready")
        try:
            path, media = asset_path(job["result"], name)
        except KeyError as exc:
            raise HTTPException(404, "No such world asset") from exc
        return FileResponse(path, media_type=media, filename=path.name)

    def _loop(self) -> None:
        while True:
            job_id = self.q.get()
            job = self.jobs[job_id]
            job["status"] = "in_progress"
            job["message"] = "Generating"

            def report(frac: float, msg: str = "", job: dict = job) -> None:
                job["progress"] = max(0.0, min(frac, 1.0))
                if msg:
                    job["message"] = msg

            try:
                job["result"] = self.run(job["payload"], report)
                job["status"] = "completed"
                job["progress"] = 1.0
                job["message"] = "Done"
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                job["status"] = "failed"
                job["error"] = f"{type(e).__name__}: {e}"
                job["message"] = "Failed"


def register_jobs(app: FastAPI, state: WorkerState, store_getter: Callable[[], JobStore]) -> None:
    from fastapi import Body

    @app.post("/jobs")
    def submit(payload: dict = Body(...)) -> dict:
        state.require_ready()
        return {"id": store_getter().submit(payload)}

    @app.get("/jobs/{job_id}")
    def status(job_id: str) -> dict:
        return store_getter().get(job_id)

    @app.get("/jobs/{job_id}/result")
    def result(job_id: str):
        return store_getter().result(job_id)

    @app.get("/jobs/{job_id}/assets")
    def assets(job_id: str):
        return store_getter().assets(job_id)

    @app.get("/jobs/{job_id}/assets/{name:path}")
    def asset(job_id: str, name: str):
        return store_getter().asset(job_id, name)
