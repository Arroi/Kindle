"""Text/image-to-video worker built on diffusers (Wan, CogVideoX, LTX-Video, SVD, ...). Runs as jobs."""

from __future__ import annotations

import base64
import inspect
import io
import os
import random
import time

from .common import JobStore, WorkerState, create_app, parse_args, register_jobs, serve, start_loading
from .image_diffusers import load_pipeline


def run_job(pipe, payload: dict, defaults: dict, outputs: str, report) -> str:
    import torch
    from diffusers.utils import export_to_video

    params = inspect.signature(pipe.__call__).parameters
    steps = int(payload.get("steps") or defaults.get("steps") or 30)
    width = int(payload.get("width") or defaults.get("width") or 832)
    height = int(payload.get("height") or defaults.get("height") or 480)
    if payload.get("size") and "x" in str(payload["size"]):
        width, height = (int(v) for v in str(payload["size"]).lower().split("x", 1))
    fps = int(payload.get("fps") or defaults.get("fps") or 16)
    frames = payload.get("frames")
    if not frames and payload.get("seconds"):
        frames = int(float(payload["seconds"]) * fps) + 1
    frames = int(frames or defaults.get("frames") or 33)
    seed = payload.get("seed")
    seed = int(seed) if seed not in (None, "") else random.randint(0, 2**31 - 1)

    call: dict = {
        "prompt": payload.get("prompt", ""),
        "negative_prompt": payload.get("negative_prompt") or None,
        "num_inference_steps": steps,
        "num_frames": frames,
        "width": width - width % 16,
        "height": height - height % 16,
        "fps": fps,
        "generator": torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed),
    }
    if payload.get("guidance") is not None:
        call["guidance_scale"] = float(payload["guidance"])
    elif defaults.get("guidance") is not None:
        call["guidance_scale"] = float(defaults["guidance"])
    if payload.get("image_b64"):
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(payload["image_b64"]))).convert("RGB")
        call["image"] = img.resize((call["width"], call["height"]))
        if "video" in params and "image" not in params:
            call["video"] = [call.pop("image")]
    elif defaults.get("require_image"):
        raise ValueError("This world model requires a starting image. Upload one before generating.")
    elif "image" in params and params["image"].default is inspect.Parameter.empty:
        raise ValueError("This model is image-to-video: provide an input image")

    if "callback_on_step_end" in params:
        def on_step(_pipe, step, _t, kw):
            report((step + 1) / steps, f"Step {step + 1}/{steps}")
            return kw

        call["callback_on_step_end"] = on_step
    call = {k: v for k, v in call.items() if k in params and v is not None}

    result = pipe(**call)
    video = result.video if hasattr(result, "video") else result.frames[0]
    report(0.99, "Encoding video")
    path = os.path.join(outputs, f"video-{int(time.time() * 1000)}.mp4")
    export_to_video(video, path, fps=fps, macro_block_size=1)
    return path


def main() -> None:
    args = parse_args()
    state = WorkerState()
    app = create_app(state)
    store: dict = {}

    def get_store() -> JobStore:
        if "s" not in store:
            store["s"] = JobStore(args.outputs, lambda payload, report: run_job(
                state.obj, payload, args.options, args.outputs, report))
        return store["s"]

    register_jobs(app, state, get_store)

    def loader():
        opts = dict(args.options)
        opts.setdefault("offload", "model")  # video text encoders are big; offload by default on Kaggle
        pipe = load_pipeline(args.model, args.variant, opts)
        state.info = {"pipeline": type(pipe).__name__}
        return pipe

    start_loading(state, loader)
    serve(app, args.port)


if __name__ == "__main__":
    main()
