"""Text-to-image worker built on diffusers (SD 1.5/2.x/XL/3.x, FLUX, Qwen-Image, ...)."""

from __future__ import annotations

import base64
import inspect
import io
import random

from fastapi import Body

from .common import WorkerState, create_app, device, parse_args, serve, start_loading, torch_dtype


def quantized_components(model: str) -> list[str]:
    """What nf4 quantizes; big T5-style text encoders (Wan's UMT5-XXL is 11 GB) too, small CLIPs not.

    Mirrors ``kaggleui.catalog.fit.nf4_quantized`` so the fit estimate matches what loads.
    """
    from pathlib import Path

    names = ["transformer", "transformer_2", "text_encoder_2", "text_encoder_3", "unet"]
    folder = Path(model) / "text_encoder"
    if folder.is_dir() and sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) > 2 * 1024**3:
        names.append("text_encoder")
    present = [n for n in names if (Path(model) / n).is_dir()]
    return present or names


def load_pipeline(model: str, variant: str | None, options: dict):
    from diffusers import DiffusionPipeline

    pipeline_name = options.get("pipeline")
    if pipeline_name in ("Cosmos3OmniPipeline", "CosmosTextToWorldPipeline", "CosmosVideoToWorldPipeline"):
        import diffusers

        DiffusionPipeline = getattr(diffusers, pipeline_name)

    dtype = torch_dtype(options)
    kwargs: dict = {"torch_dtype": dtype}
    if variant:
        kwargs["variant"] = variant
    if options.get("quantize") == "nf4":
        from diffusers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": dtype},
            components_to_quantize=quantized_components(model),
        )
    try:
        pipe = DiffusionPipeline.from_pretrained(model, **kwargs)
    except ValueError:
        if "variant" in kwargs:  # some components have no fp16 variant
            kwargs.pop("variant")
            pipe = DiffusionPipeline.from_pretrained(model, **kwargs)
        else:
            raise
    if type(pipe).__name__ == "Cosmos3OmniPipeline":
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=options.get("flow_shift", 10.0), use_karras_sigmas=False)
    dev = device()
    offload = options.get("offload")
    if dev == "cuda" and offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif dev == "cuda" and (offload == "model" or (options.get("quantize") and offload != "none")):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(dev)
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, req: dict, defaults: dict) -> list[str]:
    import torch

    params = inspect.signature(pipe.__call__).parameters
    n = max(1, min(int(req.get("n") or 1), 4))
    width = int(req.get("width") or defaults.get("width") or 1024)
    height = int(req.get("height") or defaults.get("height") or 1024)
    if req.get("size") and "x" in str(req["size"]):
        width, height = (int(v) for v in str(req["size"]).lower().split("x", 1))
    seed = req.get("seed")
    seed = int(seed) if seed not in (None, "") else random.randint(0, 2**31 - 1)
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    call = {
        "prompt": req.get("prompt", ""),
        "negative_prompt": req.get("negative_prompt") or None,
        "num_inference_steps": int(req.get("steps") or defaults.get("steps") or 30),
        "guidance_scale": float(req.get("guidance") if req.get("guidance") is not None else defaults.get("guidance", 5.0)),
        "width": width - width % 8,
        "height": height - height % 8,
        "num_images_per_prompt": n,
        "generator": torch.Generator(gen_device).manual_seed(seed),
    }
    call = {k: v for k, v in call.items() if k in params and v is not None}
    images = pipe(**call).images
    out = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


def main() -> None:
    args = parse_args()
    state = WorkerState()
    app = create_app(state)
    defaults = args.options

    @app.post("/images")
    def images(req: dict = Body(...)) -> dict:
        pipe = state.require_ready()
        with state.gpu_lock:
            data = generate(pipe, req, defaults)
        return {"images": data}

    def loader():
        pipe = load_pipeline(args.model, args.variant, args.options)
        state.info = {"pipeline": type(pipe).__name__}
        return pipe

    start_loading(state, loader)
    serve(app, args.port)


if __name__ == "__main__":
    main()
