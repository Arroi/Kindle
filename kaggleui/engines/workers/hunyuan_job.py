"""Adapters for Tencent's official HunyuanWorld, WorldPlay and HY-World 2.0 code."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

PREFIX = "KUI_JOB_PROGRESS "


def report(progress: float, message: str) -> None:
    print(PREFIX + json.dumps({"progress": progress, "message": message}), flush=True)


def module(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def run(args: list[str], cwd: Path, env: dict | None = None) -> None:
    subprocess.run([str(a) for a in args], cwd=cwd, env={**os.environ, **(env or {})}, check=True)


def input_image(payload: dict, folder: Path) -> Path | None:
    if not payload.get("image_b64"):
        return None
    path = folder / "input.png"
    import io

    from PIL import Image

    Image.open(io.BytesIO(base64.b64decode(payload["image_b64"]))).convert("RGB").save(path)
    return path


def world1(src: Path, weights: Path, folder: Path, payload: dict, image: Path | None) -> None:
    env = {"KUI_WORLD1_WEIGHTS": str(weights), "KUI_WORLD1_STEPS": str(payload.get("steps") or 50)}
    pano = folder / "panorama.png"
    if image and payload.get("input_mode") == "panorama":
        shutil.copy2(image, pano)
    else:
        report(0.03, "Generating the 360° panorama")
        args = [sys.executable, "demo_panogen.py", "--prompt", str(payload.get("prompt") or ""),
                "--negative_prompt", str(payload.get("negative_prompt") or ""), "--output_path", str(folder),
                "--seed", str(payload.get("seed", 42))]
        if image:
            args += ["--image_path", str(image)]
        run(args, src, env)
    report(0.4, "Decomposing the panorama and reconstructing the 3D world")
    args = [sys.executable, "demo_scenegen.py", "--image_path", str(pano), "--output_path", str(folder),
            "--classes", str(payload.get("scene_class") or "outdoor"), "--seed", str(payload.get("seed", 42))]
    for key in ("labels_fg1", "labels_fg2"):
        labels = payload.get(key) or []
        if isinstance(labels, str):
            labels = [v.strip() for v in labels.split(",") if v.strip()]
        if labels:
            if any(not isinstance(label, str) or label.startswith("-") or len(label) > 100 for label in labels):
                raise ValueError("Object labels must be short names separated by commas")
            args += ["--" + key, *labels[:20]]
    run(args, src, env)
    if not list(folder.glob("mesh_layer*.ply")):
        raise RuntimeError("HunyuanWorld finished without exporting its mesh layers")


def worldplay(src: Path, weights: Path, folder: Path, payload: dict, image: Path | None) -> None:
    chunks = max(1, min(int(payload.get("chunks") or 1), 16))
    pose = str(payload.get("pose") or f"w-{chunks * 4}")
    segments = [segment.strip() for segment in pose.split(",")]
    durations = []
    for segment in segments:
        match = re.fullmatch(r"(w|a|s|d|up|down|left|right)-(\d+(?:\.\d+)?)", segment)
        if not match:
            raise ValueError("Use pose actions such as w-4 or w-2,right-2")
        durations.append(float(match.group(2)))
    if abs(sum(durations) - chunks * 4) > 0.001:
        raise ValueError(f"Pose durations must total {chunks * 4} latents for {chunks} chunk(s)")
    count = max(1, len([v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if v]))
    report(0.02, "Generating action-conditioned WorldPlay video")
    args = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={count}",
            "wan/generate.py", "--input", str(payload.get("prompt") or ""), "--num_chunk", str(chunks),
            "--num_frames", str(chunks * 16 + 1), "--num_inference_steps", str(payload.get("steps") or 4),
            "--pose", pose, "--ar_model_path", str(weights / "wan_transformer"),
            "--ckpt_path", str(weights / "wan_distilled_model/model.pt"), "--out", str(folder)]
    if image:
        args += ["--image_path", str(image)]
    env = {"PYTHONPATH": os.pathsep.join([str(src), str(src / "wan"), os.environ.get("PYTHONPATH", "")]),
           "KUI_WORLDPLAY_SEED": str(payload.get("seed") or 0)}
    run(args, src, env)


def world2(src: Path, weights: Path, folder: Path, payload: dict, image: Path | None) -> None:
    import httpx

    endpoint = str(payload.get("vlm_url") or "").strip().rstrip("/")
    parts = urlsplit(endpoint)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise ValueError("HY-World 2.0 needs an OpenAI-compatible vision-language server URL, including /v1")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    key = str(payload.get("vlm_key") or "EMPTY")
    with httpx.Client(timeout=15) as client:
        reply = client.get(endpoint + "/models", headers={"Authorization": "Bearer " + key})
        reply.raise_for_status()
    name = str(payload.get("vlm_model") or "Qwen/Qwen3-VL-8B-Instruct")
    pano = folder / "panorama.png"
    if image and payload.get("input_mode") == "panorama":
        shutil.copy2(image, pano)
    else:
        # Text inputs first create a perspective image, then use Tencent's HY-Pano 2.0 expansion.
        if image is None:
            report(0.02, "Generating the starting image from text")
            script = folder.parent / "make_image.py"
            script.write_text('''import json,sys,torch
from diffusers import QwenImagePipeline
spec=json.load(open(sys.argv[1]))
pipe=QwenImagePipeline.from_pretrained("Qwen/Qwen-Image",torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe(prompt=spec["prompt"],negative_prompt=spec.get("negative_prompt") or " ",height=512,width=1024,num_inference_steps=40,true_cfg_scale=4.0,generator=torch.Generator("cpu").manual_seed(int(spec.get("seed",42)))).images[0].save(sys.argv[2])
''')
            image = folder / "starting_image.png"
            config = folder.parent / "input.json"
            run([sys.executable, str(script), str(config), str(image)], src)
        report(0.12, "Generating the HY-Pano 2.0 panorama")
        script = folder.parent / "make_panorama.py"
        script.write_text('''import json,sys,torch
sys.path.insert(0,sys.argv[4])
from pipeline_with_qwen_image import HunyuanPanoPipeline, PanoDiffusionPipeline
spec=json.load(open(sys.argv[1]))
pipe=PanoDiffusionPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509",torch_dtype=torch.bfloat16)
pipe.load_lora_weights(sys.argv[5],subfolder="HY-Pano-2.0",weight_name="pytorch_lora_weights.safetensors")
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
HunyuanPanoPipeline(pipe)(sys.argv[2],prompt=spec.get("prompt") or "",negative_prompt=spec.get("negative_prompt") or "",seed=int(spec.get("seed",42)),num_inference_steps=int(spec.get("steps") or 40)).save(sys.argv[3])
''')
        run([sys.executable, str(script), str(folder.parent / "input.json"), str(image), str(pano),
             str(src / "hyworld2/panogen"), str(weights)], src)
    gen = src / "hyworld2/worldgen"
    env = {"KUI_WORLD2_VLM_URL": endpoint, "KUI_WORLD2_VLM_KEY": key,
           "KUI_WORLD2_WEIGHTS": str(weights),
           "PYTHONPATH": os.pathsep.join([str(gen), str(src), os.environ.get("PYTHONPATH", "")])}
    # The upstream CLI still takes host/port, while the patched client also accepts HTTPS and API keys.
    vlm = ["--llm_addr", parts.hostname, "--llm_port", str(parts.port or (443 if parts.scheme == "https" else 80)),
           "--llm_name", name]
    count = max(1, len([v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if v]))
    distributed = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={count}"]
    report(0.25, "Planning navigation trajectories")
    run([sys.executable, "traj_generate.py", "--target_path", str(folder), *vlm,
         "--apply_nav_traj", "--apply_up_route", "--apply_recon_iteration", "--force_vlm"], gen, env)
    report(0.4, "Rendering trajectories and describing views")
    run([*distributed, "traj_render.py", "--target_path", str(folder), *vlm], gen, env)
    report(0.55, "Expanding the world with WorldStereo 2.0")
    run([*distributed, "video_gen.py", "--target_path", str(folder), "--fsdp"], gen, env)
    report(0.7, "Preparing Gaussian scene training data")
    run([*distributed, "gen_gs_data.py", "--root_path", str(folder), "--save_normal", "--split_sky"], gen, env)
    report(0.82, "Optimizing and exporting the 3D world")
    steps = max(100, min(int(payload.get("training_steps") or (8000 if count == 1 else 4000 if count == 2 else 2000)), 20000))
    run([sys.executable, "-m", "world_gs_trainer", "default", "--data_dir", str(folder / "gs_data"),
         "--result_dir", str(folder / "world"), "--max_steps", str(steps), "--save_steps", str(steps),
         "--eval_steps", str(steps), "--ply_steps", str(steps), "--save_ply", "--convert_to_spz", "--export_mesh",
         "--disable_video", "--disable_viewer", "--use_mask_gaussian", "--antialiased", "--depth_loss",
         "--normal_loss", "--sky_depth_from_pcd", "--use_scale_regularization"], gen, env)
    if not list((folder / "world").rglob("*.ply")):
        raise RuntimeError("HY-World 2.0 finished without exporting its 3D scene")


def main() -> None:
    kind, src, weights, folder, config = sys.argv[1:]
    src, weights, folder = Path(src), Path(weights), Path(folder)
    payload = json.loads(Path(config).read_text())
    image = input_image(payload, folder)
    if not image and not str(payload.get("prompt") or "").strip():
        raise ValueError("Enter a text prompt or provide an input image")
    {"hunyuan_world_1": world1, "hy_worldplay": worldplay, "hy_world_2": world2}[kind](
        src, weights, folder, payload, image)


if __name__ == "__main__":
    main()
