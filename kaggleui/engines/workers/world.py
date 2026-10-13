"""World-model worker: runs a recipe's upstream inference script per job (image + actions → video)."""

from __future__ import annotations

import base64
import glob
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .common import JobStore, WorkerState, create_app, parse_args, register_jobs, serve, start_loading

RECIPES = Path(__file__).resolve().parents[1] / "recipes" / "world"
TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[")

OASIS_KEYS = ["inventory", "ESC", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5", "hotbar.6",
              "hotbar.7", "hotbar.8", "hotbar.9", "forward", "back", "left", "right", "camera", "jump", "sneak",
              "sprint", "swapHands", "attack", "use", "pickItem", "drop"]


def load_recipe(name: str) -> dict:
    return yaml.safe_load((RECIPES / f"{name}.yaml").read_text())


def oasis_actions(segments: list[dict], frames: int) -> list[dict]:
    """Expand UI segments like {"keys": ["forward"], "camera": [0.2, 0], "frames": 8} into per-frame dicts.

    Oasis camera values are bucket indices in [0, 80] with 40 = no movement.
    """
    per_frame: list[dict] = []
    for seg in segments or []:
        n = max(1, int(seg.get("frames", 1)))
        keys = set(seg.get("keys") or [])
        cam = seg.get("camera") or [0, 0]
        act = {k: (1 if k in keys else 0) for k in OASIS_KEYS if k != "camera"}
        act["camera"] = [round(40 + max(-1, min(1, float(cam[0]))) * 40), round(40 + max(-1, min(1, float(cam[1]))) * 40)]
        per_frame += [dict(act) for _ in range(n)]
    idle = {k: 0 for k in OASIS_KEYS if k != "camera"} | {"camera": [40, 40]}
    if not per_frame:
        per_frame = [dict(idle, forward=1)]
    while len(per_frame) < frames + 1:
        per_frame.append(dict(per_frame[-1]))
    return per_frame[: frames + 1]


def run_job(recipe: dict, src: str, weights: str, outputs: str, payload: dict, report) -> str:
    stamp = str(int(time.time() * 1000))
    job_dir = Path(outputs) / f"world-{stamp}"
    job_dir.mkdir(parents=True, exist_ok=True)
    defaults = recipe.get("defaults") or {}
    frames = min(int(payload.get("frames") or defaults.get("frames", 32)), int(recipe.get("max_frames", 300)))
    steps = int(payload.get("steps") or defaults.get("steps", 10))
    if recipe["name"] == "lingbot_world_base":
        frames = max(5, 4 * ((frames - 1) // 4) + 1)
    fps = int(payload.get("fps") or defaults.get("fps", 16))
    seed = int(payload.get("seed") or random.randint(0, 2**31 - 1))

    image = os.path.join(src, recipe.get("default_image", ""))
    if payload.get("image_b64"):
        image = str(job_dir / "prompt.png")
        Path(image).write_bytes(base64.b64decode(payload["image_b64"]))

    actions = ""
    if recipe.get("actions_format") == "oasis":
        import torch

        actions = str(job_dir / "input.actions.pt")
        torch.save(oasis_actions(payload.get("actions") or [], frames), actions)

    output = str(job_dir / "out.mp4")
    values = {"python": sys.executable, "weights": weights, "image": image, "actions": actions, "frames": frames,
              "steps": steps, "fps": fps, "seed": seed, "output": output, "outdir": str(job_dir), "src": src,
              "prompt": str(payload.get("prompt") or "A coherent scene with natural motion."),
              "iterations": max(1, math.ceil((frames - 57) / 40) + 1)}
    cmd = [str(part).format(**values) for part in recipe["command"]]
    print("KUI_WORLD_CMD " + " ".join(cmd), flush=True)
    report(0.01, "Starting")
    proc = subprocess.Popen(cmd, cwd=src, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    assert proc.stdout
    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch in "\r\n":
            line = buf.strip()
            buf = ""
            if not line:
                continue
            tail = (tail + [line])[-20:]
            print(line, flush=True)
            m = TQDM_RE.search(line)
            if m and int(m.group(2)) > 0:
                report(int(m.group(1)) / int(m.group(2)) * 0.98, f"{m.group(1)}/{m.group(2)}")
        else:
            buf += ch
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{recipe['title']} exited with {rc}: " + " | ".join(tail[-4:]))
    if not os.path.exists(output):
        vids = sorted(glob.glob(str(job_dir / "**" / "*.mp4"), recursive=True), key=os.path.getmtime)
        if not vids:
            raise RuntimeError("The model finished but produced no video")
        output = vids[-1]
    return output


def main() -> None:
    args = parse_args()
    recipe = load_recipe(args.options["recipe"])
    src = args.options["src"]
    state = WorkerState()
    app = create_app(state)
    store: dict = {}

    def get_store() -> JobStore:
        if "s" not in store:
            store["s"] = JobStore(args.outputs, lambda payload, report: run_job(
                recipe, src, args.model, args.outputs, payload, report))
        return store["s"]

    register_jobs(app, state, get_store)

    @app.get("/recipe")
    def info() -> dict:
        return {"name": recipe["name"], "title": recipe["title"], "controls": recipe.get("controls"),
                "defaults": recipe.get("defaults"), "max_frames": recipe.get("max_frames")}

    def loader():
        if not os.path.isdir(src):
            raise RuntimeError(f"recipe source missing: {src}")
        state.info = {"recipe": recipe["name"], "controls": recipe.get("controls")}
        return recipe

    start_loading(state, loader)
    serve(app, args.port)


if __name__ == "__main__":
    main()
