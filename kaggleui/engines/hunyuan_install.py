"""Additional installation and checkpoint preparation for Tencent's upstream runners."""

from __future__ import annotations

from pathlib import Path

from ..catalog import hf
from .context import EngineContext
from .download import hf_download, repo_files
from .install import pip_install


def requirements(recipe: dict, src: Path) -> list[str]:
    reqs = list(recipe.get("requirements", []))
    if recipe.get("requirements_file"):
        # CUDA wheels avoid compiling CuPy itself; both distributions provide the same cupy module.
        text = (src / recipe["requirements_file"]).read_text().replace("cupy==", "cupy-cuda12x==")
        path = src / ".kaggleui-requirements.txt"
        path.write_text(text)
        reqs += ["-r", str(path)]
    return reqs


async def prepare(recipe: dict, src: Path, python: str, ctx: EngineContext) -> None:
    adapter = recipe.get("adapter")
    marker = src / ".kaggleui-native-ready"
    if not marker.exists():
        if adapter == "hunyuan_world_1":
            await pip_install(ctx, ["--no-deps", "realesrgan==0.3.0", "gfpgan==1.3.8"], python=python, label="Real-ESRGAN")
            await pip_install(ctx, ["--no-build-isolation", "git+https://github.com/facebookresearch/pytorch3d.git",
                                   "git+https://github.com/microsoft/MoGe.git"],
                              python=python, label="HunyuanWorld reconstruction libraries")
            await pip_install(ctx, ["--no-deps", "git+https://github.com/naver-ai/ZIM.git"], python=python, label="ZIM")
        elif adapter == "hy_world_2":
            await pip_install(ctx, ["--no-build-isolation", "-e", str(src / "hyworld2/worldgen/third_party/gsplat_maskgaussian")],
                              python=python, label="World 2.0 Gaussian renderer")
            await pip_install(ctx, ["--no-build-isolation", "-r", str(src / "requirements_git.txt"), "flash-attn"],
                              python=python, label="World 2.0 CUDA extensions")
            await pip_install(ctx, ["--no-build-isolation", str(src / "hyworld2/worldgen/third_party/navmesh")],
                              python=python, label="World 2.0 navigation")
        marker.write_text("ok")
    if adapter == "hunyuan_world_1":
        # Upstream defaults point at Hub IDs. Keep local/custom LoRA imports usable as well.
        for file in [src / "demo_panogen.py", src / "hy3dworld/models/layer_decomposer.py"]:
            text = file.read_text()
            replacement = 'os.environ.get("KUI_WORLD1_WEIGHTS", "tencent/HunyuanWorld-1")'
            text = text.replace('"tencent/HunyuanWorld-1"', replacement) if "KUI_WORLD1_WEIGHTS" not in text else text
            text = text.replace("self.num_inference_steps = 50", 'self.num_inference_steps = int(os.environ.get("KUI_WORLD1_STEPS", "50"))')
            file.write_text(text)
    elif adapter == "hy_worldplay":
        file = src / "wan/generate.py"
        text = file.read_text().replace("seed_idx = 0", 'seed_idx = int(os.environ.get("KUI_WORLDPLAY_SEED", "0"))')
        text = text.replace('if os.path.isfile(args.input):', 'if False:  # KaggleUI always sends a literal prompt')
        text = text.replace('if "@" in model_input:', 'if False:  # Image input uses --image_path')
        file.write_text(text)
    elif adapter == "hy_world_2":
        for file in [src / "hyworld2/worldgen/traj_generate.py", src / "hyworld2/worldgen/traj_render.py",
                     src / "hyworld2/worldgen/src/vlm_utils.py"]:
            text = file.read_text()
            if "KUI_WORLD2_VLM_URL" not in text:
                text = text.replace('api_key="EMPTY"', 'api_key=os.environ.get("KUI_WORLD2_VLM_KEY", "EMPTY")')
                text = text.replace('base_url=f"http://{LLM_ADDR}:{LLM_PORT}/v1"',
                                    'base_url=os.environ.get("KUI_WORLD2_VLM_URL", f"http://{LLM_ADDR}:{LLM_PORT}/v1")')
                if "import os" not in text:
                    text = "import os\n" + text
                file.write_text(text)
    for model in recipe.get("extra_models", []):
        siblings = await repo_files(model["repo"])
        if model.get("mode") == "diffusers":
            chosen = hf.diffusers_files(siblings)
            allow, expected = chosen["files"], chosen["size"]
        elif model.get("mode") == "transformers":
            chosen = hf.transformers_files(siblings)
            allow, expected = chosen["files"], chosen["size"]
        else:
            import fnmatch

            allow = model.get("allow")
            expected = sum(int(s.get("size") or 0) for s in siblings
                           if not allow or any(fnmatch.fnmatch(s["rfilename"], p) for p in allow))
        snapshot = Path(await hf_download(ctx, model["repo"], allow, expected or None))
        if model.get("link"):
            dest = src / model["link"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(snapshot / model.get("subfolder", ""), target_is_directory=True)
