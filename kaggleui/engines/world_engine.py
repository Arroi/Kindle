"""World models via recipes: clone the upstream repo, install its deps, download weights, run per job."""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

import yaml

from ..proc import run_cmd
from ..schemas import DeploymentSpec
from .base import Backend, WorkerBackend
from .context import DeployError, EngineContext
from .download import hf_download, repo_files
from .install import ensure_venv, pip_install, worker_pythonpath
from .workers_engine import WORKER_BASE_REQS, start_worker

RECIPES = Path(__file__).with_name("recipes") / "world"


def list_recipes() -> list[dict]:
    return [yaml.safe_load(p.read_text()) for p in sorted(RECIPES.glob("*.yaml"))]


def recipe_for(spec: DeploymentSpec) -> dict:
    name = spec.options.get("recipe")
    for r in list_recipes():
        if r["name"] == name or r["repo"] == spec.model.id:
            return r
    raise DeployError(f"No world recipe for {spec.model.id}")


async def deploy(spec: DeploymentSpec, ctx: EngineContext) -> tuple[Backend, dict]:
    recipe = recipe_for(spec)
    src_root = ctx.paths.envs / "src" / recipe["name"]
    if not (src_root / ".git").exists():
        ctx.progress("installing", None, f"Cloning {recipe['git']}")
        src_root.parent.mkdir(parents=True, exist_ok=True)
        rc = await run_cmd(["git", "clone", "--depth", "1", recipe["git"], str(src_root)], on_line=ctx.log)
        if rc != 0:
            raise DeployError(f"Couldn't clone {recipe['git']}")
    src = src_root / recipe.get("subdir", "")

    extra_env: dict[str, str] = {}
    if recipe.get("env") == "venv":
        python = await ensure_venv(ctx, f"world-{recipe['name']}", recipe.get("requirements", []) + WORKER_BASE_REQS,
                                   system_site_packages=bool(recipe.get("system_site_packages")),
                                   label=recipe["title"])
        extra_env["PYTHONPATH"] = worker_pythonpath(ctx)
    else:
        python = sys.executable
        await pip_install(ctx, recipe.get("requirements", []), label=recipe["title"])
    marker = src / ".kaggleui-setup"
    if recipe.get("setup") and not marker.exists():
        for step in recipe["setup"]:
            rc = await run_cmd(step.format(python=python), cwd=src, on_line=ctx.log)
            if rc != 0:
                raise DeployError(f"Setup step failed: {step}")
        marker.write_text("ok")

    siblings = await repo_files(recipe["repo"])
    expected = sum(int(s.get("size") or 0) for s in siblings
                   if any(fnmatch.fnmatch(s["rfilename"], p) for p in recipe.get("allow", ["*"])))
    weights = await hf_download(ctx, recipe["repo"], recipe.get("allow"), expected or None)

    _, base = await start_worker(ctx, python=python, module="world", model=weights, engine="world_recipe",
                                 options={**spec.options, "recipe": recipe["name"], "src": str(src)},
                                 extra_env=extra_env, label=recipe["title"])
    return WorkerBackend("world", base), {"recipe": recipe["name"], "controls": recipe.get("controls"),
                                          "start_frame": recipe.get("start_frame"),
                                          "unverified": bool(recipe.get("unverified"))}
