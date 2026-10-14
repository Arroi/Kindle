"""Download worker. Its whole process group is stopped on cancellation/shutdown."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import httpx

from ..util import dir_size
from . import attached, hf, world
from .imports import PREFIX


def report(**data) -> None:
    print(PREFIX + json.dumps(data), flush=True)


def check_space(root: Path, required: int) -> None:
    if required > max(0, shutil.disk_usage(root).free - 256 * 1024**2):
        raise ValueError("Not enough free disk space for this model download")


def extract(archive: Path, dest: Path) -> None:
    """Extract regular files only, after validating every member and its expanded size."""
    def target(name: str) -> Path:
        path = dest / name
        if "\\" in name or not path.resolve().is_relative_to(dest.resolve()):
            raise ValueError("Archive contains an unsafe file path")
        return path

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            entries = bundle.infolist()
            if len(entries) > 100_000:
                raise ValueError("Archive has too many files")
            for entry in entries:
                target(entry.filename)
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Archive contains symbolic links")
            check_space(dest, sum(e.file_size for e in entries))
            for entry in entries:
                out = target(entry.filename)
                if entry.is_dir():
                    out.mkdir(parents=True, exist_ok=True)
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(entry) as src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 1024**2)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            entries = bundle.getmembers()
            if len(entries) > 100_000:
                raise ValueError("Archive has too many files")
            for entry in entries:
                target(entry.name)
                if not entry.isfile() and not entry.isdir():
                    raise ValueError("Archive contains links or special files")
            check_space(dest, sum(e.size for e in entries))
            for entry in entries:
                out = target(entry.name)
                if entry.isdir():
                    out.mkdir(parents=True, exist_ok=True)
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(entry) as src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 1024**2)
    else:
        return
    archive.unlink()


def direct(url: str, root: Path) -> None:
    headers = {}
    if urlsplit(url).hostname == "huggingface.co" and os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    with httpx.Client(timeout=httpx.Timeout(60, connect=20), follow_redirects=True) as client:
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("content-type", ""):
                raise ValueError("That URL returns a web page. Use a direct download link or a repository URL.")
            name = unquote(urlsplit(url).path.rsplit("/", 1)[-1]) or "model"
            disposition = response.headers.get("content-disposition", "")
            match = re.search(r'filename="?([^";]+)', disposition)
            if match:
                name = match.group(1).strip()
            name = Path(name.replace("\\", "/")).name
            if name in ("", ".", "..") or len(name) > 240:
                name = "model"
            total = int(response.headers.get("content-length") or 0)
            check_space(root, total)
            temp = root / (name + ".part")
            got, last = 0, 0.0
            with temp.open("wb") as output:
                for chunk in response.iter_bytes(1024**2):
                    check_space(root, len(chunk))
                    output.write(chunk)
                    got += len(chunk)
                    if time.monotonic() - last > 0.3:
                        report(status="downloading", bytes=got, progress=min(got / total, 0.99) if total else None,
                               message=f"Downloading {name}")
                        last = time.monotonic()
            archive = root / name
            temp.rename(archive)
            if zipfile.is_zipfile(archive) or tarfile.is_tarfile(archive):
                report(status="extracting", progress=None, message="Extracting model files")
                # Separate output prevents members from overwriting the open archive.
                dest = root / "extracted"
                dest.mkdir()
                extract(archive, dest)


def download(job: dict, root: Path) -> dict:
    root.mkdir(parents=True)
    profile = world.profile(job.get("repo", ""))
    if profile and profile.get("recipe"):
        # Upstream LingBot selects camera/action conditioning from the checkpoint directory name.
        root = root / profile["id"].split("/")[-1]
        root.mkdir()
    if job["kind"] == "hf":
        from huggingface_hub import snapshot_download

        revision = job.get("revision") or (profile or {}).get("options", {}).get("revision")
        with httpx.Client(timeout=30) as client:
            info = client.get(hf.API + "/models/" + job["repo"] +
                              ("/revision/" + quote(revision, safe="") if revision else ""),
                              params={"blobs": "true"}, headers=hf._headers())
            info.raise_for_status()
            siblings = info.json().get("siblings", [])
        allow = None
        if profile and profile.get("recipe"):
            from ..engines.world_engine import list_recipes

            recipe = next(r for r in list_recipes() if r["name"] == profile["recipe"])
            allow = [s["rfilename"] for s in siblings
                     if any(fnmatch.fnmatch(s["rfilename"], pattern) for pattern in recipe.get("allow", ["*"]))]
        elif any(s["rfilename"] == "model_index.json" for s in siblings):
            allow = hf.diffusers_files(siblings)["files"]
        elif "kokoro" in job["repo"].lower():
            allow = [s["rfilename"] for s in siblings if s["rfilename"].endswith((".json", ".pth")) or
                     (s["rfilename"].startswith("voices/") and s["rfilename"].endswith(".pt"))]
        elif any(s["rfilename"] == "config.json" for s in siblings) and not profile:
            allow = hf.transformers_files(siblings)["files"]
        check_space(root, sum(int(s.get("size") or 0) for s in siblings if not allow or s["rfilename"] in allow))
        snapshot_download(job["repo"], revision=revision, local_dir=root, allow_patterns=allow,
                          token=os.environ.get("HF_TOKEN") or None)
    elif job["kind"] == "git":
        if not shutil.which("git"):
            raise ValueError("Git is required to download a Git repository")
        command = ["git", "-c", "credential.helper=", "clone", "--depth", "1", "--progress"]
        if job.get("revision"):
            command += ["--branch", job["revision"]]
        command += ["--", job["url"], str(root)]
        subprocess.run(command, check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        for directory, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d != ".git"]
            for name in files:
                path = Path(directory) / name
                if path.is_file() and path.stat().st_size < 1024:
                    with path.open("rb") as file:
                        if file.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
                            raise ValueError("Repository contains Git LFS pointers instead of weights. Install git-lfs or use the Hugging Face model URL.")
    else:
        direct(job["url"], root)
    report(status="scanning", progress=None, message="Finding runnable models")
    if profile and profile.get("engine"):
        models = [{"path": str(root), "name": job["name"], "modality": "world", "engine": profile["engine"],
                   "size": dir_size(root), "options": profile.get("options", {}), "recipe": profile.get("recipe"),
                   "params": profile.get("params_b", 0) * 1e9 or None, "experimental": True,
                   "min_sm": profile.get("min_sm"),
                   "min_vram": int(float(profile.get("min_vram_gb", 0)) * 1e9) or None}]
    else:
        models = attached.scan(root)
        if profile:  # custom world pipelines must not be misclassified as generic image/video pipelines
            models = []
    for model in models:
        model.update(url=job["url"], imported=True)
        if len(models) == 1:
            model["name"] = job["name"]
    count = sum(bool(m.get("engine")) for m in models)
    return {"models": models, "message": f"Downloaded · {count} runnable model(s)" if count else
            "Files downloaded. This repository's format has no configured runner yet."}


if __name__ == "__main__":
    job, root, result = json.loads(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    try:
        answer = download(job, root)
    except Exception as exc:
        answer = {"error": f"{type(exc).__name__}: {exc}"}
    result.write_text(json.dumps(answer))
