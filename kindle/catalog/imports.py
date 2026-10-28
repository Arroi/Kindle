"""Persistent, cancellable downloads of user-supplied model URLs."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import unquote, urlsplit

from ..proc import ManagedProcess
from ..util import dir_size

PREFIX = "KUI_IMPORT "
ACTIVE = {"queued", "downloading", "extracting", "scanning", "abliterating"}
REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def parse_source(url: str, kind: str = "auto", revision: str | None = None) -> dict:
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise ValueError("Enter an HTTP or HTTPS URL without embedded credentials")
    if kind not in ("auto", "hf", "git", "file"):
        raise ValueError("Unknown download type")
    segments = [unquote(s) for s in parts.path.strip("/").split("/") if s]
    if kind == "hf" or (kind == "auto" and parts.hostname == "huggingface.co" and len(segments) >= 2
                        and (len(segments) == 2 or segments[2] == "tree")):
        if parts.hostname != "huggingface.co" or len(segments) < 2 or segments[0] in ("datasets", "spaces"):
            raise ValueError("Use a Hugging Face model repository URL")
        repo = "/".join(segments[:2]).removesuffix(".git")
        if len(segments) > 2 and segments[2] != "tree":
            raise ValueError("Choose Direct file for a Hugging Face resolve/file URL")
        return {"url": url, "kind": "hf", "repo": repo,
                "revision": revision or ("/".join(segments[3:]) if len(segments) > 3 else None)}
    if kind == "auto":
        kind = "git" if parts.path.endswith(".git") or parts.hostname in ("github.com", "gitlab.com", "bitbucket.org") else "file"
    if kind == "git" and parts.hostname == "github.com" and len(segments) != 2:
        raise ValueError("Use the Git repository root URL, or choose Direct file for an archive")
    return {"url": url, "kind": kind, "revision": revision}


def saved_models(root: Path | None) -> list[dict]:
    found = []
    if root and root.exists():
        for manifest in root.glob("*/manifest.json"):
            try:
                data = json.loads(manifest.read_text())
                if data.get("status") == "complete":
                    found.extend(m for m in data.get("models", []) if Path(m["path"]).exists())
            except (OSError, ValueError, KeyError):
                continue
    return found


class ModelImports:
    def __init__(self, root: Path, envs: Path | None = None):
        self.root = root
        self.envs = envs or root.parent.parent / "envs"
        self._gpu = asyncio.Semaphore(1)  # one abliteration at a time: it needs the GPUs
        root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self._sem = asyncio.Semaphore(2)
        for path in root.glob("*/manifest.json"):
            try:
                job = json.loads(path.read_text())
                if job["status"] in ACTIVE:
                    job.update(status="cancelled", message="Download interrupted by server restart")
                    path.write_text(json.dumps(job))
                self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError):
                continue

    def list(self) -> list[dict]:
        return sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)

    def _save(self, job: dict) -> None:
        path = self.root / job["id"] / "manifest.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job))
        tmp.replace(path)

    def create(self, url: str, kind: str = "auto", name: str = "", revision: str | None = None) -> dict:
        source = parse_source(url, kind, revision)
        ident = uuid.uuid4().hex[:16]
        job = {**source, "id": ident, "name": name.strip()[:120] or source.get("repo") or urlsplit(url).path.rsplit("/", 1)[-1],
               "created": time.time(), "status": "queued", "progress": None, "bytes": 0, "message": "Waiting to download", "models": []}
        (self.root / ident).mkdir()
        self.jobs[ident] = job
        self._save(job)
        self.tasks[ident] = asyncio.create_task(self._run(job))
        return job

    def abliterate(self, repo: str, method: str = "quick", trials: int = 40, quantize: bool = False,
                   name: str = "", revision: str | None = None) -> dict:
        """Queue a job that removes a text model's refusal direction and imports the result."""
        repo = repo.strip().removeprefix("https://huggingface.co/").strip("/")
        if not REPO_ID.match(repo):
            raise ValueError("Enter a Hugging Face model id like Qwen/Qwen3-4B-Instruct-2507")
        if method not in ("quick", "heretic"):
            raise ValueError("Method must be quick or heretic")
        ident = uuid.uuid4().hex[:16]
        job = {"id": ident, "kind": "abliterate", "url": f"https://huggingface.co/{repo}", "repo": repo,
               "revision": revision, "method": method, "trials": max(10, min(int(trials), 300)),
               "quantize": bool(quantize), "name": name.strip()[:120] or f"{repo.split('/')[-1]}-abliterated",
               "created": time.time(), "status": "queued", "progress": None, "bytes": 0,
               "message": "Waiting for the GPUs", "models": []}
        (self.root / ident).mkdir()
        self.jobs[ident] = job
        self._save(job)
        self.tasks[ident] = asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: dict) -> None:
        folder = self.root / job["id"]
        data = folder / "files"
        result = folder / "result.json"
        proc = None
        try:
            abliterate = job.get("kind") == "abliterate"
            async with (self._gpu if abliterate else self._sem):
                job.update(status="downloading", message="Starting download" if not abliterate else "Starting")
                self._save(job)

                def on_line(line: str) -> None:
                    if line.startswith(PREFIX):
                        try:
                            update = json.loads(line[len(PREFIX):])
                            job.update({k: v for k, v in update.items() if k in ("status", "progress", "message", "bytes")})
                        except ValueError:
                            pass

                worker = "kindle.catalog.abliterate_worker" if abliterate else "kindle.catalog.import_worker"
                spec = {**job, "envs": str(self.envs)} if abliterate else job
                proc = ManagedProcess("import-" + job["id"], [sys.executable, "-m", worker,
                                      json.dumps(spec), str(data), str(result)], env={"PYTHONUNBUFFERED": "1"}, on_line=on_line)
                await proc.start()
                while proc.running:
                    await asyncio.sleep(2)
                    job["bytes"] = await asyncio.to_thread(dir_size, data)
                rc = await proc.wait()
                answer = json.loads(result.read_text()) if result.exists() else {"error": f"Download process exited ({rc})"}
                if answer.get("error"):
                    raise ValueError(answer["error"])
                job.update(status="complete", progress=1, models=answer["models"], message=answer["message"])
        except asyncio.CancelledError:
            job.update(status="cancelled", progress=None, message="Cancelled")
        except Exception as exc:
            job.update(status="error", progress=None, message=str(exc))
        finally:
            if proc:
                await proc.stop(timeout=2)
            if job["status"] != "complete":
                shutil.rmtree(data, ignore_errors=True)
            result.unlink(missing_ok=True)
            self._save(job)

    async def cancel(self, ident: str) -> dict:
        job = self.jobs[ident]
        task = self.tasks.get(ident)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # A queued task can be cancelled before its coroutine starts.
            if job["status"] in ACTIVE:
                job.update(status="cancelled", message="Download cancelled", progress=None)
                self._save(job)
        return job

    async def close(self) -> None:
        await asyncio.gather(*(self.cancel(ident) for ident in self.tasks))
