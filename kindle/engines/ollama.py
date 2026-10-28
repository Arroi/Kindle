"""Ollama engine: one shared ``ollama serve`` for all text deployments (GGUF from Ollama, HF or /kaggle/input)."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import tempfile

import httpx

from ..capabilities import profile
from ..proc import ManagedProcess, run_cmd
from ..schemas import DeploymentSpec
from ..util import which
from .base import OpenAIUpstream
from .context import DeployError, EngineContext

OLLAMA_PORT = int(os.environ.get("KINDLE_OLLAMA_PORT", "11434"))


class OllamaServer:
    _instance: OllamaServer | None = None

    def __init__(self) -> None:
        self.proc: ManagedProcess | None = None
        self.base = f"http://127.0.0.1:{OLLAMA_PORT}"
        self.exe: str | None = None
        self._lock = asyncio.Lock()
        self.version: str | None = None

    @classmethod
    def get(cls) -> OllamaServer:
        if cls._instance is None:
            cls._instance = OllamaServer()
        return cls._instance

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{self.base}/api/version")
                if r.status_code == 200:
                    self.version = r.json().get("version")
                    return True
        except httpx.HTTPError:
            pass
        return False

    async def ensure(self, ctx: EngineContext) -> str:
        async with self._lock:
            if await self.healthy():
                return self.base
            self.exe = await self._binary(ctx)
            ctx.progress("installing", None, "Starting Ollama")
            env = {
                "OLLAMA_HOST": f"127.0.0.1:{OLLAMA_PORT}",
                "OLLAMA_MODELS": os.environ.get("OLLAMA_MODELS", str(ctx.paths.ollama_models)),
                "OLLAMA_KEEP_ALIVE": "-1",
                "OLLAMA_MAX_LOADED_MODELS": "4",
                # Ollama reserves the full context for every slot, so slots multiply memory with long
                # contexts. Two keep an agent's side requests moving; the fit check counts both.
                "OLLAMA_NUM_PARALLEL": os.environ.get("KINDLE_OLLAMA_PARALLEL", "2"),
                "OLLAMA_ORIGINS": "*",
            }
            env.pop("CUDA_VISIBLE_DEVICES", None)
            self.proc = ManagedProcess("ollama", [self.exe, "serve"], env=env,
                                       log_path=ctx.paths.logs / "ollama.log")
            await self.proc.start()
            for _ in range(120):
                if await self.healthy():
                    ctx.log(f"Ollama {self.version} is up")
                    return self.base
                if not self.proc.running:
                    raise DeployError("Ollama exited: " + " | ".join(list(self.proc.tail)[-4:]))
                await asyncio.sleep(0.5)
            raise DeployError("Ollama didn't start within 60s")

    async def _binary(self, ctx: EngineContext) -> str:
        exe = which("ollama", [ctx.paths.bin / "ollama-dist" / "bin", ctx.paths.bin])
        if exe:
            return exe
        if platform.system() != "Linux":
            raise DeployError("Ollama isn't installed. Get it from https://ollama.com/download "
                              "(or start Kindle with --mock-engines for UI development).")
        return await install_ollama(ctx)

    async def stop(self) -> None:
        if self.proc:
            await self.proc.stop()


OLLAMA_ARCHIVE = "https://ollama.com/download/ollama-linux-{arch}.tar.zst"
OLLAMA_DISK = 8 * 1000**3  # archive (~1.5 GB) plus the unpacked CUDA runtimes


async def install_ollama(ctx: EngineContext) -> str:
    """Install Ollama into Kindle's own folder, without root.

    Ollama's install.sh needs root or sudo, apt-get for zstd, and writes several GB to /usr/local.
    Hosted notebooks often have neither root nor room there, so every deployment failed. The release
    archive is self-contained (bin/ollama finds its libraries in ../lib/ollama), so it can live here.
    """
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower())
    if not arch:
        raise DeployError(f"Ollama has no Linux build for {platform.machine()}")
    dest = ctx.paths.bin / "ollama-dist"
    staging = ctx.paths.bin / "ollama-dist.partial"
    archive = ctx.paths.cache / f"ollama-linux-{arch}.tar.zst"
    free = shutil.disk_usage(ctx.paths.bin).free
    if free < OLLAMA_DISK:
        raise DeployError(f"Installing Ollama needs about {OLLAMA_DISK / 1e9:.0f} GB of disk; {free / 1e9:.1f} GB is free "
                          f"at {ctx.paths.home}. Free space or set KINDLE_HOME to a bigger disk.")
    url = OLLAMA_ARCHIVE.format(arch=arch)
    ctx.log(f"Installing Ollama from {url} into {dest} (no root needed)")
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(connect=20, read=120, write=60, pool=20)) as c:
            async with c.stream("GET", url) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length") or 0)
                got, last = 0, 0.0
                with archive.open("wb") as out:
                    async for block in r.aiter_bytes(1 << 20):
                        out.write(block)
                        got += len(block)
                        now = asyncio.get_running_loop().time()
                        if now - last > 1:
                            last = now
                            ctx.progress("installing", got / total if total else None,
                                         f"Downloading Ollama: {got / 1e9:.2f}" + (f" / {total / 1e9:.2f} GB" if total else " GB"))
    except httpx.HTTPError as exc:
        archive.unlink(missing_ok=True)
        raise DeployError(f"Couldn't download Ollama ({type(exc).__name__}: {exc}). Is the notebook online?") from exc
    ctx.progress("installing", None, "Unpacking Ollama")
    if _zstd_module() is None:
        from .install import pip_install

        await pip_install(ctx, ["zstandard"], label="the zstd decompressor")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        await asyncio.to_thread(_unpack_zst, archive, staging)
    except Exception as exc:  # noqa: BLE001 - a corrupt download or a full disk
        shutil.rmtree(staging, ignore_errors=True)
        raise DeployError(f"Couldn't unpack Ollama: {type(exc).__name__}: {exc}") from exc
    finally:
        archive.unlink(missing_ok=True)
    exe = staging / "bin" / "ollama"
    if not exe.exists():
        shutil.rmtree(staging, ignore_errors=True)
        raise DeployError("The Ollama archive had no bin/ollama; its layout may have changed.")
    exe.chmod(0o755)
    shutil.rmtree(dest, ignore_errors=True)
    staging.rename(dest)
    ctx.log(f"Ollama installed at {dest}")
    return str(dest / "bin" / "ollama")


def _zstd_module():
    try:
        from compression import zstd  # Python 3.14+

        return lambda f: zstd.ZstdFile(f)
    except ImportError:
        pass
    try:
        import zstandard

        return lambda f: zstandard.ZstdDecompressor().stream_reader(f)
    except ImportError:
        return None


def _unpack_zst(archive, dest) -> None:
    import importlib
    import tarfile

    importlib.invalidate_caches()
    opener = _zstd_module()
    if opener is None:
        raise RuntimeError("no zstd decompressor available")
    dest.mkdir(parents=True)
    with archive.open("rb") as raw, opener(raw) as stream, tarfile.open(fileobj=stream, mode="r|") as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:  # Pythons before the extraction-filter backport
            tar.extractall(dest)  # noqa: S202 - official Ollama release archive


def model_ref(spec: DeploymentSpec) -> str | None:
    m = spec.model
    if m.source == "attached":
        return None
    if m.source == "hf":
        return f"hf.co/{m.id}:{m.variant}" if m.variant else f"hf.co/{m.id}"
    if ":" in m.id:
        return m.id
    return f"{m.id}:{m.variant or 'latest'}"


async def _pull(base: str, ref: str, ctx: EngineContext) -> None:
    ctx.progress("downloading", 0.0, f"Pulling {ref}")
    totals: dict[str, int] = {}
    done: dict[str, int] = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=None, write=60, pool=10)) as c:
        async with c.stream("POST", f"{base}/api/pull", json={"model": ref, "stream": True}) as r:
            if r.status_code != 200:
                raise DeployError(f"Pull failed ({r.status_code}): {(await r.aread()).decode()[:300]}")
            last = 0.0
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("error"):
                    raise DeployError(f"Pull failed: {ev['error']}")
                digest = ev.get("digest")
                if digest and ev.get("total"):
                    totals[digest] = int(ev["total"])
                    done[digest] = int(ev.get("completed") or 0)
                total = sum(totals.values())
                now = asyncio.get_running_loop().time()
                if total and now - last > 0.7:
                    got = sum(done.values())
                    ctx.progress("downloading", min(got / total, 0.999),
                                 f"Pulling {ref}: {got / 1e9:.2f} / {total / 1e9:.2f} GB")
                    last = now
                elif ev.get("status") and not digest:
                    ctx.log(f"ollama: {ev['status']}")
    ctx.progress("downloading", 1.0, f"Pulled {ref}")


async def deploy(spec: DeploymentSpec, ctx: EngineContext) -> tuple[OpenAIUpstream, dict]:
    server = OllamaServer.get()
    base = await server.ensure(ctx)
    ref = model_ref(spec)
    if ref:
        await _pull(base, ref, ctx)
    alias = f"kui-{ctx.dep_id}"
    num_ctx = int(spec.options.get("ctx") or 8192)
    ctx.progress("loading", None, "Creating model with your settings")
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=1800, write=600, pool=10)) as c:
        if ref:
            r = await c.post(f"{base}/api/create", json={
                "model": alias, "from": ref, "parameters": {"num_ctx": num_ctx}, "stream": False})
            if r.status_code != 200:
                raise DeployError(f"Couldn't create model: {r.text[:300]}")
        else:
            # Attached GGUF on disk: the CLI uploads local files into Ollama's blob store.
            mf = tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False)
            path = spec.model.id
            if any(c in path for c in ('\n', '\r', '"')):
                raise DeployError("The model path contains unsupported quote or newline characters")
            mf.write(f'FROM "{path}"\nPARAMETER num_ctx {num_ctx}\n')
            mf.close()
            rc = await run_cmd([server.exe or "ollama", "create", alias, "-f", mf.name],
                               env={"OLLAMA_HOST": base.replace("http://", "")}, on_line=ctx.log)
            os.unlink(mf.name)
            if rc != 0:
                raise DeployError("ollama create failed for the attached GGUF")
        r = await c.post(f"{base}/api/show", json={"model": alias})
        metadata = r.json() if r.status_code == 200 else {}
        caps = metadata.get("capabilities", [])
        detected = profile(spec.model.id, tags=caps, thinking=metadata.get("thinking"), authoritative=True)
        ctx.progress("loading", None, "Loading into GPU memory")
        if "embedding" in caps and "completion" not in caps:
            r = await c.post(f"{base}/api/embed", json={"model": alias, "input": "warmup", "keep_alive": -1})
        else:
            r = await c.post(f"{base}/api/generate", json={"model": alias, "prompt": "", "keep_alive": -1})
        if r.status_code != 200:
            raise DeployError(f"Loading failed: {r.text[:300]}")
    info = {"capabilities": caps, "upstream": alias, "ref": ref or spec.model.id, "num_ctx": num_ctx,
            "ollama_version": server.version, **detected}
    return OpenAIUpstream(base, alias, native=frozenset({"anthropic", "ollama"}), profile=detected), info


async def undeploy(dep_id: str) -> None:
    server = OllamaServer.get()
    alias = f"kui-{dep_id}"
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            await c.post(f"{server.base}/api/generate", json={"model": alias, "keep_alive": 0})
            await c.request("DELETE", f"{server.base}/api/delete", json={"model": alias})
    except httpx.HTTPError:
        pass
