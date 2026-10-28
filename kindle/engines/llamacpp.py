"""llama.cpp engine: one GGUF model on this session's GPUs, optionally pooled with linked sessions.

Pooling uses llama.cpp's RPC backend. Each helper session (a linked session with "Share this GPU"
on) runs an RPC server; this session runs ``llama-server --rpc`` and splits the model's layers
across every GPU. RPC traffic travels over the helpers' public links (see ``pool/bridge.py``).
Each generated token makes a round trip to every helper, so a pool trades speed for capacity:
use it for models that don't fit one session.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..capabilities import profile
from ..catalog import hf
from ..pool import binaries
from ..pool.bridge import Bridge
from ..pool.gguf import GGUFError, layer_devices, read_header
from ..proc import ManagedProcess
from ..schemas import DeploymentSpec
from ..util import free_port
from .base import OpenAIUpstream
from .context import DeployError, EngineContext
from .download import hf_download, repo_files

GiB = 1024**3
HASH_THRESHOLD = 10 * 1024 * 1024  # llama.cpp only looks up cached tensors larger than this
HELPER_RESERVE = 2 * GiB  # KV cache and compute buffers on each device
LOCAL_RESERVE = 4 * GiB  # plus the output layer and the head's own buffers
POLL_SECONDS = 2


@dataclass
class Device:
    owner: str  # link id, or "local"
    name: str
    free: int


def plan_split(devices: list[Device]) -> list[float]:
    """``--tensor-split`` weights: free VRAM minus headroom, in llama.cpp's device order."""
    return [round(max(d.free - (LOCAL_RESERVE if d.owner == "local" else HELPER_RESERVE), GiB) / GiB, 2)
            for d in devices]


def seed_plan(shards: list[tuple[str, list]], block_count: int, split: list[float],
              devices: list[Device]) -> dict[str, list[dict]]:
    """The tensors each helper will hold, as byte ranges of the repo's GGUF files, keyed by link id."""
    where = layer_devices(block_count, split)
    plan: dict[str, list[dict]] = {}
    for rel, tensors in shards:
        for t in tensors:
            if t.layer is None or t.layer >= block_count or not t.size or t.size <= HASH_THRESHOLD:
                continue
            owner = devices[where[t.layer]].owner
            if owner != "local":
                plan.setdefault(owner, []).append({"file": rel, "offset": t.offset, "size": t.size, "name": t.name})
    return plan


class LlamaCppBackend(OpenAIUpstream):
    """Also owns the pool: bridges to helpers, released when the deployment stops."""

    def __init__(self, base: str, name: str, profile: dict, bridges: list[Bridge], helpers: list):
        super().__init__(base, name, template_thinking=True, profile=profile)
        self.bridges = bridges
        self.helpers = helpers

    async def close(self) -> None:
        await super().close()
        await release(self.bridges, self.helpers)


async def release(bridges: list[Bridge], helpers: list) -> None:
    await asyncio.gather(*(b.close() for b in bridges), return_exceptions=True)
    await asyncio.gather(*(link.client.post(f"{link.url}/_kui/pool/release", timeout=20)
                           for link in helpers if link.client), return_exceptions=True)


def pick_helpers(ctx: EngineContext, choice) -> list:
    """Linked sessions to pool with: ``"all"`` sharing sessions, or a list of link ids."""
    if not choice or ctx.app is None:
        return []
    links = [link for link in ctx.app.links.links.values() if link.status == "up"]
    if choice != "all":
        wanted = set(choice) if isinstance(choice, list) else {str(choice)}
        missing = wanted - {link.id for link in links}
        if missing:
            raise DeployError(f"Linked session(s) {', '.join(sorted(missing))} aren't connected. Check Linked sessions.")
        links = [link for link in links if link.id in wanted]
    sharing = [link for link in links if (link.pool or {}).get("enabled")]
    if choice != "all" and len(sharing) != len(links):
        off = [link.url for link in links if link not in sharing]
        raise DeployError(f"{', '.join(off)} {'is' if len(off) == 1 else 'are'}n't sharing a GPU. Turn on "
                          "'Share this GPU' in that session's Linked sessions tab.")
    if not sharing:
        raise DeployError("No linked session is sharing its GPU. Link another session and turn on 'Share this GPU' there.")
    return sharing


async def _model_files(spec: DeploymentSpec, ctx: EngineContext) -> tuple[Path, list[tuple[str, Path]], str | None]:
    """(first shard, [(repo-relative name, local path)], revision)."""
    if spec.model.source == "attached":
        path = Path(spec.model.id)
        if not path.is_file():
            raise DeployError(f"{path} isn't a GGUF file")
        return path, [(path.name, path)], None
    revision = spec.options.get("revision")
    variants = hf.gguf_variants(await repo_files(spec.model.id, revision=revision))
    if not variants:
        raise DeployError(f"{spec.model.id} has no GGUF files. Pick a GGUF repo (for example from unsloth or bartowski).")
    want = (spec.model.variant or "").upper()
    chosen = next((v for v in variants if v["quant"] == want), None)
    if chosen is None:
        raise DeployError(f"{spec.model.id} has no {want or 'default'} quant. Available: "
                          + ", ".join(v["quant"] for v in variants))
    root = Path(await hf_download(ctx, spec.model.id, chosen["files"], chosen["size"], revision=revision))
    shards = [(name, root / name) for name in sorted(chosen["files"])]
    return shards[0][1], shards, revision


async def _prepare_helpers(ctx: EngineContext, helpers: list, tag: str, seeds: dict[str, list[dict]],
                           repo: str | None, revision: str | None) -> None:
    """Ask each helper to install the same llama.cpp, start its RPC server and seed its share."""
    for link in helpers:
        tensors = seeds.get(link.id) or []
        seed = {"repo": repo, "revision": revision, "tensors": tensors} if repo and tensors else None
        r = await link.client.post(f"{link.url}/_kui/pool/prepare", json={"tag": tag, "seed": seed}, timeout=30)
        if r.status_code >= 400:
            raise DeployError(f"{link.url} refused to join the pool: {r.text[:300]}")
    started = time.monotonic()
    while True:
        statuses = []
        for link in helpers:
            try:
                r = await link.client.get(f"{link.url}/_kui/pool/status", timeout=20)
                statuses.append(r.json() if r.status_code == 200 else {"status": "error", "error": r.text[:200]})
            except httpx.HTTPError as exc:
                statuses.append({"status": "error", "error": f"unreachable ({type(exc).__name__})"})
        failed = [(link, s) for link, s in zip(helpers, statuses) if s.get("status") == "error"]
        if failed:
            raise DeployError("Pool helper failed: " + "; ".join(f"{link.url}: {s.get('error')}" for link, s in failed))
        if all(s.get("status") == "ready" for s in statuses):
            return
        fractions = [s.get("progress") or 0 for s in statuses]
        busy = next((s for s in statuses if s.get("status") != "ready"), {})
        ctx.progress("downloading", sum(fractions) / len(fractions),
                     f"Preparing {len(helpers)} pool session(s): {busy.get('message', '')}")
        if time.monotonic() - started > 6 * 3600:
            raise DeployError("Pool helpers didn't finish preparing within 6 hours")
        await asyncio.sleep(POLL_SECONDS)


async def deploy(spec: DeploymentSpec, ctx: EngineContext) -> tuple[OpenAIUpstream, dict]:
    if ctx.hw.accelerator == "tpu" or not ctx.hw.gpus:
        raise DeployError("The llama.cpp engine needs a CUDA GPU.")
    opts = spec.options
    tag = str(opts.get("llamacpp_tag") or binaries.tag())
    ctx.progress("installing", None, f"Installing llama.cpp {tag}")
    try:
        build = await binaries.ensure(ctx.paths.bin, tag, ctx.log)
    except (binaries.BinaryError, httpx.HTTPError) as exc:
        raise DeployError(f"Couldn't install llama.cpp: {exc}") from exc
    first, shards, revision = await _model_files(spec, ctx)
    helpers = pick_helpers(ctx, opts.get("pool"))

    split: list[float] | None = None
    bridges: list[Bridge] = []
    if helpers:
        devices = [Device(link.id, g["name"], int(g.get("vram_free") or 0)) for link in helpers
                   for g in (link.pool or {}).get("gpus") or []]
        devices += [Device("local", g.name, g.vram_free) for g in ctx.hw.gpus if g.index in (ctx.gpus or [g.index])]
        split = [float(x) for x in opts["tensor_split"]] if opts.get("tensor_split") else plan_split(devices)
        if len(split) != len(devices):
            raise DeployError(f"tensor_split needs {len(devices)} values, one per pooled GPU")
        need = sum(p.stat().st_size for _, p in shards)
        have = sum(max(d.free - HELPER_RESERVE, 0) for d in devices)
        ctx.log("Pool: " + ", ".join(f"{d.name} ({d.owner}, {d.free / GiB:.0f} GiB free)" for d in devices)
                + f"; split {split}")
        if need > have and not opts.get("cpu_moe"):
            raise DeployError(f"The model needs {need / GiB:.0f} GiB but the pool has about {have / GiB:.0f} GiB free. "
                              "Link more sessions, choose a smaller quant, or set cpu_moe to keep some experts in RAM.")
        seeds: dict[str, list[dict]] = {}
        if spec.model.source != "attached" and opts.get("seed_cache", True):
            try:
                headers = [(rel, read_header(path).tensors) for rel, path in shards]
                blocks = read_header(first).block_count
                if blocks:
                    seeds = seed_plan(headers, blocks, split, devices)
            except (GGUFError, OSError) as exc:
                ctx.log(f"Couldn't plan weight seeding ({exc}); helpers will receive weights over the link.")
        for owner, tensors in seeds.items():
            ctx.log(f"Helper {owner} will download {len(tensors)} tensors "
                    f"({sum(t['size'] for t in tensors) / GiB:.1f} GiB) from Hugging Face itself")
        try:
            await _prepare_helpers(ctx, helpers, tag, seeds, spec.model.id if spec.model.source != "attached" else None,
                                   revision)
            for link in helpers:
                bridge = Bridge(link.url, link.api_key, ctx.log)
                await bridge.start()
                bridges.append(bridge)
        except BaseException:
            await release(bridges, helpers)
            raise

    port = free_port()
    cmd = [str(build.server), "-m", str(first), "--host", "127.0.0.1", "--port", str(port),
           "-c", str(int(opts.get("ctx") or 8192)), "-ngl", "999", "--jinja", "--alias", ctx.served_name,
           "--parallel", str(int(opts.get("parallel") or 4))]
    if int(opts.get("parallel") or 4) > 1:
        # One shared KV cache: each request can use the whole window instead of ctx / parallel.
        cmd.append("--kv-unified")
    if bridges:
        cmd += ["--rpc", ",".join(f"127.0.0.1:{b.port}" for b in bridges)]
    if split:
        cmd += ["--tensor-split", ",".join(f"{x:g}" for x in split)]
    if opts.get("cpu_moe"):
        cmd += ["--n-cpu-moe", str(int(opts["cpu_moe"]))]
    cmd += [str(a) for a in opts.get("extra_args") or []]
    env = {**ctx.env(build.env())}
    proc = ManagedProcess(f"llamacpp-{ctx.dep_id}", cmd, env=env, log_path=ctx.paths.logs / f"{ctx.dep_id}-llamacpp.log",
                          on_line=ctx.log)
    ctx.processes.append(proc)
    where = f"{len(bridges) + 1} sessions" if bridges else "this session"
    ctx.progress("loading", None, f"Loading weights on {where}"
                 + (" (weights that weren't seeded travel over the link, which can take a while)" if bridges else ""))
    try:
        await proc.start()
        await _wait_ready(proc, port, ctx, hours=float(opts.get("load_timeout_hours") or (8 if bridges else 1)))
    except BaseException:
        await release(bridges, helpers)
        raise
    detected = profile(spec.model.id)
    detected["capabilities"] = sorted({*detected["capabilities"], "tools"})
    info = {"port": port, "path": str(first), "llamacpp": tag, "num_ctx": int(opts.get("ctx") or 8192),
            "pool": [link.url for link in helpers], "tensor_split": split, "device": "gpu", **detected}
    return LlamaCppBackend(f"http://127.0.0.1:{port}", ctx.served_name, detected, bridges, helpers), info


async def _wait_ready(proc: ManagedProcess, port: int, ctx: EngineContext, hours: float) -> None:
    started = last_note = time.monotonic()
    async with httpx.AsyncClient(timeout=10) as client:
        while time.monotonic() - started < hours * 3600:
            if not proc.running:
                raise DeployError("llama-server exited: " + " | ".join(list(proc.tail)[-5:]))
            try:
                r = await client.get(f"http://127.0.0.1:{port}/health")
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() - last_note > 30:
                last_note = time.monotonic()
                ctx.progress("loading", None, f"Loading weights ({(last_note - started) / 60:.0f} min so far)")
            await asyncio.sleep(2)
    raise DeployError(f"llama-server didn't finish loading within {hours:g} hours")
