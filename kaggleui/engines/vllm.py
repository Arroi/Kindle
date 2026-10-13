"""vLLM engine (isolated venv; tensor-parallel across both T4s). Experimental on Turing GPUs."""

from __future__ import annotations

import asyncio

import httpx

from ..catalog import hf
from ..proc import ManagedProcess
from ..schemas import DeploymentSpec
from ..util import free_port
from .base import OpenAIUpstream
from .context import DeployError, EngineContext
from .download import hf_download, repo_files
from .install import ensure_venv


async def deploy(spec: DeploymentSpec, ctx: EngineContext) -> tuple[OpenAIUpstream, dict]:
    if (ctx.hw.min_sm or 0) < 70:
        raise DeployError("vLLM needs a T4 or newer GPU. Use the Ollama or Transformers engine instead.")
    py = await ensure_venv(ctx, "vllm", ["vllm"], python="3.12", label="vLLM")
    if spec.model.source == "attached":
        path = spec.model.id
    else:
        sel = hf.transformers_files(await repo_files(spec.model.id))
        path = await hf_download(ctx, spec.model.id, sel["files"], sel["size"])
    port = free_port()
    opts = spec.options
    tp = max(1, len(ctx.gpus))
    cmd = [py, "-m", "vllm.entrypoints.openai.api_server", "--model", path,
           "--served-model-name", ctx.served_name, "--host", "127.0.0.1", "--port", str(port),
           "--dtype", str(opts.get("dtype") or ("half" if (ctx.hw.min_sm or 0) < 80 else "auto")),
           "--max-model-len", str(int(opts.get("ctx") or 8192)),
           "--gpu-memory-utilization", str(float(opts.get("gpu_memory_utilization") or 0.90)),
           "--tensor-parallel-size", str(tp)]
    if opts.get("enforce_eager", (ctx.hw.min_sm or 0) < 80):
        cmd.append("--enforce-eager")
    if opts.get("quantization"):
        cmd += ["--quantization", str(opts["quantization"])]
    if opts.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    proc = ManagedProcess(f"vllm-{ctx.dep_id}", cmd, env=ctx.env(), log_path=ctx.paths.logs / f"{ctx.dep_id}-vllm.log",
                          on_line=ctx.log)
    ctx.processes.append(proc)
    ctx.progress("loading", None, "Starting vLLM (loading weights + warming up)")
    await proc.start()
    base = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient(timeout=5) as c:
        for _ in range(1800):
            if not proc.running:
                raise DeployError("vLLM exited: " + " | ".join(list(proc.tail)[-5:]))
            try:
                r = await c.get(f"{base}/health")
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
        else:
            raise DeployError("vLLM didn't become ready within 30 minutes")
    return OpenAIUpstream(base, ctx.served_name, template_thinking=True), {"port": port, "tensor_parallel": tp, "path": path}
