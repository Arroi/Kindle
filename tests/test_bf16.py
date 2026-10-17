"""fp32 pipelines (official Wan/LTX repos) are sized and stored as BF16."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from kaggleui.catalog import hf
from kaggleui.catalog.service import CatalogService
from kaggleui.engines import workers_engine
from kaggleui.engines.context import EngineContext
from kaggleui.engines.workers_engine import download_model
from kaggleui.schemas import DeploymentSpec

GB = 1024**3
FILES = [{"rfilename": "model_index.json", "size": 500},
         {"rfilename": "transformer/config.json", "size": 300},
         {"rfilename": "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "size": 10 * GB},
         {"rfilename": "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "size": 10 * GB},
         {"rfilename": "text_encoder/model.safetensors", "size": 6 * GB},
         {"rfilename": "vae/diffusion_pytorch_model.safetensors", "size": 1 * GB}]
REAL_PLAN = hf.bf16_plan


async def test_plan_halves_fp32_components_and_keeps_vae(monkeypatch):
    dtypes = {"transformer": {"F32": 100}, "text_encoder": {"BF16": 100}}

    async def headers(client, repo, filename, revision=None):
        return dtypes[filename.split("/")[0]]

    monkeypatch.setattr(hf, "safetensors_dtypes", headers)
    plan = await REAL_PLAN(None, "Wan-AI/x", FILES)
    assert plan["saved"] == 10 * GB  # 20 GB of fp32 transformer → 10 GB
    assert plan["largest"] == 10 * GB
    assert "vae/diffusion_pytorch_model.safetensors" not in plan["files"]
    assert len(plan["files"]) == 3


async def test_plan_skips_pipelines_already_in_16_bit(monkeypatch):
    monkeypatch.setattr(hf, "safetensors_dtypes", AsyncMock(return_value={"BF16": 100}))
    assert await REAL_PLAN(None, "Qwen/x", FILES) is None


async def test_catalog_sizes_with_the_bf16_plan(hw, monkeypatch):
    service = CatalogService(hw)
    monkeypatch.setattr(hf, "model_info", AsyncMock(return_value={
        "id": "Wan-AI/x", "pipeline_tag": "text-to-video", "library_name": "diffusers", "tags": ["diffusers"],
        "siblings": FILES}))
    monkeypatch.setattr(hf, "bf16_plan", AsyncMock(return_value={"files": [], "size": 7 * GB, "largest": 5 * GB,
                                                                  "saved": 10 * GB}))
    try:
        detail = await service.detail("hf:Wan-AI/x")
    finally:
        await service.close()
    assert detail["size"] == 7 * GB and detail["largest"] == 5 * GB and detail["bf16_saved"] == 10 * GB


async def test_deploy_converts_into_its_own_folder(paths, hw, monkeypatch):
    monkeypatch.setattr(workers_engine, "repo_files", AsyncMock(return_value=FILES))
    plan = {"files": ["transformer/a.safetensors"], "size": 7 * GB, "largest": 5 * GB, "saved": 10 * GB}
    monkeypatch.setattr(hf, "bf16_plan", AsyncMock(return_value=plan))
    download = AsyncMock(return_value="/converted")
    monkeypatch.setattr(workers_engine, "hf_download", download)
    ctx = EngineContext(paths, "t", "t", [], hw(), lambda _: None, lambda *_: None, "t", 1)
    spec = DeploymentSpec(modality="video", model={"source": "hf", "id": "Wan-AI/x", "engine": "diffusers_video"})
    assert (await download_model(spec, ctx, "diffusers"))[0] == "/converted"
    kwargs = download.await_args.kwargs
    assert kwargs["convert"] == plan["files"] and kwargs["out_dir"] == paths.models / "bf16" / "Wan-AI--x"

    spec.options["weights"] = "original"
    await download_model(spec, ctx, "diffusers")
    assert "convert" not in download.await_args.kwargs


def test_to_bf16_rewrites_only_float32(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    from kaggleui.engines.download import to_bf16

    src = tmp_path / "a.safetensors"
    save_file({"w": torch.ones(4, dtype=torch.float32), "i": torch.arange(3)}, str(src), metadata={"format": "pt"})
    to_bf16(src, tmp_path / "out" / "a.safetensors")
    out = load_file(str(tmp_path / "out" / "a.safetensors"))
    assert out["w"].dtype == torch.bfloat16 and out["i"].dtype == torch.int64


async def test_assess_choice_follows_settings(hw, monkeypatch):
    """The drawer re-checks fit as settings change: memory mode, 4-bit and download precision."""
    service = CatalogService(hw)
    monkeypatch.setattr(hf, "model_info", AsyncMock(return_value={
        "id": "Wan-AI/x", "pipeline_tag": "text-to-video", "library_name": "diffusers", "tags": ["diffusers"],
        "siblings": FILES}))
    comps = {"transformer": 3 * GB, "text_encoder": 10 * GB, "vae": GB}
    monkeypatch.setattr(hf, "bf16_plan", AsyncMock(return_value={"files": [], "size": sum(comps.values()),
                                                                  "largest": comps["text_encoder"], "saved": 10 * GB,
                                                                  "components": comps}))
    try:
        auto = await service.assess_choice("hf:Wan-AI/x")
        all_gpu = await service.assess_choice("hf:Wan-AI/x", options={"offload": "none"})
        nf4 = await service.assess_choice("hf:Wan-AI/x", options={"offload": "model", "quantize": "nf4"})
        fp32 = await service.assess_choice("hf:Wan-AI/x", options={"offload": "model", "weights": "original"})
    finally:
        await service.close()
    assert auto["fit"]["suggest"] == {"offload": "model"} and auto["fit"]["verdict"] == "good"
    assert all_gpu["fit"]["verdict"] == "worst"
    assert nf4["fit"]["vram_needed"] < auto["fit"]["vram_needed"]
    assert fp32["size"] == auto["size"] + 10 * GB and fp32["fit"]["vram_needed"] == auto["fit"]["vram_needed"]
