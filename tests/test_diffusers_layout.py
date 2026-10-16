"""Original-format releases must never be launched as diffusers pipelines (Wan VACE, LTX-2.3)."""

import json
from unittest.mock import AsyncMock

import pytest

from kaggleui.catalog import hf
from kaggleui.catalog.service import CatalogService
from kaggleui.engines import workers_engine
from kaggleui.engines.context import DeployError, EngineContext
from kaggleui.engines.workers_engine import download_model, ensure_pipeline_class, pipeline_class
from kaggleui.schemas import DeploymentSpec

# Wan-AI's original release: tagged "diffusers" on the Hub but laid out for the Wan repo's own code.
ORIGINAL_FILES = [{"rfilename": "config.json", "size": 300},
                  {"rfilename": "diffusion_pytorch_model-00001-of-00007.safetensors", "size": 9_000_000_000},
                  {"rfilename": "google/umt5-xxl/tokenizer.json", "size": 16_000_000},
                  {"rfilename": "models_t5_umt5-xxl-enc-bf16.pth", "size": 11_000_000_000}]
PIPELINE_FILES = [{"rfilename": "model_index.json", "size": 500},
                  {"rfilename": "transformer/config.json", "size": 300},
                  {"rfilename": "transformer/diffusion_pytorch_model.safetensors", "size": 2_000_000_000},
                  {"rfilename": "text_encoder/model.safetensors", "size": 1_000_000_000}]


def listing(repo, files, downloads=100):
    return {"id": repo, "pipeline_tag": "image-to-video", "library_name": "diffusers", "tags": ["diffusers"],
            "downloads": downloads, "siblings": [{"rfilename": f["rfilename"]} for f in files]}


def test_pipeline_layout_needs_model_index():
    assert not hf.is_pipeline(ORIGINAL_FILES)
    assert hf.is_pipeline(PIPELINE_FILES)
    assert hf.twin_ids("Wan-AI/Wan2.1-VACE-14B") == ["Wan-AI/Wan2.1-VACE-14B-Diffusers"]
    assert hf.twin_ids("Wan-AI/Wan2.1-VACE-14B-diffusers") == []


@pytest.fixture
async def catalog(hw):
    service = CatalogService(hw)
    try:
        yield service
    finally:
        await service.close()


async def test_listed_twin_replaces_original(catalog, monkeypatch):
    monkeypatch.setattr(hf, "search", AsyncMock(return_value=[
        listing("Wan-AI/Wan2.1-VACE-14B", ORIGINAL_FILES, 900),
        listing("Wan-AI/Wan2.1-VACE-14B-diffusers", PIPELINE_FILES, 50),
    ]))
    monkeypatch.setattr(hf, "model_info", AsyncMock(side_effect=AssertionError("no lookup needed")))
    result = await catalog.search("video", q="vace", show_unsupported=True)
    ids = [it["id"] for it in result["items"]]
    assert "Wan-AI/Wan2.1-VACE-14B" not in ids
    assert "Wan-AI/Wan2.1-VACE-14B-diffusers" in ids


async def test_unlisted_twin_is_found_and_swapped_in(catalog, monkeypatch):
    monkeypatch.setattr(hf, "search", AsyncMock(return_value=[listing("Wan-AI/Wan2.1-VACE-1.3B", ORIGINAL_FILES)]))

    async def info(client, repo, revision=None):
        assert repo == "Wan-AI/Wan2.1-VACE-1.3B-Diffusers"
        return {**listing("Wan-AI/Wan2.1-VACE-1.3B-diffusers", PIPELINE_FILES), "siblings": PIPELINE_FILES}

    monkeypatch.setattr(hf, "model_info", info)
    result = await catalog.search("video", q="vace")
    [item] = result["items"]
    assert item["id"] == "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    assert item["engine"] == "diffusers_video"
    assert item["spec"]["model"]["id"] == "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    assert item["size"] == sum(f["size"] for f in PIPELINE_FILES)


async def test_single_file_release_is_findable_but_not_launchable(catalog, monkeypatch):
    monkeypatch.setattr(hf, "search", AsyncMock(return_value=[
        listing("Lightricks/LTX-2.3", [{"rfilename": "ltx-2.3-22b-dev.safetensors"}], 9000),
        listing("someone/LTX-2.3-dev-Diffusers", PIPELINE_FILES, 10),
    ]))
    async def info(client, repo, revision=None):
        # Lightricks publishes no -Diffusers twin; the community conversion is a real pipeline.
        return {"id": repo, "siblings": PIPELINE_FILES if repo.startswith("someone/") else []}

    monkeypatch.setattr(hf, "model_info", info)
    default = await catalog.search("video", q="ltx 2.3")
    assert [it["id"] for it in default["items"]] == ["someone/LTX-2.3-dev-Diffusers"]
    assert default["counts"]["hidden_unsupported"] == 1
    assert default["hidden_examples"] == ["Lightricks/LTX-2.3"]

    everything = await catalog.search("video", q="ltx 2.3", show_unsupported=True)
    official = next(it for it in everything["items"] if it["id"] == "Lightricks/LTX-2.3")
    assert official["engine"] is None and official["spec"] is None
    assert "single-file" in official["description"].lower()


async def test_detail_of_original_opens_its_diffusers_version(catalog, monkeypatch):
    async def info(client, repo, revision=None):
        if repo == "Wan-AI/Wan2.1-VACE-14B":
            return {**listing(repo, ORIGINAL_FILES), "siblings": ORIGINAL_FILES}
        return {**listing("Wan-AI/Wan2.1-VACE-14B-diffusers", PIPELINE_FILES), "siblings": PIPELINE_FILES}

    monkeypatch.setattr(hf, "model_info", info)
    monkeypatch.setattr(hf, "capability_files", AsyncMock(return_value={"config": {}, "template": None}))
    detail = await catalog.detail("hf:Wan-AI/Wan2.1-VACE-14B")
    assert detail["id"] == "Wan-AI/Wan2.1-VACE-14B-diffusers"
    assert detail["original"] == "Wan-AI/Wan2.1-VACE-14B"
    assert detail["spec"]["model"]["id"] == "Wan-AI/Wan2.1-VACE-14B-diffusers"


def context(paths, hw, logs):
    return EngineContext(paths, "test", "test", [], hw(), logs.append, lambda *_: None, "test", 8999)


async def test_deploying_an_original_release_downloads_its_twin(paths, hw, monkeypatch):
    monkeypatch.setattr(workers_engine, "repo_files", AsyncMock(return_value=ORIGINAL_FILES))
    monkeypatch.setattr(hf, "diffusers_twin", AsyncMock(return_value={
        "id": "Wan-AI/Wan2.1-VACE-14B-diffusers", "siblings": PIPELINE_FILES}))
    download = AsyncMock(return_value="/snapshot")
    monkeypatch.setattr(workers_engine, "hf_download", download)
    spec = DeploymentSpec(modality="video", model={"source": "hf", "id": "Wan-AI/Wan2.1-VACE-14B",
                                                   "engine": "diffusers_video"})
    logs: list = []
    await download_model(spec, context(paths, hw, logs), "diffusers")
    assert download.await_args.args[1] == "Wan-AI/Wan2.1-VACE-14B-diffusers"
    assert "model_index.json" in download.await_args.args[2]
    assert any("diffusers version" in line for line in logs)


async def test_single_file_deploy_fails_before_downloading(paths, hw, monkeypatch):
    monkeypatch.setattr(workers_engine, "repo_files", AsyncMock(return_value=[
        {"rfilename": "ltx-2.3-22b-dev.safetensors", "size": 46_000_000_000}]))
    monkeypatch.setattr(hf, "diffusers_twin", AsyncMock(return_value=None))
    download = AsyncMock()
    monkeypatch.setattr(workers_engine, "hf_download", download)
    spec = DeploymentSpec(modality="video", model={"source": "hf", "id": "Lightricks/LTX-2.3",
                                                   "engine": "diffusers_video"})
    with pytest.raises(DeployError, match="not a diffusers pipeline"):
        await download_model(spec, context(paths, hw, []), "diffusers")
    download.assert_not_awaited()


def test_pipeline_class_is_read_from_model_index(tmp_path):
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
    assert pipeline_class(str(tmp_path)) == "QwenImage21Pipeline"
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "os.system('x')"}))
    assert pipeline_class(str(tmp_path)) is None
    assert pipeline_class(str(tmp_path / "missing")) is None


async def test_missing_pipeline_class_upgrades_diffusers(tmp_path, paths, hw, monkeypatch):
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
    checks = iter([3, 0])  # missing before the upgrade, present after
    probes: list = []

    async def run_cmd(cmd, **_):
        probes.append(cmd)
        return next(checks)

    upgrade = AsyncMock()
    monkeypatch.setattr(workers_engine, "run_cmd", run_cmd)
    monkeypatch.setattr(workers_engine, "pip_upgrade", upgrade)
    await ensure_pipeline_class(context(paths, hw, []), "/py", str(tmp_path))
    assert upgrade.await_args.args[1] == ["diffusers"]
    assert "QwenImage21Pipeline" in probes[0][-1]


async def test_pipeline_class_falls_back_to_github_then_explains(tmp_path, paths, hw, monkeypatch):
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "FuturePipeline"}))
    monkeypatch.setattr(workers_engine, "run_cmd", AsyncMock(return_value=3))
    upgrade = AsyncMock()
    monkeypatch.setattr(workers_engine, "pip_upgrade", upgrade)
    with pytest.raises(DeployError, match="FuturePipeline"):
        await ensure_pipeline_class(context(paths, hw, []), "/py", str(tmp_path))
    assert [c.args[1][0] for c in upgrade.await_args_list] == [
        "diffusers", "diffusers @ git+https://github.com/huggingface/diffusers"]
