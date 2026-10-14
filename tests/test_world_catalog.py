"""World discovery must be broad without turning arbitrary checkpoints into launchable models."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from kaggleui.catalog import hf, world
from kaggleui.catalog.service import CatalogService
from kaggleui.engines.context import DeployError, EngineContext
from kaggleui.engines.download import _worker
from kaggleui.engines.workers_engine import deploy, download_model
from kaggleui.schemas import DeploymentSpec

DISCOVERED = {"id": "community/new-interactive-world", "pipeline_tag": "image-to-video",
              "library_name": "custom-world-runtime", "tags": ["world-model"],
              "safetensors": {"total": 3_000_000_000}, "downloads": 120}


@pytest.fixture
async def catalog(hw, monkeypatch):
    service = CatalogService(hw)
    search = AsyncMock(return_value=[DISCOVERED, {**DISCOVERED, "id": "community/Yume-Qwen-4B",
                                                "pipeline_tag": "text-generation"}])

    async def info(client, repo, revision=None):
        return {"id": repo, "pipeline_tag": "image-to-video", "tags": ["world-model"],
                "library_name": "cosmos", "siblings": [
                    {"rfilename": "model_index.json", "size": 300},
                    {"rfilename": "transformer/config.json", "size": 200},
                    {"rfilename": "transformer/model.safetensors", "size": 4_000_000_000},
                ]}

    monkeypatch.setattr(hf, "search", search)
    monkeypatch.setattr(hf, "model_info", info)
    try:
        yield service, search
    finally:
        await service.close()


async def test_world_default_discovers_more_than_recipes(catalog):
    service, search = catalog
    result = await service.search("world")
    items = {item["id"]: item for item in result["items"]}
    assert len(items) > 18
    assert DISCOVERED["id"] in items
    assert "community/Yume-Qwen-4B" not in items
    assert items[DISCOVERED["id"]]["engine"] is None
    assert items[DISCOVERED["id"]]["spec"] is None
    assert result["counts"]["launchable"] == 14
    assert items["Skywork/Matrix-Game-3.0"]["engine"] == "world_recipe"
    assert items["Etched/oasis-500m"]["spec"]["options"]["recipe"] == "oasis"
    assert search.await_count > 1


async def test_world_launchable_and_fit_filters(catalog):
    service, _ = catalog
    supported = await service.search("world", show_unsupported=False)
    assert len(supported["items"]) == 14
    assert all(item["engine"] and item["spec"] for item in supported["items"])
    assert supported["has_more"] is False
    fitting = await service.search("world", fit="fits")
    assert fitting["items"]
    assert all(item["engine"] and item["fit"]["verdict"] != "worst" for item in fitting["items"])
    assert "nvidia/Cosmos-Predict2.5-2B" not in {item["id"] for item in fitting["items"]}
    picks = await service.search("world", source="curated")
    assert len(picks["items"]) == 2


async def test_world_offline_fallback_keeps_published_models(catalog):
    service, search = catalog
    search.side_effect = httpx.ConnectError("offline")
    result = await service.search("world", source="hf", sort="params_asc")
    assert len(result["items"]) == 20
    assert result["sources"]["hf"].startswith("error")
    known = [item["params"] for item in result["items"] if item["params"]]
    assert known == sorted(known)
    assert result["items"][0]["id"] == "Etched/oasis-500m"


async def test_world_search_matches_requested_family(catalog):
    service, search = catalog
    search.return_value = []
    result = await service.search("world", q="HunyuanWorld")
    assert len(result["items"]) == 3
    assert all("hunyuanworld" in item["id"].lower() for item in result["items"])
    assert all(call.kwargs["query"] == "HunyuanWorld" for call in search.await_args_list)


async def test_world_detail_preserves_modality_and_runtime(catalog):
    service, _ = catalog
    unsupported = await service.detail("hf:" + DISCOVERED["id"])
    assert unsupported["modality"] == "world"
    assert unsupported["engine"] is None and unsupported["spec"] is None
    cosmos = await service.detail("hf:nvidia/Cosmos-Predict2.5-2B")
    assert cosmos["engine"] == "diffusers_world"
    assert cosmos["spec"]["options"]["revision"] == "diffusers/base/post-trained"
    assert cosmos["spec"]["options"]["dtype"] == "bfloat16"
    assert cosmos["fit"]["verdict"] == "worst"  # T4 is not bf16 hardware.


@pytest.mark.parametrize("repo,task,tags,expected", [
    ("new/world-video", "image-to-video", ["world-model"], True),
    ("new/Yume-5B-720P", "image-to-video", [], True),
    ("new/Yume-Qwen-4B", None, ["text-generation"], False),
    ("new/Matrix-Game-embeddings", "sentence-similarity", [], False),
    ("new/random-image-model", "text-to-image", ["world-model"], False),
])
def test_world_discovery_rejects_unrelated_models(repo, task, tags, expected):
    assert world.matches({"id": repo, "pipeline_tag": task, "tags": tags}) is expected


async def test_hf_revision_is_encoded_in_metadata_request():
    def respond(request):
        assert b"diffusers%2Fbase%2Fpost-trained" in request.url.raw_path
        return httpx.Response(200, json={"id": "test/world", "siblings": []})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        info = await hf.model_info(client, "test/world", revision="diffusers/base/post-trained")
        assert info["id"] == "test/world"


def test_download_worker_uses_requested_revision(tmp_path, monkeypatch):
    import huggingface_hub
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    result = tmp_path / "result.json"
    _worker({"repo_id": "test/world", "revision": "diffusers/base/post-trained"}, result)
    assert calls[0]["revision"] == "diffusers/base/post-trained"
    assert json.loads(result.read_text())["path"] == "/snapshot"


async def test_world_download_passes_revision_to_listing_and_worker(paths, hw, monkeypatch):
    from kaggleui.engines import workers_engine
    files = AsyncMock(return_value=[{"rfilename": "model_index.json", "size": 1},
                                  {"rfilename": "transformer/model.safetensors", "size": 10}])
    download = AsyncMock(return_value="/snapshot")
    monkeypatch.setattr(workers_engine, "repo_files", files)
    monkeypatch.setattr(workers_engine, "hf_download", download)
    spec = DeploymentSpec(modality="world", model={"source": "hf", "id": "test/world", "engine": "diffusers_world"},
                          options={"revision": "diffusers/base/post-trained"})
    ctx = EngineContext(paths, "test", "test", [], hw(), lambda _: None, lambda *_: None, "test", 8999)
    assert await download_model(spec, ctx, "diffusers") == ("/snapshot", None)
    assert files.await_args.kwargs["revision"] == "diffusers/base/post-trained"
    assert download.await_args.kwargs["revision"] == "diffusers/base/post-trained"
    with pytest.raises(DeployError, match="Ampere"):
        await deploy(spec, ctx)
