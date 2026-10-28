"""Linked sessions: two real apps talk over an in-memory transport."""

import httpx
import pytest

from kindle.config import Paths
from kindle.deploy.manager import Deployment
from kindle.engines.base import Backend
from kindle.links import HOP_HEADER, LinkError
from kindle.schemas import DeploymentSpec, ModelRef, Settings
from kindle.server.app import create_app
from kindle.server.state import AppState

REMOTE = "http://remote.test"


class Echo(Backend):
    def __init__(self, where: str):
        self.where = where

    async def chat(self, body: dict) -> dict:
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": self.where}}]}

    async def chat_stream(self, body: dict):
        yield {"choices": [{"index": 0, "delta": {"content": self.where}}]}


def make_state(tmp_path, name: str, hw: str) -> AppState:
    settings = Settings()
    settings.exposure.provider = "none"
    return AppState(paths=Paths(tmp_path / name).ensure(), settings=settings, port=8999, mock=True, fake_hw=hw)


def serve(state: AppState, model: str, where: str) -> None:
    spec = DeploymentSpec(modality="text", model=ModelRef(source="hf", id=model, engine="mock"))
    dep = Deployment(id=where, spec=spec, served_name=model, status="ready", backend=Echo(where))
    state.manager.deployments[dep.id] = dep


@pytest.fixture
def pair(tmp_path):
    hub, remote = make_state(tmp_path, "hub", "t4x2"), make_state(tmp_path, "remote", "t4x2")
    hub.links._transport = httpx.ASGITransport(app=create_app(remote))
    yield hub, remote


async def test_linked_model_is_served_and_pooled(pair):
    hub, remote = pair
    serve(remote, "qwen3", "remote")
    await hub.links.add(REMOTE + "/?token=abc", remote.auth.api_key)
    [link] = hub.links.list()["links"]
    assert link["url"] == REMOTE and link["status"] == "up" and link["gpu_label"] == "2× Tesla T4"

    dep = hub.manager.resolve("qwen3", "text")
    reply = await dep.backend.chat({"messages": [{"role": "user", "content": "hi"}]})
    assert reply["choices"][0]["message"]["content"] == "remote"
    chunks = [c async for c in dep.backend.chat_stream({"messages": []})]
    assert chunks[0]["choices"][0]["delta"]["content"] == "remote"

    # The same model running here too: requests alternate between both copies.
    serve(hub, "qwen3", "hub")
    served = {(await hub.manager.resolve("qwen3", "text").backend.chat({}))["choices"][0]["message"]["content"]
              for _ in range(4)}
    assert served == {"hub", "remote"}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(hub)), base_url="http://hub.test") as c:
        models = (await c.get("/v1/models", headers={"Authorization": f"Bearer {hub.auth.api_key}"})).json()["data"]
    assert [m["id"] for m in models] == ["qwen3"]
    await hub.links.close()


async def test_different_gpus_are_refused(tmp_path):
    hub, remote = make_state(tmp_path, "hub", "t4x2"), make_state(tmp_path, "remote", "p100")
    hub.links._transport = httpx.ASGITransport(app=create_app(remote))
    with pytest.raises(LinkError) as err:
        await hub.links.add(REMOTE, remote.auth.api_key)
    assert err.value.status == 409
    assert "2× Tesla T4" in err.value.message and "1× Tesla P100-PCIE-16GB" in err.value.message
    assert hub.links.list()["links"] == []


async def test_wrong_api_key_is_reported(pair):
    hub, _ = pair
    with pytest.raises(LinkError) as err:
        await hub.links.add(REMOTE, "kui-wrong")
    assert err.value.status == 401


async def test_forwarded_requests_are_not_forwarded_again(pair):
    hub, remote = pair
    serve(remote, "qwen3", "remote")
    await hub.links.add(REMOTE, remote.auth.api_key)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(hub)), base_url="http://hub.test") as c:
        auth = {"Authorization": f"Bearer {hub.auth.api_key}"}
        body = {"model": "qwen3", "messages": [{"role": "user", "content": "hi"}]}
        assert (await c.post("/v1/chat/completions", json=body, headers=auth)).status_code == 200
        hopped = await c.post("/v1/chat/completions", json=body, headers={**auth, HOP_HEADER: "1"})
    assert hopped.status_code == 404
    await hub.links.close()


async def test_links_survive_restart_and_report_gpu_changes(pair, tmp_path):
    hub, remote = pair
    await hub.links.add(REMOTE, remote.auth.api_key)
    await hub.links.close()
    again = AppState(paths=hub.paths, settings=hub.settings, port=8999, mock=True, fake_hw="t4x2")
    swapped = make_state(tmp_path, "swapped", "p100")
    swapped.settings.api.require_key = False
    again.links._transport = httpx.ASGITransport(app=create_app(swapped))
    [link] = again.links.links.values()
    await again.links.refresh(link)
    assert link.status == "mismatch" and "GPUs don't match" in link.error
    assert again.links.deployments() == []
    await again.links.close()
