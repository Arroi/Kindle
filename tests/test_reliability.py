"""Reliability: deployments survive a server restart, crashed runners restart, dead copies fail over."""

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from kindle.deploy import manager as manager_mod
from kindle.deploy.manager import Deployment
from kindle.engines.base import Backend, Unreachable
from kindle.schemas import DeploymentSpec, ModelRef, Settings
from kindle.server.app import create_app
from kindle.server.state import AppState


def make_state(paths) -> AppState:
    s = Settings()
    s.exposure.provider = "none"
    return AppState(paths=paths, settings=s, port=8999, mock=True, fake_hw="t4x2")


ADMIN = lambda st: {"Authorization": f"Bearer {st.auth.admin_token}"}  # noqa: E731
SPEC = {"modality": "text", "name": "brain", "model": {"source": "ollama", "id": "qwen3", "variant": "0.6b", "engine": "ollama"}}


def wait_ready(client, st, name="brain"):
    for _ in range(100):
        deps = [d for d in client.get("/_kui/api/deployments", headers=ADMIN(st)).json() if d["name"] == name]
        if deps and deps[0]["status"] == "ready":
            return deps[0]
        time.sleep(0.1)
    raise AssertionError(deps)


class Dead(Backend):
    """A copy whose runner vanished: every call fails before any response."""

    def __init__(self):
        self.calls = 0

    async def chat(self, body):
        self.calls += 1
        raise Unreachable("The linked session", httpx.ConnectError("gone"))

    async def chat_stream(self, body):
        self.calls += 1
        raise Unreachable("The linked session", httpx.ConnectError("gone"))
        yield  # pragma: no cover

    async def healthy(self):
        return False


def add_copy(st, name: str, backend: Backend) -> Deployment:
    spec = DeploymentSpec(modality="text", name=name, model=ModelRef(source="hf", id=name, engine="linked"))
    dep = Deployment(id=f"copy-{name}-{id(backend)}", spec=spec, served_name=name, status="ready", backend=backend)
    st.manager.deployments[dep.id] = dep
    return dep


def test_deployments_restore_after_restart(paths):
    st = make_state(paths)
    with TestClient(create_app(st)) as c:
        c.post("/_kui/api/deployments", json=SPEC, headers=ADMIN(st))
        wait_ready(c, st)
    saved = json.loads((paths.home / "deployments.json").read_text())
    assert [s["name"] for s in saved] == ["brain"]  # shutdown kept the list

    st2 = make_state(paths)
    with TestClient(create_app(st2)) as c:
        assert wait_ready(c, st2)["name"] == "brain"
        dep = next(iter(st2.manager.deployments))
        assert c.delete(f"/_kui/api/deployments/{dep}", headers=ADMIN(st2)).status_code == 200
    assert json.loads((paths.home / "deployments.json").read_text()) == []  # removed by the user: forgotten


def test_restore_can_be_turned_off(paths, monkeypatch):
    (paths.home / "deployments.json").write_text(json.dumps([SPEC]))
    monkeypatch.setenv("KINDLE_RESTORE", "0")
    st = make_state(paths)
    with TestClient(create_app(st)):
        time.sleep(0.5)
        assert not st.manager.deployments


def test_watchdog_restarts_a_dead_runner_under_the_same_name(paths, monkeypatch):
    monkeypatch.setattr(manager_mod, "WATCH_SECONDS", 0.05)
    st = make_state(paths)
    with TestClient(create_app(st)) as c:
        c.post("/_kui/api/deployments", json=SPEC, headers=ADMIN(st))
        first = wait_ready(c, st)
        st.manager.deployments[first["id"]].backend.healthy = lambda: asyncio.sleep(0, result=False)
        for _ in range(100):
            again = [d for d in st.manager.list() if d["name"] == "brain" and d["id"] != first["id"]]
            if again and again[0]["status"] == "ready":
                break
            time.sleep(0.05)
        else:
            raise AssertionError(st.manager.list())
        assert len(st.manager._restarts["brain"]) == 1


def test_watchdog_gives_up_after_repeated_crashes(paths):
    st = make_state(paths)
    dep = add_copy(st, "flaky", Dead())

    async def run():
        st.manager._restarts["flaky"] = [time.time()] * manager_mod.MAX_RESTARTS
        await st.manager._crashed(dep, "stopped answering health checks")

    asyncio.run(run())
    assert dep.status == "error" and "left stopped" in dep.error


@pytest.mark.parametrize("stream", [False, True])
def test_requests_fail_over_to_a_live_copy(paths, stream):
    st = make_state(paths)
    key = {"Authorization": f"Bearer {st.auth.api_key}"}
    with TestClient(create_app(st)) as c:
        c.post("/_kui/api/deployments", json=SPEC, headers=ADMIN(st))
        wait_ready(c, st)
        dead = Dead()
        add_copy(st, "brain", dead)
        body = {"model": "brain", "stream": stream, "messages": [{"role": "user", "content": "hi"}]}
        for _ in range(4):  # copies alternate; each request still succeeds
            r = c.post("/v1/chat/completions", json=body, headers=key)
            assert r.status_code == 200, r.text
            assert "error" not in r.text
        assert dead.calls >= 1


def test_failover_never_switches_to_a_different_model(paths):
    st = make_state(paths)
    key = {"Authorization": f"Bearer {st.auth.api_key}"}
    with TestClient(create_app(st)) as c:
        c.post("/_kui/api/deployments", json=SPEC, headers=ADMIN(st))
        wait_ready(c, st)
        add_copy(st, "big", Dead())
        r = c.post("/v1/chat/completions", json={"model": "big", "messages": [{"role": "user", "content": "x"}]},
                   headers=key)
        assert r.status_code == 503 and "unreachable" in r.json()["error"]["message"]
