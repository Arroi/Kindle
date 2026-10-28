"""Studio conversations are saved server-side behind the Studio password."""

import pytest
from fastapi.testclient import TestClient

from kindle.schemas import Settings
from kindle.server.app import create_app
from kindle.server.state import AppState


@pytest.fixture
def client(paths):
    s = Settings()
    s.exposure.provider = "none"
    state = AppState(paths=paths, settings=s, port=8999, mock=True, fake_hw="t4x2")
    with TestClient(create_app(state)) as c:
        c.state_ = state
        yield c


def test_chats_need_studio_access(client):
    assert client.get("/_kui/api/studio/chats").status_code == 401
    assert client.put("/_kui/api/studio/chats/abc", json={"messages": []}).status_code == 401


def test_save_list_open_delete(client):
    client.post("/_kui/api/access-login", json={"password": client.state_.auth.password})
    chat = {"title": "Plan", "model": "qwen3:8b", "messages": [{"id": "1", "role": "user", "content": "hi"}]}
    assert client.put("/_kui/api/studio/chats/c1", json=chat).status_code == 200
    listed = client.get("/_kui/api/studio/chats").json()
    assert listed[0]["id"] == "c1" and listed[0]["title"] == "Plan" and listed[0]["messages"] == 1
    assert client.get("/_kui/api/studio/chats/c1").json()["messages"][0]["content"] == "hi"
    assert client.delete("/_kui/api/studio/chats/c1").status_code == 200
    assert client.get("/_kui/api/studio/chats/c1").status_code == 404


def test_chat_ids_cannot_escape_the_folder(client):
    client.post("/_kui/api/access-login", json={"password": client.state_.auth.password})
    assert client.put("/_kui/api/studio/chats/..%2Fsession", json={"messages": []}).status_code in (400, 404)
    assert client.put("/_kui/api/studio/chats/a.b", json={"messages": []}).status_code == 400
    assert client.put("/_kui/api/studio/chats/ok", json={"messages": "nope"}).status_code == 400
