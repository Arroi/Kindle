"""The molab start cell: role check and the card each role prints."""

import pytest

from kindle import molab


def test_role_must_be_head_or_helper():
    with pytest.raises(ValueError, match="head"):
        molab.start("worker")


def _start_with(monkeypatch, tmp_path, fails: set[str]) -> list[str]:
    tried = []

    def launch(*, tunnel, **kwargs):
        tried.append(tunnel)
        if tunnel in fails:
            raise RuntimeError("The public tunnel didn't connect.")
        return {"port": 8890, "url": f"https://{tunnel}.example", "admin_token": "adm", "api_key": "k"}

    def released(_):
        raise KeyboardInterrupt  # end the status loop right away

    monkeypatch.setenv("KINDLE_PLATFORM", "molab")  # start() sets it; restored after the test
    monkeypatch.setattr(molab, "launch", launch)
    monkeypatch.setattr(molab, "stop", lambda home: True)
    monkeypatch.setattr(molab, "_read_state", lambda paths: None)
    monkeypatch.setattr(molab, "old_data_folders", lambda: [])
    monkeypatch.setattr(molab, "_machine", lambda state, home: [])
    monkeypatch.setattr(molab.time, "sleep", released)
    molab.start("head", home=str(tmp_path))
    return tried


def test_burrow_is_the_heads_primary_link_with_gradio_as_fallback(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BURROW_TOKEN", "eyJ-key")
    monkeypatch.setenv("BURROW_HOSTNAME", "ai.example.com")
    assert _start_with(monkeypatch, tmp_path, set()) == ["burrow"]
    assert "https://burrow.example/?token=adm" in capsys.readouterr().out
    assert _start_with(monkeypatch, tmp_path, {"burrow"}) == ["burrow", "gradio"]
    out = capsys.readouterr().out
    assert "Burrow tunnel didn't connect" in out and "https://gradio.example/?token=adm" in out


def test_without_burrow_molab_uses_gradio(monkeypatch, tmp_path):
    monkeypatch.delenv("BURROW_TOKEN", raising=False)
    assert _start_with(monkeypatch, tmp_path, set()) == ["gradio"]
    with pytest.raises(ValueError, match="BURROW_TOKEN"):
        molab.start("head", tunnel="burrow")


def test_cards_tell_each_role_what_to_do(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(molab, "_machine", lambda state, home: [" GPU:      RTX PRO 6000 · 96 GB VRAM"])
    state = {"admin_token": "adm", "api_key": "kui-key", "port": 1}
    molab._card("helper", "https://h.gradio.live", state, str(tmp_path))
    helper = capsys.readouterr().out
    assert "HELPER" in helper and "https://h.gradio.live" in helper and "kui-key" in helper
    assert "Linked sessions" in helper and "adm" not in helper  # a helper's card never shows its admin token
    molab._card("head", "https://x.gradio.live", state, str(tmp_path))
    head = capsys.readouterr().out
    assert "https://x.gradio.live/?token=adm" in head and "https://x.gradio.live/v1" in head and "96 GB" in head
