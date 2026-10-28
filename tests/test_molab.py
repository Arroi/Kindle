"""The molab start cell: role check and the card each role prints."""

import pytest

from kindle import molab


def test_role_must_be_head_or_helper():
    with pytest.raises(ValueError, match="head"):
        molab.start("worker")


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
