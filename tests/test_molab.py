"""The molab start cell: role check and the card each role prints."""

import pytest

from kindle import molab


def test_role_must_be_head_or_helper():
    with pytest.raises(ValueError, match="head"):
        molab.start("worker")


def _start_with(monkeypatch, tmp_path, fails: set[str], *, crash: bool = False,
                tried: list[str] | None = None) -> list[str]:
    tried = [] if tried is None else tried
    after_failure = {}

    def launch(*, tunnel, **kwargs):
        tried.append(tunnel)
        if crash:
            raise RuntimeError("Kindle exited during startup; see the log above.")
        if tunnel in fails:
            after_failure.update(status="tunnel_error", tunnel={"error": "no route to the domain"})
            raise RuntimeError("The public tunnel didn't connect.")
        return {"port": 8890, "url": f"https://{tunnel}.example", "admin_token": "adm", "api_key": "k"}

    def released(_):
        raise KeyboardInterrupt  # end the status loop right away

    monkeypatch.setenv("KINDLE_PLATFORM", "molab")  # start() sets it; restored after the test
    monkeypatch.setattr(molab, "launch", launch)
    monkeypatch.setattr(molab, "stop", lambda home: True)
    monkeypatch.setattr(molab, "_read_state", lambda paths: dict(after_failure) or None)
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
    assert "Burrow tunnel didn't connect (no route to the domain)" in out
    assert "https://gradio.example/?token=adm" in out


def test_a_server_that_crashes_is_not_retried_on_other_tunnels(monkeypatch, tmp_path):
    monkeypatch.setenv("BURROW_TOKEN", "eyJ-key")
    monkeypatch.setenv("BURROW_HOSTNAME", "ai.example.com")
    tried: list[str] = []
    with pytest.raises(RuntimeError, match="exited during startup"):
        _start_with(monkeypatch, tmp_path, set(), crash=True, tried=tried)
    assert tried == ["burrow"]


def test_missing_dependencies_are_installed_before_the_server_starts(monkeypatch):
    import importlib.metadata

    from kindle import launcher

    installed = {"fastapi": False, "httpx": True, "nvidia-ml-py": False}

    def version(name):
        if not installed.get(name):
            raise importlib.metadata.PackageNotFoundError(name)
        return "1"

    monkeypatch.setattr(importlib.metadata, "requires", lambda name: [
        "fastapi>=0.110", "httpx>=0.27", "nvidia-ml-py>=12; platform_system == 'Nothing'", "pytest; extra == 'dev'"])
    monkeypatch.setattr(importlib.metadata, "version", version)
    assert launcher.missing_dependencies() == ["fastapi>=0.110"]


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
