"""The build stamp that makes launch() restart a server after an update with the same version."""

import json
from types import SimpleNamespace

from kindle import build


def test_build_includes_the_git_commit(monkeypatch):
    direct = {"url": "https://github.com/Arroi/Kindle", "vcs_info": {"vcs": "git", "commit_id": "6fc47ce28e788cd143a9"}}
    dist = SimpleNamespace(version="0.2.16", read_text=lambda name: json.dumps(direct) if name == "direct_url.json" else None)
    monkeypatch.setattr(build.importlib.metadata, "distribution", lambda name: dist)
    assert build.installed_build() == "0.2.16+6fc47ce28e78"


def test_build_without_git_is_the_version(monkeypatch):
    dist = SimpleNamespace(version="0.2.16", read_text=lambda name: None)
    monkeypatch.setattr(build.importlib.metadata, "distribution", lambda name: dist)
    assert build.installed_build() == "0.2.16"
