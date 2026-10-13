from __future__ import annotations

import pytest

from kaggleui.config import Paths
from kaggleui.system.probe import Probe


@pytest.fixture
def paths(tmp_path):
    return Paths(tmp_path / "kui").ensure()


@pytest.fixture
def hw(paths):
    def make(profile: str = "t4x2"):
        return Probe(paths, fake=profile).snapshot()

    return make
