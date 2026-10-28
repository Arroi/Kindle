from __future__ import annotations

import pytest

from kindle.config import Paths
from kindle.system.probe import Probe


@pytest.fixture
def paths(tmp_path):
    return Paths(tmp_path / "kui").ensure()


@pytest.fixture
def hw(paths):
    def make(profile: str = "t4x2"):
        return Probe(paths, fake=profile).snapshot()

    return make


@pytest.fixture(autouse=True)
def _no_weight_header_reads(monkeypatch):
    """Precision planning reads safetensors headers from the Hub; tests opt in with their own stub."""
    from unittest.mock import AsyncMock

    from kindle.catalog import hf

    monkeypatch.setattr(hf, "bf16_plan", AsyncMock(return_value=None))
