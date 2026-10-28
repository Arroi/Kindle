"""Paths, environment detection and preset loading."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .schemas import Preset

ON_KAGGLE = os.path.isdir("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
# marimo's hosted notebooks. Cloudflare quick tunnels don't work there; Gradio share links do.
ON_MOLAB = (os.environ.get("KINDLE_PLATFORM") == "molab"
            or any(key.upper().startswith("MOLAB") for key in os.environ))

DEFAULT_PORT = int(os.environ.get("KINDLE_PORT", "8890"))


def default_home() -> str:
    """Where models, caches and logs go. Kaggle and Colab: /tmp (a real disk). Elsewhere, e.g. molab: a
    local scratch disk that isn't stored in RAM, never the notebook's own (persistent, quota'd) folder.
    Pinned in KINDLE_HOME so the notebook and the server agree."""
    if not os.environ.get("KINDLE_HOME"):
        if ON_KAGGLE or ON_COLAB:
            os.environ["KINDLE_HOME"] = "/tmp/kindle-data"
        else:
            from .system.memory import roomiest_home

            os.environ["KINDLE_HOME"] = str(roomiest_home())
    return os.environ["KINDLE_HOME"]


MOLAB_DISK_BUDGET_GB = 200


def disk_budget() -> int | None:
    """Most bytes Kindle may write to its data folder, or None for no limit beyond the disk itself.

    molab sandboxes can be stopped without an error when they write more than their storage allows (a
    quota free-space checks can't see), so Kindle keeps a budget there: 200 GB by default.
    KINDLE_DISK_BUDGET_GB overrides it; 0 turns the budget off.
    """
    value = os.environ.get("KINDLE_DISK_BUDGET_GB")
    if value is not None:
        try:
            gb = float(value)
        except ValueError:
            return None
        return int(gb * 1000**3) if gb > 0 else None
    molab = ON_MOLAB or os.environ.get("KINDLE_PLATFORM") == "molab"  # set by kindle.molab.start() after import
    return MOLAB_DISK_BUDGET_GB * 1000**3 if molab else None


class Paths:
    """Everything big lives on the ephemeral disk, never in /kaggle/working (20 GB, saved as output)."""

    def __init__(self, home: Path | None = None):
        self.home = Path(home or default_home())
        self.models = self.home / "models"
        self.hf_home = self.models / "hf"
        self.ollama_models = self.models / "ollama"
        self.envs = self.home / "envs"
        self.bin = self.home / "bin"
        self.logs = self.home / "logs"
        self.outputs = self.home / "outputs"
        self.cache = self.home / "cache"
        self.state_file = self.home / "state.json"

    def ensure(self) -> Paths:
        for p in (self.home, self.models, self.hf_home, self.ollama_models, self.envs, self.bin,
                  self.logs, self.outputs, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def apply_env(self) -> None:
        """Point every library's cache at our models dir so workers share downloads."""
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("OLLAMA_MODELS", str(self.ollama_models))
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_preset(value: str | dict | Preset | None) -> Preset | None:
    """Accept a Preset, dict, JSON string, or path to a JSON file."""
    if value is None or value == "":
        return None
    if isinstance(value, Preset):
        return value
    if isinstance(value, dict):
        return Preset.model_validate(value)
    text = str(value).strip()
    if not text.startswith("{"):
        text = Path(text).expanduser().read_text()
    return Preset.model_validate(json.loads(text))


def burrow_configured() -> bool:
    return bool(os.environ.get("BURROW_TOKEN") and os.environ.get("BURROW_HOSTNAME"))


def default_tunnel() -> str:
    env = os.environ.get("KINDLE_TUNNEL")
    if env:
        return env
    if ON_MOLAB:
        return "burrow" if burrow_configured() else "gradio"
    return "cloudflare" if (ON_KAGGLE or ON_COLAB) else "none"
