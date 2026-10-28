"""Paths, environment detection and preset loading."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .schemas import Preset

ON_KAGGLE = os.path.isdir("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
# marimo's hosted notebooks. Cloudflare quick tunnels don't work there; Gradio share links do.
ON_MOLAB = (os.environ.get("KAGGLEUI_PLATFORM") == "molab"
            or any(key.upper().startswith("MOLAB") for key in os.environ))

DEFAULT_PORT = int(os.environ.get("KAGGLEUI_PORT", "8890"))


class Paths:
    """Everything big lives on the ephemeral disk, never in /kaggle/working (20 GB, saved as output)."""

    def __init__(self, home: Path | None = None):
        self.home = Path(home or os.environ.get("KAGGLEUI_HOME", "/tmp/kaggleui"))
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


def default_tunnel() -> str:
    env = os.environ.get("KAGGLEUI_TUNNEL")
    if env:
        return env
    if ON_MOLAB:
        return "gradio"
    return "cloudflare" if (ON_KAGGLE or ON_COLAB) else "none"
