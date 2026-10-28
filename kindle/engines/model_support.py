"""Read checkpoint metadata without importing or executing model code."""

from __future__ import annotations

import json
from pathlib import Path


def model_config(path: str) -> dict:
    file = Path(path) / "config.json"
    try:
        if file.stat().st_size <= 2 * 1024**2:
            value = json.loads(file.read_text())
            return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        pass
    return {}


def audio_adapter(config: dict) -> str | None:
    kind = config.get("model_type")
    if kind == "midashenglm-gen":
        return "midasheng"
    if kind == "vibevoice":
        # Original Microsoft weights and converted Transformers weights use different layouts.
        return "vibevoice_hf" if "text_config" in config and "audio_config" in config else "vibevoice"
    return None


def audio_environment(adapter: str) -> tuple[str, list[str]]:
    if adapter == "midasheng":
        return "midasheng", ["torch", "torchaudio", "transformers>=4.51,<5", "accelerate", "einops",
                              "safetensors", "soundfile", "tqdm", "numpy", "x-transformers", "sentencepiece"]
    if adapter == "vibevoice_hf":
        return "vibevoice-hf", ["torch", "torchaudio", "transformers>=5.17", "accelerate", "diffusers",
                                 "soundfile", "librosa"]
    # The original TTS implementation was removed from Microsoft's main branch. Pin the
    # official revision that contains its inference classes and compatible dependency versions.
    return "vibevoice", ["https://github.com/microsoft/VibeVoice/archive/6065c5224e1fee5b2eaeef15f79ee8ef7de947d0.zip",
                         "soundfile", "torchaudio"]
