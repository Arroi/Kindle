"""World discovery across Hub tags and model families, with explicit runtime support."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ENTRIES: list[dict] = yaml.safe_load(Path(__file__).with_name("world.yaml").read_text()) or []
INDEX = {entry["id"].lower(): entry for entry in ENTRIES}
RECIPES = {entry["id"].lower(): entry for entry in
           (yaml.safe_load(Path(__file__).with_name("curated.yaml").read_text()) or {}).get("world", [])}
FAMILIES = ("matrix-game", "lingbot-world", "hunyuanworld", "hy-world", "cosmos-predict",
            "cosmos-transfer", "cosmos3", "cosmos-1.0-diffusion", "oasis-")
TAGS = {"world-model", "world-models", "world_model", "world-generation", "world-simulation"}
# A text model or an unrelated image fine-tune can share a family name or Hub tag.
EXCLUDED_TASKS = {"text-generation", "text2text-generation", "image-text-to-text", "text-to-image",
                  "image-to-image", "sentence-similarity", "feature-extraction", "image-classification"}


def profile(repo: str) -> dict | None:
    return INDEX.get(repo.lower()) or RECIPES.get(repo.lower())


def matches(raw: dict) -> bool:
    repo = str(raw.get("id") or "").lower()
    if repo in INDEX:
        return True
    tags = {str(tag).lower() for tag in raw.get("tags") or []}
    if raw.get("pipeline_tag") in EXCLUDED_TASKS or tags & EXCLUDED_TASKS:
        return False
    name = repo.partition("/")[2]
    return (bool(tags & TAGS) or any(family in name for family in FAMILIES)
            or re.search(r"yume-(?:\d+(?:\.\d+)?b|1\.5)", name) is not None)


def queries(q: str, want: int) -> list[dict]:
    limit = min(max(want, 40), 300)
    if q:
        return [{"query": q, "limit": limit},
                *({"query": q, "tag_filter": tag, "limit": limit} for tag in ("world-model", "world-models"))]
    return [*({"tag_filter": tag, "limit": limit} for tag in ("world-model", "world-models", "world-generation")),
            *({"query": family, "limit": min(max(want // 2, 20), 150)} for family in
              ("Matrix-Game", "lingbot-world", "HunyuanWorld", "Yume-", "Cosmos-Predict", "Cosmos-Transfer", "Cosmos3", "oasis-"))]
