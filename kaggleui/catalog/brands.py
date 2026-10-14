"""Shared company/family identities used by catalog filters and the frontend logos."""

from __future__ import annotations

import json
import re
from pathlib import Path

BRANDS = json.loads(Path(__file__).with_name("brands.json").read_text())
_PATTERNS = [(entry, re.compile(entry["pattern"], re.I)) for entry in BRANDS]


def describe(ident: str, name: str = "", source: str = "hf") -> dict:
    owner = ident.split("/", 1)[0].lower() if source == "hf" and "/" in ident else ""
    match = next((entry for entry in BRANDS if owner in entry["owners"]), None)
    if match is None:
        match = next((entry for entry, pattern in _PATTERNS if pattern.search(f"{ident} {name}")), None)
    if match:
        return {key: match[key] for key in ("id", "label", "logo")}
    if owner:
        return {"id": f"publisher:{owner}", "label": ident.split("/", 1)[0], "logo": None}
    return {"id": "other", "label": "Other / unknown", "logo": None}


def search_terms(brand: str) -> list[str]:
    if brand.startswith("publisher:"):
        return [brand.removeprefix("publisher:")]
    return next((entry["search"] for entry in BRANDS if entry["id"] == brand), [])


def hf_search_terms(brand: str) -> list[str]:
    terms = search_terms(brand)
    entry = next((entry for entry in BRANDS if entry["id"] == brand), None)
    # Include a publisher namespace as well as family names: e.g. Facebook's MusicGen
    # and MMS checkpoints belong to Meta even though their names do not contain Llama.
    return list(dict.fromkeys([*terms, *(entry["owners"][-1:] if entry else [])]))


def facets(items: list[dict]) -> list[dict]:
    choices = {entry["id"]: {key: entry[key] for key in ("id", "label", "logo")} | {"count": 0}
               for entry in BRANDS}
    for item in items:
        brand = item["brand"]
        choices.setdefault(brand["id"], {**brand, "count": 0})["count"] += 1
    return sorted(choices.values(), key=lambda entry: entry["label"].casefold())
