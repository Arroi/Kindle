"""Hugging Face Hub REST client (plain HTTP so it works with any installed huggingface_hub version)."""

from __future__ import annotations

import os
import re
from datetime import datetime

import httpx

API = "https://huggingface.co/api"
EXPAND = ["downloads", "likes", "trendingScore", "lastModified", "gated", "library_name",
          "pipeline_tag", "tags", "safetensors", "gguf"]
SORTS = {"trending": "trendingScore", "downloads": "downloads", "likes": "likes", "newest": "lastModified",
         "recommended": "trendingScore"}

GGUF_QUANT_RE = re.compile(
    r"(?:^|[-_.])((?:I?Q\d(?:_[0-9A-Z]+)*)|MXFP4(?:_MOE)?|F16|BF16|F32)(?=[-_.]|$)", re.I)
SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.I)

WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".h5", ".onnx", ".gguf")
AUX_EXTS = (".json", ".txt", ".model", ".tiktoken", ".py", ".jinja", ".vocab")


def _headers() -> dict[str, str]:
    h = {"User-Agent": "kaggleui/0.1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


async def search(client: httpx.AsyncClient, *, pipeline_tag: str | None = None, query: str = "",
                 sort: str = "trending", limit: int = 30, library: str | None = None,
                 tag_filter: str | None = None) -> list[dict]:
    params: list[tuple[str, str]] = [("limit", str(limit)), ("sort", SORTS.get(sort, "trendingScore"))]
    if pipeline_tag:
        params.append(("pipeline_tag", pipeline_tag))
    if query:
        params.append(("search", query))
    if library:
        params.append(("library", library))
    if tag_filter:
        params.append(("filter", tag_filter))
    params += [("expand[]", e) for e in EXPAND]
    r = await client.get(f"{API}/models", params=params, headers=_headers(), timeout=20)
    r.raise_for_status()
    items = r.json()
    for it in items:
        gg = it.get("gguf")
        if isinstance(gg, dict):
            gg.pop("chat_template", None)  # can be huge
    return items


async def model_info(client: httpx.AsyncClient, repo_id: str) -> dict:
    """Full model metadata including per-file sizes (``siblings[].size``)."""
    r = await client.get(f"{API}/models/{repo_id}", params={"blobs": "true"}, headers=_headers(), timeout=20)
    r.raise_for_status()
    info = r.json()
    gg = info.get("gguf")
    if isinstance(gg, dict):
        gg.pop("chat_template", None)
    return info


def parse_time(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def is_gated(value) -> bool:
    return value not in (False, None, "false")


# -- file selection (shared by fit estimates and the downloader) ------------------------------

def gguf_variants(siblings: list[dict]) -> list[dict]:
    """Group *.gguf files by quant. Returns [{quant, files, size}] (mmproj projectors excluded)."""
    groups: dict[str, dict] = {}
    for s in siblings:
        name = s.get("rfilename", "")
        if not name.lower().endswith(".gguf") or "mmproj" in name.lower():
            continue
        base = SHARD_RE.sub(".gguf", name.rsplit("/", 1)[-1])
        m = GGUF_QUANT_RE.search(base[:-5])
        quant = m.group(1).upper() if m else base[:-5]
        g = groups.setdefault(quant, {"quant": quant, "files": [], "size": 0})
        g["files"].append(name)
        g["size"] += int(s.get("size") or 0)
    return sorted(groups.values(), key=lambda g: g["size"])


def diffusers_files(siblings: list[dict]) -> dict:
    """Pick the files a diffusers pipeline needs: component subfolders, preferring fp16 variants.

    Root-level single-file checkpoints (often several GB) are skipped.
    """
    by_comp: dict[str, list[dict]] = {}
    root_files = []
    for s in siblings:
        name = s.get("rfilename", "")
        if "/" in name:
            by_comp.setdefault(name.split("/", 1)[0], []).append(s)
        else:
            root_files.append(s)
    chosen: list[dict] = [s for s in root_files if s["rfilename"] == "model_index.json"]
    comp_sizes: dict[str, int] = {}
    any_fp16 = False
    for comp, files in by_comp.items():
        weights = [f for f in files if f["rfilename"].endswith(WEIGHT_EXTS)]
        aux = [f for f in files if f["rfilename"].endswith(AUX_EXTS)]
        st = [f for f in weights if f["rfilename"].endswith(".safetensors")]
        fp16 = [f for f in st if ".fp16." in f["rfilename"]]
        plain = [f for f in st if not re.search(r"\.(fp16|bf16|fp32|ema|non_ema)\.", f["rfilename"])]
        if fp16:
            any_fp16 = True
            pick = fp16
        elif plain:
            pick = plain
        else:
            pick = [f for f in weights if f["rfilename"].endswith(".bin") and ".fp16." not in f["rfilename"]]
        comp_sizes[comp] = sum(int(f.get("size") or 0) for f in pick)
        chosen += pick + aux
    # If some components only have plain files and others fp16, diffusers' variant loading still
    # falls back to the plain files, so mixing is fine.
    total = sum(int(f.get("size") or 0) for f in chosen)
    return {
        "files": [f["rfilename"] for f in chosen],
        "size": total,
        "largest": max(comp_sizes.values()) if comp_sizes else total,
        "variant": "fp16" if any_fp16 else None,
    }


def transformers_files(siblings: list[dict]) -> dict:
    """Weights + tokenizer/config files for a transformers/vLLM model (skips duplicates like original/)."""
    names = [s.get("rfilename", "") for s in siblings]
    has_st = any(n.endswith(".safetensors") for n in names)
    chosen = []
    for s in siblings:
        n = s.get("rfilename", "")
        if n.startswith(("original/", "consolidated", "onnx/", "openvino/", "coreml/")):
            continue
        if n.endswith(".gguf"):
            continue
        if n.endswith(".safetensors") or (not has_st and n.endswith(".bin")):
            chosen.append(s)
        elif n.endswith(AUX_EXTS) or n.startswith("tokenizer"):
            chosen.append(s)
    weights = sum(int(s.get("size") or 0) for s in chosen if s["rfilename"].endswith((".safetensors", ".bin")))
    return {"files": [s["rfilename"] for s in chosen], "size": sum(int(s.get("size") or 0) for s in chosen),
            "weights": weights}


def repo_size(siblings: list[dict], skip_ext: tuple[str, ...] = (".onnx",)) -> int:
    return sum(int(s.get("size") or 0) for s in siblings if not s.get("rfilename", "").endswith(skip_ext))
