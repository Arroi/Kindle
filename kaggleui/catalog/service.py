"""Merge curated picks, live Hugging Face / Ollama results and attached models; tag each for this machine."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import yaml

from ..system.probe import HardwareProfile
from ..util import parse_params
from . import attached, hf, ollama
from .fit import Candidate, Fit, assess, gguf_size, pick_quant, rank, weights_from_safetensors
from .resolve import ENGINE_LABELS, MODALITY_PIPELINES, resolve_hf

CURATED_PATH = Path(__file__).with_name("curated.yaml")
VERDICT_ORDER = {"best": 0, "good": 1, "tight": 2, "worst": 3}


class CatalogService:
    def __init__(self, probe: Callable[[], HardwareProfile]):
        self.probe = probe
        self.client = httpx.AsyncClient(timeout=20, headers={"User-Agent": "kaggleui/0.1"})
        self.curated: dict[str, list[dict]] = yaml.safe_load(CURATED_PATH.read_text()) or {}
        self._cache: dict[str, tuple[float, Any]] = {}
        self._sem = asyncio.Semaphore(8)

    async def close(self) -> None:
        await self.client.aclose()

    async def _cached(self, key: str, ttl: float, factory: Callable[[], Any]) -> Any:
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        value = await factory()
        self._cache[key] = (time.time(), value)
        return value

    # =========================================================================================
    # Search
    # =========================================================================================
    async def search(self, modality: str, q: str = "", source: str = "all", sort: str = "recommended",
                     limit: int = 40, show_unsupported: bool = False) -> dict:
        hw = self.probe()
        q = q.strip()
        statuses: dict[str, str] = {}
        items: dict[str, dict] = {}

        def add(item: dict) -> None:
            existing = items.get(item["key"])
            if existing is None:
                items[item["key"]] = item
            elif existing.get("curated") and not item.get("curated"):
                # Keep curated settings but take live popularity numbers.
                for k in ("downloads", "likes", "updated", "capabilities"):
                    if item.get(k) and not existing.get(k):
                        existing[k] = item[k]

        if source in ("all", "curated"):
            for entry in self.curated.get(modality, []):
                if q and q.lower() not in (entry["id"] + " " + entry.get("name", "")).lower():
                    continue
                add(self._from_curated(entry, modality))
            statuses["curated"] = "ok"

        jobs = []
        if source in ("all", "hf") and MODALITY_PIPELINES.get(modality):
            jobs.append(("hf", self._hf_items(modality, q, sort, hw)))
        if source in ("all", "ollama") and modality == "text":
            jobs.append(("ollama", self._ollama_items(q)))
        if source in ("all", "attached"):
            jobs.append(("attached", self._attached_items(modality)))
        results = await asyncio.gather(*(j for _, j in jobs), return_exceptions=True)
        for (name, _), res in zip(jobs, results):
            if isinstance(res, BaseException):
                statuses[name] = f"error: {type(res).__name__}: {res}"[:200]
                if name == "ollama":
                    for raw in ollama.snapshot(q):
                        add(self._from_ollama(raw))
                    statuses[name] += " (showing bundled snapshot)"
                continue
            statuses[name] = "ok"
            for it in res:
                add(it)

        all_items = list(items.values())
        await self._enrich(all_items)
        fits = [self._apply_fit(it, hw) for it in all_items]
        rank(fits)
        for it, f in zip(all_items, fits):
            it["fit"] = f.to_dict()

        if not show_unsupported:
            all_items = [it for it in all_items if it.get("engine")]
        all_items = self._sort(all_items, sort)[:limit]
        return {"modality": modality, "items": all_items, "sources": statuses, "hardware": _hw_summary(hw)}

    @staticmethod
    def _sort(items: list[dict], sort: str) -> list[dict]:
        if sort == "downloads":
            return sorted(items, key=lambda i: -(i.get("downloads") or 0))
        if sort == "newest":
            return sorted(items, key=lambda i: -(i.get("updated") or 0))
        if sort == "trending":
            return sorted(items, key=lambda i: i.get("_order", 1e9))
        return sorted(items, key=lambda i: (VERDICT_ORDER.get(i["fit"]["verdict"], 9), -i["fit"]["score"]))

    # -- item builders ------------------------------------------------------------------------
    def _from_curated(self, e: dict, modality: str) -> dict:
        size = int(float(e.get("size_gb", 0)) * 1e9) or None
        params = float(e["params_b"]) * 1e9 if e.get("params_b") else None
        base = {
            "curated": True, "selftest": bool(e.get("selftest")), "description": e.get("blurb", ""),
            "experimental": bool(e.get("experimental")), "gated": bool(e.get("gated")),
            "options": dict(e.get("options") or {}), "recipe": e.get("recipe"),
            "min_vram": int(float(e["min_vram_gb"]) * 1e9) if e.get("min_vram_gb") else None,
            "largest": int(float(e["largest_gb"]) * 1e9) if e.get("largest_gb") else None,
            "capabilities": e.get("tags", []), "downloads": 0, "likes": 0, "updated": None,
        }
        if e["source"] == "ollama":
            name, tag = ollama.split_ref(e["id"])
            return {
                **base, "key": f"ollama:{name}", "source": "ollama", "id": name, "name": e.get("name", name),
                "modality": "text", "engine": "ollama", "params": params, "size": size, "variant": tag,
                "variants": [{"id": tag, "ref": f"{name}:{tag}", "label": tag, "size": size, "params": params}],
                "url": f"https://ollama.com/library/{name}",
            }
        return {
            **base, "key": f"hf:{e['id']}", "source": "hf", "id": e["id"], "name": e.get("name", e["id"]),
            "modality": modality, "engine": e.get("engine"), "params": params, "size": size,
            "variant": e.get("variant"), "variants": [], "url": f"https://huggingface.co/{e['id']}",
        }

    def _from_hf(self, raw: dict, modality: str, hw: HardwareProfile, order: int) -> dict:
        rid = raw["id"]
        res = resolve_hf(modality, rid, raw.get("library_name"), raw.get("tags"), hw.min_sm)
        st = raw.get("safetensors") or {}
        gg = raw.get("gguf") or {}
        params = gg.get("total") or st.get("total")
        is_gguf = res.engine == "ollama"
        size = None
        if not is_gguf and modality == "text":
            size = weights_from_safetensors(st.get("parameters"))
        dtypes = set((st.get("parameters") or {}).keys())
        return {
            "key": f"hf:{rid}", "source": "hf", "id": rid, "name": rid.split("/")[-1], "modality": modality,
            "engine": res.engine, "description": res.note or "", "params": params, "size": size,
            "variant": None, "variants": [], "downloads": raw.get("downloads") or 0,
            "likes": raw.get("likes") or 0, "updated": hf.parse_time(raw.get("lastModified")),
            "gated": hf.is_gated(raw.get("gated")), "library": raw.get("library_name"),
            "pipeline_tag": raw.get("pipeline_tag"), "capabilities": [], "experimental": res.experimental,
            "curated": False, "options": {}, "url": f"https://huggingface.co/{rid}",
            "bf16_only": dtypes == {"BF16"} and modality in ("image", "video"),
            "context_length": gg.get("context_length"), "_order": order,
        }

    def _from_ollama(self, raw: dict, order: int = 0) -> dict:
        name = raw["name"]
        variants = []
        for s in raw.get("sizes") or []:
            p = parse_params(s)
            variants.append({"id": s, "ref": f"{name}:{s}", "label": s.upper(), "params": p,
                             "size": gguf_size(p, "Q4_K_M") if p else None})
        if not variants:
            variants.append({"id": "latest", "ref": f"{name}:latest", "label": "latest", "params": None,
                             "size": None})
        return {
            "key": f"ollama:{name}", "source": "ollama", "id": name, "name": name, "modality": "text",
            "engine": "ollama", "description": raw.get("description", ""), "params": variants[0]["params"],
            "size": variants[0]["size"], "variant": variants[0]["id"], "variants": variants,
            "downloads": raw.get("downloads", 0), "likes": 0, "updated": None, "gated": False,
            "capabilities": raw.get("capabilities", []), "experimental": False, "curated": False,
            "options": {}, "url": f"https://ollama.com/library/{name}", "_order": order,
        }

    async def _hf_items(self, modality: str, q: str, sort: str, hw: HardwareProfile) -> list[dict]:
        queries: list[dict] = []
        if modality == "text":
            queries = [{"pipeline_tag": "text-generation", "library": "gguf", "limit": 30},
                       {"pipeline_tag": "text-generation", "limit": 20},
                       {"pipeline_tag": "image-text-to-text", "library": "gguf", "limit": 10}]
        elif modality in ("image", "video"):
            queries = [{"pipeline_tag": p, "library": "diffusers", "limit": 30} for p in MODALITY_PIPELINES[modality]]
        else:
            queries = [{"pipeline_tag": p, "limit": 40} for p in MODALITY_PIPELINES[modality]]

        async def run(params: dict) -> list[dict]:
            key = f"hf-search:{params}:{q}:{sort}"
            return await self._cached(key, 900, lambda: hf.search(self.client, query=q, sort=sort, **params))

        batches = await asyncio.gather(*(run(p) for p in queries))
        out, seen = [], set()
        order = 0
        for batch in batches:
            for raw in batch:
                if raw["id"] in seen:
                    continue
                seen.add(raw["id"])
                out.append(self._from_hf(raw, modality, hw, order))
                order += 1
        return out

    async def _ollama_items(self, q: str) -> list[dict]:
        async def fetch() -> list[dict]:
            pages = [ollama.search(self.client, q)]
            if not q:
                pages.append(ollama.search(self.client, "", "embedding"))
            res = await asyncio.gather(*pages)
            merged, seen = [], set()
            for batch in res:
                for r in batch:
                    if r["name"] not in seen:
                        seen.add(r["name"])
                        merged.append(r)
            if not merged:
                raise RuntimeError("no results parsed")
            return merged

        raws = await self._cached(f"ollama-search:{q}", 1800, fetch)
        return [self._from_ollama(r, i) for i, r in enumerate(raws)]

    async def _attached_items(self, modality: str) -> list[dict]:
        root = Path(os.environ.get("KAGGLEUI_INPUT_DIR", "/kaggle/input"))
        found = await asyncio.to_thread(attached.scan, root)
        out = []
        for f in found:
            if f["modality"] != modality:
                continue
            out.append({
                "key": f"attached:{f['path']}", "source": "attached", "id": f["path"], "name": f["name"],
                "modality": modality, "engine": f["engine"], "description": "Attached to this notebook: no download",
                "params": f.get("params"), "size": f["size"], "variant": None, "variants": [],
                "downloads": 0, "likes": 0, "updated": None, "gated": False, "capabilities": ["attached"],
                "experimental": False, "curated": False, "options": {}, "url": None, "attached": True,
            })
        return out

    # -- enrichment ---------------------------------------------------------------------------
    async def _enrich(self, items: list[dict]) -> None:
        """Fill in exact sizes: Ollama registry manifests, HF file listings for non-text models."""
        tasks = []
        for it in items:
            if it["source"] == "ollama":
                for v in it["variants"][:6]:
                    if v.get("exact"):
                        continue
                    tasks.append(self._ollama_variant_size(v))
            elif (it["source"] == "hf" and it.get("engine") and it.get("size") is None
                  and it["modality"] != "text"):
                tasks.append(self._hf_sizes(it))
        if tasks:
            await asyncio.gather(*tasks[:80], return_exceptions=True)
        for it in items:
            if it["source"] == "ollama" and it["variants"]:
                v = next((v for v in it["variants"] if v["id"] == it["variant"]), it["variants"][0])
                it["size"], it["params"] = v.get("size"), v.get("params") or it.get("params")

    async def _ollama_variant_size(self, v: dict) -> None:
        name, tag = ollama.split_ref(v["ref"])

        async def fetch() -> int | None:
            async with self._sem:
                return await ollama.manifest_size(self.client, name, tag)

        size = await self._cached(f"ollama-manifest:{v['ref']}", 86400, fetch)
        if size:
            v["size"] = size
            v["exact"] = True
            if not v.get("params"):
                v["params"] = size / 0.62  # ≈ Q4_K_M bytes per param

    async def _hf_sizes(self, it: dict) -> None:
        async def fetch() -> dict:
            async with self._sem:
                return await hf.model_info(self.client, it["id"])

        info = await self._cached(f"hf-info:{it['id']}", 6 * 3600, fetch)
        sib = info.get("siblings") or []
        if it["engine"] in ("diffusers", "diffusers_video"):
            sel = hf.diffusers_files(sib)
            has_index = any(s.get("rfilename") == "model_index.json" for s in sib)
            if not has_index or sel["size"] == 0:
                it["engine"] = None
                it["description"] = "Not a full diffusers pipeline (single-file checkpoint, LoRA or adapter)"
                return
            it["size"], it["largest"], it["variant"] = sel["size"], sel["largest"], sel["variant"]
        elif it["engine"] in ("transformers_asr", "transformers_tts", "transformers"):
            it["size"] = hf.transformers_files(sib)["weights"] or None
        else:
            it["size"] = hf.repo_size(sib) or None

    # -- fit ----------------------------------------------------------------------------------
    def _candidate(self, it: dict, size: int | None = None, params: float | None = None) -> Candidate:
        return Candidate(
            modality=it["modality"], engine=it.get("engine"), weights_bytes=size if size is not None else it.get("size"),
            params=params if params is not None else it.get("params"), largest_component_bytes=it.get("largest"),
            ctx=int((it.get("options") or {}).get("ctx", 8192)), min_vram=it.get("min_vram"),
            gated=bool(it.get("gated")), bf16_only=bool(it.get("bf16_only")),
            experimental=bool(it.get("experimental")), curated=bool(it.get("curated")),
            downloads=int(it.get("downloads") or 0), likes=int(it.get("likes") or 0), updated=it.get("updated"),
            attached=bool(it.get("attached")),
        )

    def _apply_fit(self, it: dict, hw: HardwareProfile) -> Fit:
        # GGUF repos on HF: choose the best quant for this machine.
        if it["source"] == "hf" and it.get("engine") == "ollama" and it.get("params"):
            quant, f = pick_quant(float(it["params"]), None, self._candidate(it), hw)
            it["variant"] = quant
            it["size"] = gguf_size(float(it["params"]), quant)
            it["spec"] = self._spec(it, f)
            return f
        # Ollama models with several sizes: pick the biggest one that fits comfortably.
        if it["source"] == "ollama" and len(it["variants"]) > 1:
            best = None
            for v in it["variants"]:
                f = assess(self._candidate(it, v.get("size"), v.get("params")), hw)
                v["fit"] = {"verdict": f.verdict, "reasons": f.reasons}
                key = (VERDICT_ORDER[f.verdict] if f.verdict != "worst" else 9, -(v.get("size") or 0))
                if best is None or key < best[0]:
                    best = (key, v, f)
            assert best
            _, v, f = best
            if it.get("curated") and any(x["id"] == it["variant"] for x in it["variants"]):
                pass  # curated entries keep their pinned tag
            else:
                it["variant"], it["size"], it["params"] = v["id"], v.get("size"), v.get("params")
            f = assess(self._candidate(it), hw)
            it["spec"] = self._spec(it, f)
            return f
        f = assess(self._candidate(it), hw)
        it["spec"] = self._spec(it, f)
        return f

    @staticmethod
    def _spec(it: dict, f: Fit) -> dict | None:
        if not it.get("engine"):
            return None
        options = {**(it.get("options") or {}), **f.suggest}
        if it.get("recipe"):
            options["recipe"] = it["recipe"]
        return {
            "modality": it["modality"],
            "model": {"source": it["source"], "id": it["id"], "variant": it.get("variant"), "engine": it["engine"]},
            "options": options,
        }

    # =========================================================================================
    # Detail
    # =========================================================================================
    async def detail(self, key: str) -> dict:
        hw = self.probe()
        source, _, ident = key.partition(":")
        if source == "ollama":
            return await self._ollama_detail(ident, hw)
        if source == "hf":
            return await self._hf_detail(ident, hw)
        raise KeyError(key)

    async def _ollama_detail(self, name: str, hw: HardwareProfile) -> dict:
        tags = await self._cached(f"ollama-tags:{name}", 3600, lambda: ollama.tags(self.client, name))
        curated = next((self._from_curated(e, "text") for e in self.curated.get("text", [])
                        if e["source"] == "ollama" and ollama.split_ref(e["id"])[0] == name), None)
        it = curated or self._from_ollama({"name": name, "sizes": [], "downloads": 0})
        variants = []
        for t in tags:
            if t["tag"] == "latest" or not t.get("size"):
                continue
            p = parse_params(t["tag"]) or (t["size"] / 0.62)
            f = assess(self._candidate(it, t["size"], p), hw)
            variants.append({"id": t["tag"], "ref": f"{name}:{t['tag']}", "label": t["tag"], "size": t["size"],
                             "params": p, "context": t.get("context"), "input": t.get("input"),
                             "fit": {"verdict": f.verdict, "reasons": f.reasons, "warnings": f.warnings}})
        it["variants"] = variants or it["variants"]
        return self._finish_detail(it, hw)

    async def _hf_detail(self, repo: str, hw: HardwareProfile) -> dict:
        info = await self._cached(f"hf-info:{repo}", 6 * 3600, lambda: hf.model_info(self.client, repo))
        modality = None
        for m, entries in self.curated.items():
            if any(e["source"] == "hf" and e["id"] == repo for e in entries):
                modality = m
        if not modality:
            from .resolve import PIPELINE_TO_MODALITY
            modality = PIPELINE_TO_MODALITY.get(info.get("pipeline_tag") or "", "text")
        curated = next((self._from_curated(e, modality) for e in self.curated.get(modality, [])
                        if e["source"] == "hf" and e["id"] == repo), None)
        it = curated or self._from_hf(info, modality, hw, 0)
        sib = info.get("siblings") or []
        it["card"] = (info.get("cardData") or {}).get("license")
        if it.get("engine") == "ollama":
            variants = []
            for g in hf.gguf_variants(sib):
                p = it.get("params") or g["size"] / 0.62
                f = assess(self._candidate(it, g["size"], p), hw)
                variants.append({"id": g["quant"], "ref": f"hf.co/{repo}:{g['quant']}", "label": g["quant"],
                                 "size": g["size"], "params": p,
                                 "fit": {"verdict": f.verdict, "reasons": f.reasons, "warnings": f.warnings}})
            it["variants"] = variants
            ok = [v for v in variants if v["fit"]["verdict"] in ("good",)]
            pick = max(ok, key=lambda v: v["size"]) if ok else (variants[0] if variants else None)
            if pick:
                it["variant"], it["size"] = pick["id"], pick["size"]
        elif it.get("engine") in ("diffusers", "diffusers_video"):
            sel = hf.diffusers_files(sib)
            it["size"], it["largest"], it["variant"] = sel["size"], sel["largest"], sel["variant"]
        elif it.get("engine") in ("vllm", "transformers", "transformers_asr", "transformers_tts"):
            it["size"] = hf.transformers_files(sib)["weights"] or it.get("size")
        else:
            it["size"] = hf.repo_size(sib) or it.get("size")
        it["files"] = len(sib)
        return self._finish_detail(it, hw)

    def _finish_detail(self, it: dict, hw: HardwareProfile) -> dict:
        f = assess(self._candidate(it), hw)
        it["fit"] = f.to_dict()
        it["spec"] = self._spec(it, f)
        it["engine_label"] = ENGINE_LABELS.get(it.get("engine") or "", it.get("engine"))
        it["hardware"] = _hw_summary(hw)
        return it

    def selftest_specs(self) -> list[dict]:
        """Smallest curated model per modality, used by ``kaggleui selftest``."""
        hw = self.probe()
        out = []
        for modality, entries in self.curated.items():
            for e in entries:
                if e.get("selftest"):
                    it = self._from_curated(e, modality)
                    f = assess(self._candidate(it), hw)
                    out.append(self._spec(it, f))
        return [s for s in out if s]


def _hw_summary(hw: HardwareProfile) -> dict:
    return {"accelerator": hw.accelerator_label, "gpus": len(hw.gpus),
            "vram_free": sum(g.vram_free for g in hw.gpus), "ram_available": hw.ram_available,
            "disk_free": hw.models_disk.free if hw.models_disk else None, "fake": hw.fake}
