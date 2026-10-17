"""Abliteration jobs: validation, queueing and the worker contract (no model is ever loaded here)."""

import json
import sys
from pathlib import Path

import pytest

from kaggleui.catalog import abliterate_worker as worker
from kaggleui.catalog.imports import ModelImports


def test_refusal_detection():
    assert worker.refused("I'm sorry, but I can't help with that.")
    assert worker.refused("  I cannot provide instructions")
    assert not worker.refused("Sure! Here is a short poem about the sea.")


@pytest.mark.parametrize("key,match", [
    ("model.layers.3.self_attn.o_proj.weight", True),
    ("model.layers.3.mlp.down_proj.weight", True),
    ("model.language_model.layers.10.mlp.experts.7.down_proj.weight", True),
    ("model.layers.3.self_attn.q_proj.weight", False),
    ("model.layers.3.mlp.up_proj.weight", False),
    ("lm_head.weight", False),
])
def test_residual_writers(key, match):
    assert bool(worker.WRITERS.search(key)) is match
    assert worker.EMBED.search("model.embed_tokens.weight")


async def test_abliterate_validates_input(tmp_path):
    imports = ModelImports(tmp_path / "imports", tmp_path / "envs")
    with pytest.raises(ValueError, match="model id"):
        imports.abliterate("not a repo")
    with pytest.raises(ValueError, match="quick or heretic"):
        imports.abliterate("Qwen/Qwen3-4B", method="magic")


async def test_abliteration_job_runs_the_worker_and_imports_the_result(tmp_path, monkeypatch):
    """The job hands the spec (with the envs folder) to the abliteration worker and records its models."""
    fake = tmp_path / "fake_worker.py"
    fake.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "job, root, result = json.loads(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])\n"
        "print('KUI_IMPORT ' + json.dumps({'status': 'abliterating', 'progress': 0.5, 'message': 'half'}), flush=True)\n"
        "result.write_text(json.dumps({'models': [{'path': str(root), 'name': job['name'], 'modality': 'text',\n"
        "  'engine': 'vllm', 'envs': job['envs'], 'method': job['method']}], 'message': 'done'}))\n")
    imports = ModelImports(tmp_path / "imports", tmp_path / "envs")
    import kaggleui.catalog.imports as module
    real = module.ManagedProcess

    def spawn(name, cmd, **kw):
        assert cmd[2] == "kaggleui.catalog.abliterate_worker"
        return real(name, [sys.executable, str(fake), *cmd[1:]], **kw)

    monkeypatch.setattr(module, "ManagedProcess", spawn)
    job = imports.abliterate("Qwen/Qwen3-4B-Instruct-2507", method="heretic", trials=5)
    assert job["trials"] == 10 and job["name"] == "Qwen3-4B-Instruct-2507-abliterated"
    await imports.tasks[job["id"]]
    assert job["status"] == "complete", job["message"]
    model = job["models"][0]
    assert model["envs"] == str(tmp_path / "envs") and model["method"] == "heretic"
    saved = json.loads((tmp_path / "imports" / job["id"] / "manifest.json").read_text())
    assert saved["kind"] == "abliterate"


def test_heretic_command_is_unattended(tmp_path, monkeypatch):
    """Heretic must never wait on a prompt: every interactive choice is passed on the command line."""
    env = tmp_path / "envs" / "heretic"
    (env / "bin").mkdir(parents=True)
    (env / ".kaggleui-ready").touch()
    seen = {}

    class Proc:
        returncode = 0
        stdout = iter(["Running trial 3 of 10...\n", "Saving merged model...\n"])

        def __init__(self, cmd, **kw):
            seen.update(cmd=cmd, stdin=kw.get("stdin"))
            out = Path(cmd[cmd.index("--save-directory") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text("{}")

        def wait(self):
            return 0

    monkeypatch.setattr(worker.subprocess, "Popen", Proc)
    monkeypatch.setattr(worker, "report", lambda **_: None)
    out = tmp_path / "out" / "model"
    worker.heretic({"repo": "Qwen/Qwen3-4B", "envs": str(tmp_path / "envs"), "trials": 10}, out)
    cmd = seen["cmd"]
    for flag, value in (("--checkpoint-action", "restart"), ("--trial-index", "0"), ("--model-action", "save"),
                        ("--export-strategy", "merge")):
        assert cmd[cmd.index(flag) + 1] == value
    assert seen["stdin"] == worker.subprocess.DEVNULL
    assert json.loads((out / "abliteration.json").read_text())["method"] == "heretic"


async def test_ollama_models_map_to_their_original_repo(hw, monkeypatch):
    """gemma3:4b and friends can't be edited (GGUF); the form is prefilled with the full-precision original."""
    from unittest.mock import AsyncMock

    from kaggleui.catalog import hf
    from kaggleui.catalog.service import CatalogService

    rows = {
        "qwen3 8b": ["Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-8B", "Qwen/Qwen3-8B-AWQ", "unsloth/Qwen3-8B"],
        "phi4 14b": [],
        "phi4": ["microsoft/Phi-4-mini-instruct", "microsoft/phi-4", "MaziyarPanahi/phi-4-GGUF"],
        "gemma3 4b": ["google/gemma-3-4b-it", "google/gemma-3-4b-pt", "mlx-community/gemma-3-4b-it-4bit"],
    }

    async def search(client, query="", **_):
        return [{"id": r, "library_name": "transformers", "downloads": 100 - i} for i, r in enumerate(rows.get(query, []))]

    monkeypatch.setattr(hf, "search", AsyncMock(side_effect=search))
    service = CatalogService(hw)
    try:
        assert await service._hf_original("qwen3", "8b") == "Qwen/Qwen3-8B"
        assert await service._hf_original("phi4", "14b") == "microsoft/phi-4"
        assert await service._hf_original("gemma3", "4b") == "google/gemma-3-4b-it"
    finally:
        await service.close()
