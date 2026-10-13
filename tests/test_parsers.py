from pathlib import Path

from kaggleui.catalog import hf, ollama
from kaggleui.catalog.attached import scan
from kaggleui.catalog.resolve import resolve_hf
from kaggleui.util import parse_params, parse_size

FIX = Path(__file__).parent / "fixtures"


def test_ollama_search_parse():
    items = ollama.parse_search((FIX / "ollama_search.html").read_text())
    assert len(items) == 4
    first = items[0]
    assert first["name"] and first["description"]
    assert "tools" in first["capabilities"]
    assert all(s[-1] in "bm" for s in first["sizes"])
    assert first["downloads"] > 1000


def test_ollama_tags_parse_ignores_digest_hex():
    tags = ollama.parse_tags((FIX / "ollama_tags.html").read_text())
    assert tags[0]["tag"] == "latest"
    sizes = {t["tag"]: t["size"] for t in tags}
    assert all(v and v > 100_000_000 for v in sizes.values()), sizes
    assert tags[0]["context"] == "40K"


def test_cloud_only_models_are_skipped():
    assert not ollama.is_local_model({"name": "x", "sizes": [], "downloads": 0})
    assert ollama.is_local_model({"name": "x", "sizes": [], "downloads": 5})


def test_parse_helpers():
    assert parse_size("abc0b12 • 5.2GB • 40K") == 5_200_000_000
    assert parse_size("e0b9") is None
    assert parse_params("qwen3:8b") == 8e9
    assert parse_params("Qwen3-30B-A3B") == 30e9
    assert parse_params("Kokoro-82M") == 82e6
    assert parse_params("whisper-large-v3") is None


def test_gguf_variants_group_shards_and_skip_mmproj():
    sib = [{"rfilename": "m-Q4_K_M.gguf", "size": 100}, {"rfilename": "m-Q8_0-00001-of-00002.gguf", "size": 60},
           {"rfilename": "m-Q8_0-00002-of-00002.gguf", "size": 60}, {"rfilename": "mmproj-F16.gguf", "size": 5},
           {"rfilename": "README.md", "size": 1}]
    v = {g["quant"]: g for g in hf.gguf_variants(sib)}
    assert set(v) == {"Q4_K_M", "Q8_0"}
    assert v["Q8_0"]["size"] == 120 and len(v["Q8_0"]["files"]) == 2


def test_diffusers_files_prefers_fp16_and_skips_root_checkpoints():
    sib = [{"rfilename": "model_index.json", "size": 1},
           {"rfilename": "sd_xl_base_1.0.safetensors", "size": 7000},
           {"rfilename": "unet/config.json", "size": 1},
           {"rfilename": "unet/diffusion_pytorch_model.safetensors", "size": 1000},
           {"rfilename": "unet/diffusion_pytorch_model.fp16.safetensors", "size": 500},
           {"rfilename": "vae/diffusion_pytorch_model.safetensors", "size": 80},
           {"rfilename": "text_encoder/model.fp16.safetensors", "size": 120}]
    sel = hf.diffusers_files(sib)
    assert "sd_xl_base_1.0.safetensors" not in sel["files"]
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in sel["files"]
    assert "unet/diffusion_pytorch_model.safetensors" not in sel["files"]
    assert "vae/diffusion_pytorch_model.safetensors" in sel["files"]  # no fp16 variant → plain file
    assert sel["variant"] == "fp16" and sel["largest"] == 500


def test_resolver():
    assert resolve_hf("text", "a/b-GGUF", "gguf", [], 75).engine == "ollama"
    assert resolve_hf("text", "a/b", "transformers", ["safetensors"], 75).engine == "vllm"
    assert resolve_hf("text", "a/b", "transformers", ["safetensors"], 60).engine == "transformers"
    assert resolve_hf("text", "a/b", "mlx", [], 75).engine is None
    assert resolve_hf("image", "a/b-lora", "diffusers", ["lora"], 75).engine is None
    assert resolve_hf("stt", "openai/whisper-small", "transformers", ["whisper"], 75).engine == "transformers_asr"
    assert resolve_hf("stt", "Systran/faster-whisper-tiny", "ctranslate2", [], 75).engine == "faster_whisper"
    r = resolve_hf("tts", "x/custom-tts", "transformers", [], 75)
    assert r.engine == "transformers_tts" and r.experimental


def test_attached_scan(tmp_path):
    (tmp_path / "llm").mkdir()
    (tmp_path / "llm" / "model-Q4_K_M.gguf").write_bytes(b"x" * 10)
    sd = tmp_path / "sdxl" / "pytorch" / "default" / "1"
    (sd / "unet").mkdir(parents=True)
    (sd / "model_index.json").write_text('{"_class_name": "StableDiffusionXLPipeline"}')
    (sd / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 20)
    wh = tmp_path / "whisper"
    wh.mkdir()
    (wh / "config.json").write_text('{"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"}')
    found = {f["modality"]: f for f in scan(tmp_path)}
    assert found["text"]["engine"] == "ollama" and found["text"]["size"] == 10
    assert found["image"]["engine"] == "diffusers" and found["image"]["size"] == 20
    assert found["stt"]["engine"] == "transformers_asr"
