"""Audio architectures that do not use the generic Transformers TTS pipeline."""

from __future__ import annotations

import base64
import re
import sys
import tempfile
from pathlib import Path

from .common import device, torch_dtype


def _waveform(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value, dtype=np.float32).squeeze()
    if result.ndim != 1 or not result.size:
        raise ValueError("The audio model returned no mono waveform.")
    return result


class MiDashengTTS:
    voices = ["default"]

    def __init__(self, model_path: str, options: dict):
        import torch
        from transformers import AutoModel

        if options.get("trust_remote_code") is not True:
            raise ValueError("MiDashengLM-Gen needs trust_remote_code=True. Enable 'Allow model custom code' in Settings.")
        # Upstream imports its nested models package separately from HF's dynamic modules.
        # Use the downloaded checkpoint's code, keeping it on the selected revision.
        root = Path(model_path).resolve()
        if not (root / "models" / "mdl_gen.py").is_file():
            raise ValueError("MiDashengLM-Gen needs its full repository, including models/**/*.py. Import the repository, not just its weights.")
        sys.path.insert(0, str(root))
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
        self.model.to(device()).eval()  # upstream requires float32 for its mixed audio components
        self.options = options

    def synth(self, text: str, voice: str | None, speed: float, seed=None, eval_cfg=None,
              stop_threshold=None, min_stop_step=None, **_):
        import torch

        if voice not in (None, "", "default"):
            raise ValueError("MiDashengLM-Gen uses prompt-described voices. Use voice='default' and describe the speaker in the prompt.")
        values = {"seed": seed, "eval_cfg": eval_cfg, "stop_threshold": stop_threshold, "min_stop_step": min_stop_step}
        kwargs = {key: value if value is not None else self.options.get(key) for key, value in values.items()}
        with torch.inference_mode():
            result = self.model.generate(text, **{key: value for key, value in kwargs.items() if value is not None})
        return _waveform(result["audio"]), int(result["sample_rate"])


class VibeVoiceTTS:
    voices = ["default"]
    sample_rate = 24000

    def __init__(self, model_path: str, options: dict, *, converted: bool = False):
        import torch

        self.converted, self.options = converted, options
        dtype = torch_dtype(options) if device() == "cuda" else torch.float32
        if converted:
            from transformers import AutoModelForTextToWaveform, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
            self.model = AutoModelForTextToWaveform.from_pretrained(
                model_path, torch_dtype=dtype, device_map="auto", attn_implementation="sdpa", trust_remote_code=False)
            self.sample_rate = int(self.processor.feature_extractor.sampling_rate)
        else:
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

            self.processor = VibeVoiceProcessor.from_pretrained(model_path)
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path, torch_dtype=dtype, device_map=device(), attn_implementation="sdpa")
            self.model.set_ddpm_inference_steps(num_steps=int(options.get("steps") or 10))
        self.model.eval()

    def synth(self, text: str, voice: str | None, speed: float, reference_b64: str | None = None,
              cfg_scale=None, **_):
        import torch

        if voice not in (None, "", "default"):
            raise ValueError("This VibeVoice checkpoint has no named voice presets. Use voice='default', optionally with reference audio.")
        with tempfile.TemporaryDirectory(prefix="kui-vibe-") as tmp:
            ref = None
            if reference_b64:
                ref = Path(tmp) / "reference.wav"
                ref.write_bytes(base64.b64decode(reference_b64, validate=True))
            if self.converted:
                content = [{"type": "text", "text": text}]
                if ref:
                    content.append({"type": "audio", "path": str(ref)})
                inputs = self.processor.apply_chat_template(
                    [{"role": "0", "content": content}], tokenize=True, return_dict=True, add_generation_prompt=True,
                ).to(self.model.device, self.model.dtype)
                with torch.inference_mode():
                    audio = self.model.generate(**inputs, max_new_tokens=int(self.options.get("max_tokens") or 2048))
                return _waveform(audio[0]), self.sample_rate
            script = text if re.search(r"(?im)^\s*Speaker\s+\d+:", text) else "Speaker 1: " + text
            inputs = self.processor(text=[script], voice_samples=[[str(ref)]] if ref else None,
                                    padding=True, return_tensors="pt", return_attention_mask=True)
            inputs = {key: value.to(self.model.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            with torch.inference_mode():
                result = self.model.generate(
                    **inputs, tokenizer=self.processor.tokenizer, max_new_tokens=int(self.options.get("max_tokens") or 2048),
                    cfg_scale=float(cfg_scale if cfg_scale is not None else self.options.get("cfg_scale", 1.3)),
                    generation_config={"do_sample": False}, verbose=False)
            if not result.speech_outputs or result.speech_outputs[0] is None:
                raise ValueError("VibeVoice returned no audio. Try a shorter script or provide reference audio.")
            return _waveform(result.speech_outputs[0]), self.sample_rate
