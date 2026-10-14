"""Text-to-speech worker: Kokoro, Chatterbox, or the generic transformers text-to-speech pipeline."""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

from fastapi import Body
from fastapi.responses import Response

from ..model_support import audio_adapter, model_config
from .audio import encode
from .common import WorkerState, create_app, device, parse_args, serve, start_loading

# OpenAI voice names → Kokoro voices, so OpenAI clients work unchanged.
OPENAI_TO_KOKORO = {"alloy": "af_alloy", "echo": "am_echo", "fable": "bm_fable", "onyx": "am_onyx",
                    "nova": "af_nova", "shimmer": "af_bella", "ash": "am_adam", "coral": "af_heart",
                    "sage": "af_sarah", "ballad": "bm_george", "verse": "am_michael"}


class KokoroTTS:
    sample_rate = 24000

    def __init__(self, model_path: str):
        from kokoro import KModel

        self.model_path = model_path
        weights = sorted(Path(model_path).glob("*.pth"))
        if not weights:
            raise ValueError("Kokoro needs config.json, a .pth checkpoint and a voices folder")
        self.model = KModel(config=str(Path(model_path) / "config.json"), model=str(weights[0])).to(device()).eval()
        self.pipelines: dict[str, object] = {}
        voices_dir = Path(model_path) / "voices"
        self.voices = sorted(p.stem for p in voices_dir.glob("*.pt"))
        if not self.voices:
            raise ValueError("No Kokoro voice files found. Include voices/*.pt in the imported model.")
        self._pipeline(self.voices[0][0])

    def _pipeline(self, lang: str):
        from kokoro import KPipeline

        if lang not in self.pipelines:
            self.pipelines[lang] = KPipeline(lang_code=lang, model=self.model, device=device())
        return self.pipelines[lang]

    def synth(self, text: str, voice: str | None, speed: float, **_):
        import numpy as np

        voice = OPENAI_TO_KOKORO.get(voice or "", voice) or ("af_heart" if "af_heart" in self.voices else self.voices[0])
        if voice not in self.voices:
            raise ValueError(f"Unknown voice '{voice}'. List available voices at /v1/voices.")
        pipe = self._pipeline(voice[0])
        chunks = [np.asarray(audio) for _, _, audio in pipe(text, voice=str(Path(self.model_path) / "voices" / f"{voice}.pt"), speed=speed)]
        return (np.concatenate(chunks) if chunks else np.zeros(1)), self.sample_rate


class ChatterboxTTS:
    def __init__(self, model_path: str):
        from chatterbox.tts import ChatterboxTTS as _C

        self.model = _C.from_local(model_path, device=device())
        self.sample_rate = self.model.sr
        self.voices = ["default"]

    def synth(self, text: str, voice: str | None, speed: float, reference_b64: str | None = None,
              exaggeration: float = 0.5, cfg_weight: float = 0.5, **_):
        ref_path = None
        if reference_b64:
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            f.write(base64.b64decode(reference_b64))
            f.close()
            ref_path = f.name
        try:
            wav = self.model.generate(text, audio_prompt_path=ref_path, exaggeration=exaggeration, cfg_weight=cfg_weight)
        finally:
            if ref_path:
                os.unlink(ref_path)
        return wav.squeeze().cpu().numpy(), self.sample_rate


class TransformersTTS:
    def __init__(self, model_path: str, options: dict | None = None):
        import torch
        from transformers import pipeline

        dev = 0 if device() == "cuda" else -1
        dtype = torch.float16 if dev == 0 else torch.float32
        self.pipe = pipeline("text-to-speech", model=model_path, device=dev, torch_dtype=dtype,
                             trust_remote_code=(options or {}).get("trust_remote_code") is True)
        self.voices = ["default"]

    def synth(self, text: str, voice: str | None, speed: float, **_):
        import numpy as np

        out = self.pipe(text)
        audio = np.asarray(out["audio"], dtype=np.float32).squeeze()
        return audio, int(out["sampling_rate"])


ENGINES = {"kokoro": KokoroTTS, "chatterbox": ChatterboxTTS, "transformers_tts": TransformersTTS}


def main() -> None:
    args = parse_args()
    state = WorkerState()
    app = create_app(state)

    @app.get("/voices")
    def voices() -> dict:
        tts = state.require_ready()
        return {"voices": tts.voices}

    @app.post("/speech")
    def speech(req: dict = Body(...)) -> Response:
        tts = state.require_ready()
        text = str(req.get("input") or req.get("text") or "").strip()
        if not text:
            return Response("input is empty", status_code=400)
        speed = float(req.get("speed") or 1.0)
        with state.gpu_lock:
            audio, sr = tts.synth(text, req.get("voice"), speed, reference_b64=req.get("reference_audio_b64"),
                                  exaggeration=float(req.get("exaggeration", 0.5)),
                                  cfg_weight=float(req.get("cfg_weight", 0.5)),
                                  **{key: req[key] for key in ("seed", "eval_cfg", "stop_threshold", "min_stop_step", "cfg_scale") if key in req})
        data, media = encode(audio, sr, req.get("response_format") or "mp3")
        return Response(data, media_type=media)

    def loader():
        if args.engine == "transformers_tts":
            from .tts_custom import MiDashengTTS, VibeVoiceTTS

            adapter = audio_adapter(model_config(args.model))
            if adapter == "midasheng":
                tts = MiDashengTTS(args.model, args.options)
            elif adapter in ("vibevoice", "vibevoice_hf"):
                tts = VibeVoiceTTS(args.model, args.options, converted=adapter == "vibevoice_hf")
            else:
                tts = TransformersTTS(args.model, args.options)
        else:
            tts = ENGINES[args.engine](args.model)
        state.info = {"engine": args.engine, "voices": tts.voices[:50]}
        return tts

    start_loading(state, loader)
    serve(app, args.port)


if __name__ == "__main__":
    main()
