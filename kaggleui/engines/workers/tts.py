"""Text-to-speech worker: Kokoro, Chatterbox, or the generic transformers text-to-speech pipeline."""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

from fastapi import Body
from fastapi.responses import Response

from .audio import encode
from .common import WorkerState, create_app, device, parse_args, serve, start_loading

# OpenAI voice names → Kokoro voices, so OpenAI clients work unchanged.
OPENAI_TO_KOKORO = {"alloy": "af_alloy", "echo": "am_echo", "fable": "bm_fable", "onyx": "am_onyx",
                    "nova": "af_nova", "shimmer": "af_bella", "ash": "am_adam", "coral": "af_heart",
                    "sage": "af_sarah", "ballad": "bm_george", "verse": "am_michael"}


class KokoroTTS:
    sample_rate = 24000

    def __init__(self, model_path: str):
        from kokoro import KPipeline  # noqa: F401  (import check)

        self.model_path = model_path
        self.pipelines: dict[str, object] = {}
        voices_dir = Path(model_path) / "voices"
        self.voices = sorted(p.stem for p in voices_dir.glob("*.pt")) if voices_dir.exists() else ["af_heart"]
        self._pipeline("a")

    def _pipeline(self, lang: str):
        from kokoro import KPipeline

        if lang not in self.pipelines:
            self.pipelines[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=device())
        return self.pipelines[lang]

    def synth(self, text: str, voice: str | None, speed: float, **_):
        import numpy as np

        voice = OPENAI_TO_KOKORO.get(voice or "", voice) or "af_heart"
        if voice not in self.voices:
            voice = "af_heart"
        pipe = self._pipeline(voice[0])
        chunks = [np.asarray(audio) for _, _, audio in pipe(text, voice=voice, speed=speed)]
        return (np.concatenate(chunks) if chunks else np.zeros(1)), self.sample_rate


class ChatterboxTTS:
    def __init__(self, _model_path: str):
        from chatterbox.tts import ChatterboxTTS as _C

        self.model = _C.from_pretrained(device=device())
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
    def __init__(self, model_path: str):
        import torch
        from transformers import pipeline

        dev = 0 if device() == "cuda" else -1
        dtype = torch.float16 if dev == 0 else torch.float32
        try:
            self.pipe = pipeline("text-to-speech", model=model_path, device=dev, torch_dtype=dtype)
        except Exception:
            self.pipe = pipeline("text-to-speech", model=model_path, device=dev)
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
                                  cfg_weight=float(req.get("cfg_weight", 0.5)))
        data, media = encode(audio, sr, req.get("response_format") or "mp3")
        return Response(data, media_type=media)

    def loader():
        cls = ENGINES[args.engine]
        tts = cls(args.model)
        state.info = {"engine": args.engine, "voices": tts.voices[:50]}
        return tts

    start_loading(state, loader)
    serve(app, args.port)


if __name__ == "__main__":
    main()
