"""Speech-to-text worker: faster-whisper (CTranslate2) or the transformers ASR pipeline."""

from __future__ import annotations

import os
import tempfile

from fastapi import File, Form, UploadFile

from .common import WorkerState, create_app, device, parse_args, serve, start_loading


class FasterWhisper:
    def __init__(self, model_path: str, options: dict):
        from faster_whisper import WhisperModel

        dev = device()
        compute = options.get("compute_type") or ("float16" if dev == "cuda" else "int8")
        self.model = WhisperModel(model_path, device=dev, compute_type=compute)

    def transcribe(self, path: str, language: str | None, task: str, prompt: str | None, temperature: float,
                   word_timestamps: bool) -> dict:
        segments, info = self.model.transcribe(
            path, language=language or None, task=task, initial_prompt=prompt or None,
            temperature=temperature, vad_filter=True, word_timestamps=word_timestamps)
        segs = []
        for i, s in enumerate(segments):
            item = {"id": i, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
            if word_timestamps and s.words:
                item["words"] = [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)} for w in s.words]
            segs.append(item)
        return {"text": " ".join(s["text"] for s in segs).strip(), "language": info.language,
                "duration": round(info.duration, 3), "segments": segs, "task": task}


class TransformersASR:
    def __init__(self, model_path: str, options: dict):
        import torch
        from transformers import pipeline

        dev = 0 if device() == "cuda" else -1
        self.pipe = pipeline("automatic-speech-recognition", model=model_path, device=dev,
                             torch_dtype=torch.float16 if dev == 0 else torch.float32,
                             trust_remote_code=options.get("trust_remote_code") is True)
        self.is_whisper = "whisper" in type(self.pipe.model).__name__.lower()

    def transcribe(self, path: str, language: str | None, task: str, prompt: str | None, temperature: float,
                   word_timestamps: bool) -> dict:
        kwargs: dict = {"chunk_length_s": 30, "return_timestamps": True}
        if self.is_whisper:
            gen = {"task": task}
            if language:
                gen["language"] = language
            kwargs["generate_kwargs"] = gen
        try:
            out = self.pipe(path, **kwargs)
        except Exception:
            out = self.pipe(path)
        chunks = out.get("chunks") or []
        segs = []
        for i, c in enumerate(chunks):
            ts = c.get("timestamp") or (None, None)
            segs.append({"id": i, "start": ts[0], "end": ts[1], "text": str(c.get("text", "")).strip()})
        return {"text": str(out.get("text", "")).strip(), "language": language, "duration": None,
                "segments": segs, "task": task}


ENGINES = {"faster_whisper": FasterWhisper, "transformers_asr": TransformersASR}


def main() -> None:
    args = parse_args()
    state = WorkerState()
    app = create_app(state)

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...), language: str | None = Form(None), task: str = Form("transcribe"),
                         prompt: str | None = Form(None), temperature: float = Form(0.0),
                         word_timestamps: bool = Form(False)) -> dict:
        model = state.require_ready()
        suffix = os.path.splitext(file.filename or "")[1] or ".audio"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(await file.read())
        tmp.close()
        try:
            with state.gpu_lock:
                return model.transcribe(tmp.name, language, task, prompt, temperature, word_timestamps)
        finally:
            os.unlink(tmp.name)

    def loader():
        model = ENGINES[args.engine](args.model, args.options)
        state.info = {"engine": args.engine}
        return model

    start_loading(state, loader)
    serve(app, args.port)


if __name__ == "__main__":
    main()
