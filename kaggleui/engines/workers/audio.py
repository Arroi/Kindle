"""Audio encoding helpers shared by the TTS worker."""

from __future__ import annotations

import io
import shutil
import subprocess

MEDIA_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "opus": "audio/ogg",
               "aac": "audio/aac", "pcm": "audio/L16"}


def encode(samples, sample_rate: int, fmt: str = "wav") -> tuple[bytes, str]:
    import numpy as np
    import soundfile as sf

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    fmt = (fmt or "wav").lower()
    if fmt == "pcm":
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        return pcm, MEDIA_TYPES["pcm"]
    if fmt in ("wav", "flac"):
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format=fmt.upper())
        return buf.getvalue(), MEDIA_TYPES[fmt]
    wav = io.BytesIO()
    sf.write(wav, audio, sample_rate, format="WAV")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return wav.getvalue(), MEDIA_TYPES["wav"]
    codec = {"mp3": ["-f", "mp3"], "opus": ["-c:a", "libopus", "-f", "ogg"], "aac": ["-c:a", "aac", "-f", "adts"]}
    args = codec.get(fmt)
    if not args:
        return wav.getvalue(), MEDIA_TYPES["wav"]
    proc = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *args, "pipe:1"],
                          input=wav.getvalue(), capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return wav.getvalue(), MEDIA_TYPES["wav"]
    return proc.stdout, MEDIA_TYPES[fmt]
