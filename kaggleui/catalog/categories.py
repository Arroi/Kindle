"""Separate speech synthesis, general audio generation and standalone 3D checkpoints."""

from __future__ import annotations

import re

_AUDIO = re.compile(r"midashenglm-gen|musicgen|musicldm|audioldm|stable[-_]audio|audiogen|(?:ace[-_]step)|riffusion|(?:^|/)tango[0-9-]", re.I)
_SPEECH = re.compile(r"kokoro|chatterbox|vibevoice|bark|speecht5|mms-tts|cosyvoice|f5-tts|fish-speech|outetts|orpheus|dia-|parler|speech|(?:^|[-_/])tts(?:[-_/]|$)", re.I)


def audio_family(ident: str) -> bool:
    return bool(_AUDIO.search(ident))


def audio_category(raw: dict) -> str:
    ident = raw.get("id", "")
    if audio_family(ident):
        return "audio"
    tags = {str(tag).lower() for tag in raw.get("tags") or []}
    if raw.get("pipeline_tag") == "text-to-speech" or _SPEECH.search(ident) or tags & {"tts", "text-to-speech", "speech-synthesis"}:
        return "tts"
    if tags & {"music", "music-generation", "audio-generation", "sound-effects"}:
        return "audio"
    return "tts"  # ambiguous legacy text-to-audio tags often describe speech checkpoints


MESH_ENTRIES = [
    {"id": "stabilityai/TripoSR", "name": "TripoSR", "blurb": "Fast single-image 3D object reconstruction."},
    {"id": "stabilityai/stable-fast-3d", "name": "Stable Fast 3D", "blurb": "Image-to-3D textured mesh generation."},
    {"id": "stabilityai/stable-point-aware-3d", "name": "Stable Point Aware 3D", "blurb": "Image and point-conditioned 3D asset generation."},
    {"id": "microsoft/TRELLIS-image-large", "name": "TRELLIS Image Large", "blurb": "Image-conditioned 3D assets with structured latents."},
    {"id": "microsoft/TRELLIS.2-4B", "name": "TRELLIS.2 4B", "blurb": "Image-to-3D generation with shape and material attributes."},
    {"id": "tencent/Hunyuan3D-2", "name": "Hunyuan3D 2", "blurb": "Tencent's shape generation and texture synthesis checkpoints."},
    {"id": "tencent/Hunyuan3D-2.1", "name": "Hunyuan3D 2.1", "blurb": "Image-conditioned 3D shape and PBR texture generation."},
    {"id": "tencent/Hunyuan3D-2mini", "name": "Hunyuan3D 2 Mini", "blurb": "Smaller Hunyuan3D image-to-shape checkpoint."},
    {"id": "TencentARC/InstantMesh", "name": "InstantMesh", "blurb": "Single-image mesh reconstruction using multiview generation."},
    {"id": "VAST-AI/TripoSG", "name": "TripoSG", "blurb": "Image-to-3D shape generation."},
]
