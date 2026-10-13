"""Auto-generated Gradio UI for one deployment. Talks to KaggleUI's own gateway on localhost."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time

import httpx


def main() -> None:
    import gradio as gr

    p = argparse.ArgumentParser()
    p.add_argument("--modality", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--root-path", default="")
    p.add_argument("--share", action="store_true")
    a = p.parse_args()
    headers = {"x-kui-internal": os.environ.get("KAGGLEUI_INTERNAL_KEY", "")}
    client = httpx.Client(base_url=a.api, headers=headers, timeout=httpx.Timeout(connect=10, read=1800, write=120, pool=10))
    title = f"{a.model} · KaggleUI"

    if a.modality == "text":
        def respond(message, history):
            msgs = [{"role": h["role"], "content": h["content"]} for h in history if isinstance(h.get("content"), str)]
            msgs.append({"role": "user", "content": message})
            text = ""
            with client.stream("POST", "/v1/chat/completions",
                               json={"model": a.model, "messages": msgs, "stream": True}) as r:
                for line in r.iter_lines():
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[5:])
                    except ValueError:
                        continue
                    for ch in chunk.get("choices") or []:
                        text += (ch.get("delta") or {}).get("content") or ""
                    yield text

        demo = gr.ChatInterface(respond, type="messages", title=title)

    elif a.modality == "image":
        def generate(prompt, negative, steps, guidance, width, height, seed, n):
            from PIL import Image

            body = {"model": a.model, "prompt": prompt, "negative_prompt": negative or None, "steps": int(steps),
                    "guidance": float(guidance), "width": int(width), "height": int(height), "n": int(n)}
            if seed is not None and int(seed) >= 0:
                body["seed"] = int(seed)
            r = client.post("/v1/images/generations", json=body)
            r.raise_for_status()
            return [Image.open(io.BytesIO(base64.b64decode(d["b64_json"]))) for d in r.json()["data"]]

        with gr.Blocks(title=title) as demo:
            gr.Markdown(f"## {a.model}")
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", lines=3)
                    negative = gr.Textbox(label="Negative prompt")
                    with gr.Row():
                        steps = gr.Slider(1, 60, value=25, step=1, label="Steps")
                        guidance = gr.Slider(0, 15, value=5, step=0.5, label="Guidance")
                    with gr.Row():
                        width = gr.Slider(256, 1536, value=1024, step=64, label="Width")
                        height = gr.Slider(256, 1536, value=1024, step=64, label="Height")
                    with gr.Row():
                        seed = gr.Number(value=-1, label="Seed (-1 = random)")
                        n = gr.Slider(1, 4, value=1, step=1, label="Images")
                    go = gr.Button("Generate", variant="primary")
                gallery = gr.Gallery(label="Result", columns=2)
            go.click(generate, [prompt, negative, steps, guidance, width, height, seed, n], gallery)

    elif a.modality == "tts":
        try:
            voices = client.get("/v1/audio/voices", params={"model": a.model}).json().get("voices") or ["default"]
        except Exception:
            voices = ["default"]

        def speak(text, voice, speed):
            r = client.post("/v1/audio/speech", json={"model": a.model, "input": text, "voice": voice,
                                                      "speed": float(speed), "response_format": "wav"})
            r.raise_for_status()
            path = f"/tmp/kui-tts-{int(time.time() * 1000)}.wav"
            with open(path, "wb") as f:
                f.write(r.content)
            return path

        with gr.Blocks(title=title) as demo:
            gr.Markdown(f"## {a.model}")
            text = gr.Textbox(label="Text", lines=5)
            with gr.Row():
                voice = gr.Dropdown(voices, value=voices[0], label="Voice")
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
            go = gr.Button("Speak", variant="primary")
            audio = gr.Audio(label="Audio", type="filepath")
            go.click(speak, [text, voice, speed], audio)

    elif a.modality == "stt":
        def transcribe(path, language, task):
            if not path:
                return ""
            with open(path, "rb") as f:
                files = {"file": (os.path.basename(path), f.read())}
            data = {"model": a.model, "response_format": "text"}
            if language:
                data["language"] = language
            r = client.post("/v1/audio/translations" if task == "translate" else "/v1/audio/transcriptions",
                            files=files, data=data)
            r.raise_for_status()
            return r.text

        with gr.Blocks(title=title) as demo:
            gr.Markdown(f"## {a.model}")
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
            with gr.Row():
                language = gr.Textbox(label="Language (blank = auto)", max_lines=1)
                task = gr.Radio(["transcribe", "translate"], value="transcribe", label="Task")
            go = gr.Button("Transcribe", variant="primary")
            out = gr.Textbox(label="Transcript", lines=8)
            go.click(transcribe, [audio, language, task], out)

    else:  # video / world
        def make(prompt, image, frames, steps, seed, progress=gr.Progress()):
            data = {"model": a.model, "prompt": prompt or "", "frames": str(int(frames)), "steps": str(int(steps))}
            if seed is not None and int(seed) >= 0:
                data["seed"] = str(int(seed))
            files = None
            if image:
                with open(image, "rb") as f:
                    files = {"input_reference": (os.path.basename(image), f.read())}
            r = client.post("/v1/videos", data=data, files=files)
            r.raise_for_status()
            vid = r.json()["id"]
            while True:
                s = client.get(f"/v1/videos/{vid}").json()
                progress(s.get("progress", 0) / 100, desc=s.get("message") or s.get("status"))
                if s.get("status") == "completed":
                    break
                if s.get("status") == "failed":
                    raise gr.Error((s.get("error") or {}).get("message", "Generation failed"))
                time.sleep(2)
            content = client.get(f"/v1/videos/{vid}/content")
            path = f"/tmp/kui-video-{vid}.mp4"
            with open(path, "wb") as f:
                f.write(content.content)
            return path

        with gr.Blocks(title=title) as demo:
            gr.Markdown(f"## {a.model}")
            with gr.Row():
                with gr.Column():
                    if a.modality == "world":
                        prompt = gr.Textbox(label="Describe the starting scene (painted by a running image model)",
                                            lines=3, info="World models start from an image. Leave empty to use the "
                                                          "uploaded image, or the model's sample frame.")
                    else:
                        prompt = gr.Textbox(label="Prompt", lines=3)
                    image = gr.Image(label="Start image (optional)", type="filepath")
                    with gr.Row():
                        frames = gr.Slider(8, 161, value=33, step=1, label="Frames")
                        steps = gr.Slider(1, 60, value=25, step=1, label="Steps")
                    seed = gr.Number(value=-1, label="Seed (-1 = random)")
                    go = gr.Button("Generate", variant="primary")
                video = gr.Video(label="Result")
            go.click(make, [prompt, image, frames, steps, seed], video)

    auth = None
    if a.share and os.environ.get("KAGGLEUI_GRADIO_PASSWORD"):
        auth = ("kaggle", os.environ["KAGGLEUI_GRADIO_PASSWORD"])
    demo.queue().launch(server_name="127.0.0.1", server_port=a.port, root_path=None if a.share else (a.root_path or None), share=a.share,
                        auth=auth, show_api=False, prevent_thread_lock=False)


if __name__ == "__main__":
    main()
