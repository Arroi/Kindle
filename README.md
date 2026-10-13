# KaggleUI

One notebook cell turns a free Kaggle GPU into a model host. Run it, open the link on your own computer, and pick what to serve from a web control panel:

- **Any model type:** Text (LLMs), Image, Video, Text-to-speech, Speech-to-text, World models.
- **Live catalog** from Hugging Face and Ollama, plus models attached under `/kaggle/input`. Every model is sized against *this* machine's VRAM, RAM and disk and tagged **Best fit**, **Fits**, **Tight** or **Won't fit**, with the reason. Toggle between **All models** and **Fits this GPU**, sort by **Lowest parameters** or **Highest parameters**, and use **Load more** to browse deeper. Parameter sorting uses the shown variant; unknown counts appear last.
- **Interfaces:** the built-in Studio, Gradio, Open WebUI (text) or ComfyUI (image/video).
- **Password lock** for the interfaces (generated, toggleable). The control panel always needs your admin token.
- **APIs:** OpenAI-compatible, Anthropic-compatible and Ollama-compatible endpoints, each toggleable, with an API key.
- **Live system bar:** per-GPU VRAM, RAM, disk, and the time left in the session.

The model catalog offers a compact list with parameter count, download size and GPU fit in separate columns, plus an optional grid. Your sorting and layout choices are remembered between visits. Use **Reset filters** to start a fresh search.

> Unofficial project, not affiliated with Kaggle or Google. Follow Kaggle's terms of use.

## Run it on Kaggle

1. Create a notebook on [kaggle.com](https://www.kaggle.com/code) (**+ Create → New Notebook**).
2. In the right sidebar under **Session options**:
   - **Accelerator:** `GPU T4 x2` (recommended; `GPU P100` also works).
   - **Internet:** **On** (needs a phone-verified Kaggle account).
3. Paste this into a cell and run it:

   ```python
   !curl -fsSL https://raw.githubusercontent.com/UltraFEmotes/KaggleUI/main/install.sh | bash
   import kaggleui
   kaggleui.launch()
   ```

4. After 20–60 seconds the cell prints a card with a `https://….trycloudflare.com/?token=…` link, your admin token, the access password and the API key. Open the link on your computer. The token in the link signs you in.
5. In the panel: **Launch → pick a type → pick a model → choose interface/password/API → Launch**. Install, download and load progress appear live under **Deployments**. When it shows **Ready**, open **Studio** or call the API.

The cell keeps running so the session stays alive. Interrupt it to get the notebook back; the server keeps running in the background. `kaggleui.stop()` shuts it down.

### If the link won't open

Cloudflare quick tunnels get a new hostname whenever they restart. Use the most recent link printed by the running cell. If Safari says it can't find the server but Brave opens the same link, give Safari a moment to retry its DNS lookup and open the link in a new tab.

If the link fails in every browser, reconnect the tunnel from another notebook cell:

```python
import kaggleui
kaggleui.restart_tunnel()
```

This prints a fresh link and keeps your model deployments running. Startup waits for Cloudflare to connect and for the new hostname to resolve before showing the link.

### Hosting for hours (browser closed)

Interactive sessions stop when idle. For long runs, set up a model once, then click **Deployments → Export launch cell** and use that cell in **Save Version → Save & Run All**. Batch runs last up to about 12 hours and auto-deploy everything in the preset. Batch runs don't show output live, so add a `KAGGLEUI_NOTIFY_URL` secret (an [ntfy.sh](https://ntfy.sh) topic URL or a Discord webhook) to get the link pushed to you.

### Keys & tokens (Hugging Face, ngrok, …)

The easiest way: open **Access & API → Keys & tokens** in the panel and paste the key. Your Hugging Face token is checked on the spot, and the panel shows the account it belongs to. Keys entered there stay on the session's temporary disk only, never in `/kaggle/working` or the notebook output.

Or use Kaggle secrets: add them under **Add-ons → Secrets** and **switch each one on for this notebook**. Then press **Reload Kaggle secrets** in the panel, or just re-run the launch cell. Common alternate names work too (`hf_token`, `HUGGINGFACE_TOKEN`, `NGROK_TOKEN`, …).

| Secret | What it does |
|---|---|
| `HF_TOKEN` | Gated Hugging Face models (FLUX, Llama, SD 3.5…). Accept the license on HF first. |
| `NGROK_AUTHTOKEN` (+ `NGROK_DOMAIN`) | Use ngrok instead of Cloudflare: real-time API streaming, and a stable URL if you set a free static domain. |
| `CF_TUNNEL_TOKEN` + `CF_TUNNEL_HOSTNAME` | Cloudflare named tunnel on your own domain (point its public hostname at `http://localhost:8890`). |
| `KAGGLEUI_PASSWORD`, `KAGGLEUI_ADMIN_TOKEN`, `KAGGLEUI_API_KEY` | Keep the same credentials across sessions. |
| `KAGGLEUI_NOTIFY_URL` | Push the panel link to ntfy.sh or a Discord webhook when it starts. |

## Using the API

The base URL is the panel URL. Examples, with the API key from the panel:

```python
from openai import OpenAI
client = OpenAI(base_url="https://YOUR-URL/v1", api_key="kui-…")
print(client.chat.completions.create(model="qwen3:8b", messages=[{"role": "user", "content": "Hi"}]).choices[0].message.content)
```

```python
import anthropic
client = anthropic.Anthropic(base_url="https://YOUR-URL", api_key="kui-…")
print(client.messages.create(model="qwen3:8b", max_tokens=256, messages=[{"role": "user", "content": "Hi"}]).content[0].text)
```

### Thinking / reasoning effort

Thinking models (Qwen3, gpt-oss, DeepSeek-R1…) can reason before answering. In the Studio, the **Thinking** switch in the chat header (and *Settings → Reasoning effort*) picks **Auto / Off / Low / Med / High**. Over the API, use whichever your client speaks:

| API | Parameter |
|---|---|
| OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"` (`none` turns thinking off) |
| Anthropic | `"thinking": {"type": "disabled"}` or `{"type": "enabled", "budget_tokens": 4096}` |
| Ollama | `"think": false \| true \| "low" \| "medium" \| "high"` |

The reasoning streams separately from the answer (`reasoning` in OpenAI deltas, `thinking` blocks in Anthropic, `message.thinking` in Ollama). How much the levels change depends on the model; "off" is the most reliable switch.

| Surface | Endpoints |
|---|---|
| OpenAI | `/v1/chat/completions`, `/v1/embeddings`, `/v1/images/generations`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/videos`, `/v1/models` |
| World models | `/v1/videos` with `actions` plus either `input_reference` (an image) or a `prompt`. A prompt is painted into the first frame by a running image model (pick one with `start_frame_model`). `/v1/start-frames` previews that frame |
| Anthropic | `/v1/messages` (tools, images, streaming) |
| Ollama | `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags` |

**Streaming note:** the default Cloudflare *quick* tunnel buffers streamed HTTP responses, so `stream: true` replies arrive all at once at the end. The Studio streams token by token over WebSockets either way. For live API streaming, use ngrok or a named tunnel.

## What runs where

| Type | Engines |
|---|---|
| Text | Ollama (GGUF from Ollama or any HF GGUF repo; splits big models across both T4s), vLLM (experimental on T4), Transformers |
| Image | Diffusers (SD 1.5/2/XL/3.5, FLUX with 4-bit + offload…), or ComfyUI |
| Video | Diffusers (Wan 2.1/2.2, CogVideoX, LTX-Video…), slow on T4 |
| Text-to-speech | Kokoro, Chatterbox, Transformers TTS |
| Speech-to-text | faster-whisper, Transformers ASR |
| World | Recipes: Oasis 500M (Minecraft, action-conditioned); Matrix-Game 2.0 (heavy, unverified on T4). These start from an image, not text: **describe the scene** and a running image model paints the first frame, or upload one |

Heavy or conflicting stacks (vLLM, Open WebUI, Chatterbox) install into isolated `uv` environments, so they don't break Kaggle's preinstalled PyTorch. Weights go to the ephemeral disk, not `/kaggle/working`.

## Develop locally (no GPU needed)

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
cd web && pnpm install && pnpm build && cd ..
.venv/bin/python -m kaggleui serve --mock-engines --fake-hw t4x2 --tunnel none
# open http://127.0.0.1:8890/?token=<printed token>
.venv/bin/python -m pytest
.venv/bin/python -m kaggleui selftest --mock-engines --fake-hw t4x2
```

`--fake-hw` simulates `t4x2`, `p100`, `tpu` or `cpu`; `--mock-engines` simulates installs, downloads and outputs. On a real GPU, `kaggleui selftest` deploys the smallest model of each type and checks every API surface.

## License

MIT
