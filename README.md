# KaggleUI

One notebook cell turns a free Kaggle GPU into a model host. Run it, open the link on your own computer, and pick what to serve from a web control panel:

- **Any model type:** Text (LLMs), Image, Video, Text-to-speech, Speech-to-text, World models.
- **Live catalog** from Hugging Face and Ollama, plus models attached under `/kaggle/input`. Every model is sized against *this* machine's VRAM, RAM and disk and tagged **Best fit**, **Fits**, **Tight** or **Won't fit**, with the reason. Toggle between **All models** and **Fits this GPU**, sort by **Lowest parameters** or **Highest parameters**, and use **Load more** to browse deeper. Parameter sorting uses the shown variant; unknown counts appear last.
- **Interfaces:** the built-in Studio, Gradio, Open WebUI (text) or ComfyUI (image/video).
- **Password lock** for the interfaces (generated, toggleable). The control panel always needs your admin token.
- **APIs:** OpenAI-compatible, Anthropic-compatible and Ollama-compatible endpoints, each toggleable, with an API key.
- **Live system bar:** per-GPU VRAM, RAM, disk, and the time left in the session.

The model catalog offers a compact list with parameter count, download size and GPU fit in separate columns, plus an optional grid. Your sorting and layout choices are remembered between visits. Use **Reset filters** to start a fresh search.

**World discovery** searches Hugging Face tags and families including Matrix-Game, LingBot, Yume, Cosmos and HunyuanWorld. World opens with the broader catalog; **Launchable only** shows configured runtimes, and **Fits this machine** applies the hardware filter. A bundled catalog keeps 20 published checkpoints visible when live search is unavailable.

World runners cover 14 bundled checkpoints: Oasis, Matrix-Game 2.0 and 3.0, LingBot World Base Camera, Cosmos 1.0 Text2World/Video2World 7B, Cosmos Predict 2.5 2B/14B, Cosmos 3 Edge/Nano/Super, HunyuanWorld 1.0, HY-WorldPlay 5B, and HY-World 2.0. The new runners are **experimental and have not been verified with real GPU inference**. Cosmos uses the official Diffusers pipelines, with `diffusers/base/post-trained` for Predict 2.5. Cosmos, Matrix-Game 3.0 and LingBot require an Ampere or newer GPU; T4/P100 are shown as incompatible. Matrix-Game 3.0 uses its upstream native code with SDPA attention; upstream recommends 64 GB RAM. LingBot Base is a large download (about 160 GB). Native runners produce finite video clips; their upstream real-time controls are not exposed in Studio yet.

Cosmos 1.0 Video2World and Cosmos 3 Edge require a starting image. Cosmos 3 Edge exposes image-to-video generation here; robotics action prediction and audio output are not exposed. Gated models require accepting their Hugging Face terms and saving your HF token in **Access & API → Keys & tokens**. GPU fit remains separate from runner availability. Models marked **Runner unavailable** can be downloaded, but an unimplemented architecture still needs a runner.

### Tencent world runners

- **HunyuanWorld 1.0** uses PanoDiT-Text for text prompts, PanoDiT-Image for image prompts, and the official scene reconstruction pipeline. Uploading an existing panorama skips panorama generation. Download the full mesh layers and view smaller previews in Studio. The full pipeline needs the gated FLUX.1-dev and FLUX.1-Fill-dev checkpoints, additional reconstruction models, substantial RAM/disk, and an Ampere or newer GPU; this runner uses a conservative 32 GB VRAM estimate.
- **HY-WorldPlay 5B** generates finite action-conditioned video clips from text or an image. Studio exposes camera pose sequences, chunk count and seed. This integrates the upstream WAN 5B runner; the separate 8B HunyuanVideo runner is not implemented. A compatible CUDA/SageAttention stack and an Ampere or newer GPU are required.
- **HY-World 2.0** combines HY-Pano 2.0, route planning, WorldStereo expansion and Gaussian scene training. Provide an OpenAI-compatible vision-language endpoint that accepts image input, its model name and optional key in Studio. It also requires gated SAM3 access. Upstream recommends multiple large GPUs; the 48 GB VRAM estimate is a lower bound, not a promise of completion on one GPU. T4/P100 are incompatible. Studio supports panorama/mesh/Gaussian viewing and original scene downloads.

These three native adapters follow [Tencent’s official HunyuanWorld code](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0), [WorldPlay code](https://github.com/Tencent-Hunyuan/HY-WorldPlay), and [HY-World 2.0 code](https://github.com/Tencent-Hunyuan/HY-World-2.0). They have **not been exercised on a real GPU in this release**. The catalog includes known base checkpoint sizes; auxiliary models, CUDA builds and intermediate scene exports need additional disk space.

### Download a custom model

Open **Imports** (or **Models → Import from URL**) and paste one of:

- A Hugging Face model URL, such as `https://huggingface.co/owner/model`. An optional revision selects a branch; known world models automatically use their configured checkpoint branch.
- A direct file URL, including Hugging Face `/resolve/` links to GGUF files.
- A ZIP or TAR archive URL. KaggleUI extracts it and scans for model formats.
- An HTTPS Git repository URL. Choose **Git repository** if automatic detection doesn't recognize the provider. Git LFS repositories need `git-lfs` available; unresolved LFS pointers are reported as an error.

Downloads show progress and can be cancelled. `.stop()` stops active download processes too. Completed downloads are remembered across KaggleUI restarts. They stay in **Imports** and are never added to the model catalog or notebook attachments. Click **Customize → Continue to deployment** on a detected checkpoint to choose its settings and launch it. GGUF, Transformers, Whisper, Kokoro, Chatterbox, Diffusers and the configured world checkpoints reuse their existing runners. Source code repositories or standalone weights without a recognizable model configuration are downloaded as files and aren't automatically executed. Restarting the Kaggle notebook itself may clear notebook storage.

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
5. In the panel: **Deploy model → pick a type → pick a model → choose interface/password/API → Launch**. Install, download and load progress appear live under **Deployments**. When it shows **Ready**, open **Studio** or call the API.

The cell keeps running so the session stays alive. Interrupt it to get the notebook back; the server keeps running in the background. `kaggleui.stop()` shuts it down.

### If the link won't open

Cloudflare quick tunnels get a new hostname whenever they restart. Use the most recent link printed by the running cell. If Safari says it can't find the server but Brave opens the same link, give Safari a moment to retry its DNS lookup and open the link in a new tab.

If the link fails in every browser, reconnect the tunnel from another notebook cell:

```python
import kaggleui
kaggleui.restart_tunnel()
```

This prints a fresh link and keeps your model deployments running. Startup waits for Cloudflare to connect and for the new hostname to resolve before showing the link.

### Updates and compatibility

From v0.2.0, the host sees an **Update to …** button when a newer GitHub release has an installable wheel. **System** also offers a manual check and release notes. Installing an update restarts the server and relaunches active deployments; requests in flight are interrupted. Credentials, imported checkpoints and model caches are preserved. Temporary Cloudflare URLs change: re-run `kaggleui.launch()` to display the new link after the update. If installation fails, check `logs/update.log` under the KaggleUI home directory or re-run the install cell.

`release-policy.json` controls the minimum supported version and explicit `revoked_versions`. Compatible releases check this policy periodically, retain the last valid policy across restarts, stop model deployments, and return HTTP 426 for inference when blocked. Host sign-in and updating remain available. A temporary check failure keeps the last known policy; it does not disable a previously supported server. **Versions released before v0.2.0 have no compatibility checker or update button and cannot be retroactively disabled.** Self-hosted installations can also modify the checker.

### Signing in

The public root link opens Studio access for guests. Guests enter the **Studio password**; hosts use their tokenized notebook link or **Host sign-in** (`/host`). API keys authenticate API clients. These are separate credentials. Sign-in verifies that the browser accepted its cookie, and connection errors are shown separately from password errors.

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
| OpenAI | `/v1/chat/completions`, `/v1/embeddings`, `/v1/images/generations`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/voices`, `/v1/videos`, `/v1/worlds`, `/v1/models` |
| World models | `/v1/videos` with `actions` plus either `input_reference` (an image) or a `prompt`. A prompt is painted into the first frame by a running image model (pick one with `start_frame_model`). `/v1/start-frames` previews that frame |
| Anthropic | `/v1/messages` (tools, images, streaming) |
| Ollama | `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags` |

**Streaming note:** the default Cloudflare *quick* tunnel buffers streamed HTTP responses, so `stream: true` replies arrive all at once at the end. Studio uses WebSockets for live replies. If its WebSocket is unavailable, Chat uses a normal HTTP request and displays the reply when generation completes. For live API streaming, use ngrok or a named tunnel.

### TTS voices

`GET /v1/voices` lists voices from **all ready TTS deployments**. `?model=your-model-name` restricts the result. Voice IDs are passed unchanged to `/v1/audio/speech` together with their model name. The existing `GET /v1/audio/voices?model=...` endpoint remains available.

```json
{
  "object": "list",
  "data": [{"id": "af_heart", "object": "voice", "model": "kokoro"}],
  "models": [{"model": "kokoro", "voices": ["af_heart"]}],
  "errors": []
}
```

This is an example shape; each deployed model returns its actual voice list. With no TTS models running, the lists are empty. A runner whose voices cannot be fetched is reported in `errors` without hiding voices from other deployments.

### 3D world API

Submit `POST /v1/worlds` with `model`, `prompt`, and optional `input_reference_b64` / runner settings. Poll `GET /v1/worlds/{id}`, list exports with `GET /v1/worlds/{id}/assets`, stream a file from `/v1/worlds/{id}/assets/{path}`, or download the scene ZIP from `/v1/worlds/{id}/content`. These endpoints use the same API authentication as chat. WorldPlay and other video runners continue to use `/v1/videos`.

## What runs where

| Type | Engines |
|---|---|
| Text | Ollama (GGUF from Ollama or any HF GGUF repo; splits big models across both T4s), vLLM (experimental on T4), Transformers |
| Image | Diffusers (SD 1.5/2/XL/3.5, FLUX with 4-bit + offload…), or ComfyUI |
| Video | Diffusers (Wan 2.1/2.2, CogVideoX, LTX-Video…), slow on T4 |
| Text-to-speech | Kokoro, Chatterbox, Transformers TTS |
| Speech-to-text | faster-whisper, Transformers ASR |
| World | Oasis, Matrix-Game, LingBot, Cosmos, HunyuanWorld 1.0, HY-WorldPlay 5B and HY-World 2.0. Some generate video; the Tencent scene runners export 3D assets. Check the model’s GPU requirements before deploying |

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

MIT. Bundled company/model SVG logos come from [LobeHub Icons](https://github.com/lobehub/lobe-icons) under its MIT license; trademarks belong to their respective owners. See `web/public/brands/LICENSE.txt`.
