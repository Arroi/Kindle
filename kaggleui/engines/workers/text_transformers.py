"""OpenAI-compatible chat server on plain transformers (fallback when vLLM/Ollama don't fit the model or GPU)."""

from __future__ import annotations

import json
import threading
import time
import uuid

from fastapi import Body
from fastapi.responses import JSONResponse, StreamingResponse

from .common import WorkerState, create_app, parse_args, serve, start_loading, torch_dtype


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content or "")


class Chat:
    def __init__(self, path: str, options: dict):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype(options), device_map="auto")
        self.model.eval()

    def _inputs(self, messages: list[dict], body: dict | None = None):
        body = body or {}
        msgs = [{"role": m.get("role", "user"), "content": _content_text(m.get("content"))} for m in messages]
        extra = dict(body.get("chat_template_kwargs") or {})
        if body.get("reasoning_effort") == "none":
            extra.setdefault("enable_thinking", False)
        try:
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", **extra)
        except TypeError:  # older tokenizers reject unknown template kwargs
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        return ids.to(self.model.device)

    def _gen_kwargs(self, body: dict) -> dict:
        temp = float(body.get("temperature", 0.7) or 0)
        kw = {"max_new_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024),
              "do_sample": temp > 0, "pad_token_id": self.tok.eos_token_id}
        if temp > 0:
            kw["temperature"] = temp
            kw["top_p"] = float(body.get("top_p", 0.95) or 0.95)
        return kw

    def complete(self, body: dict) -> tuple[str, int, int]:
        import torch

        ids = self._inputs(body.get("messages") or [], body)
        with torch.no_grad():
            out = self.model.generate(ids, **self._gen_kwargs(body))
        new = out[0][ids.shape[-1]:]
        return self.tok.decode(new, skip_special_tokens=True), ids.shape[-1], len(new)

    def stream(self, body: dict):
        from transformers import TextIteratorStreamer

        ids = self._inputs(body.get("messages") or [], body)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        t = threading.Thread(target=self.model.generate, kwargs={"input_ids": ids, "streamer": streamer,
                                                                    **self._gen_kwargs(body)})
        t.start()
        yield from streamer
        t.join()


def main() -> None:
    args = parse_args()
    state = WorkerState()
    app = create_app(state)

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": args.name, "object": "model", "owned_by": "kaggleui"}]}

    @app.post("/v1/chat/completions")
    def chat(body: dict = Body(...)):
        model: Chat = state.require_ready()
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        if not body.get("stream"):
            with state.gpu_lock:
                text, p_tok, c_tok = model.complete(body)
            return JSONResponse({
                "id": cid, "object": "chat.completion", "created": created, "model": args.name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok},
            })

        def events():
            def chunk(delta: dict, finish=None) -> str:
                return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                              "model": args.name, "choices": [{"index": 0, "delta": delta,
                                                                               "finish_reason": finish}]}) + "\n\n"

            yield chunk({"role": "assistant", "content": ""})
            with state.gpu_lock:
                for piece in model.stream(body):
                    if piece:
                        yield chunk({"content": piece})
            yield chunk({}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    start_loading(state, lambda: Chat(args.model, args.options))
    serve(app, args.port)


if __name__ == "__main__":
    main()
