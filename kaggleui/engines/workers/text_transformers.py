"""OpenAI-compatible chat server on plain transformers (fallback when vLLM/Ollama don't fit the model or GPU)."""

from __future__ import annotations

import json
import queue
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
        if any(isinstance(p, dict) and p.get("type") in ("image", "image_url") for p in content):
            raise ValueError("This Transformers fallback accepts text only. Deploy a compatible vision model with Ollama or vLLM for image input.")
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
        if self.tok.chat_template:
            try:
                ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", **extra)
            except TypeError:  # older tokenizers reject unknown template kwargs
                ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        else:
            # Base checkpoints often have no chat template. Give them a plain completion prompt.
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant:"
            ids = self.tok(prompt, return_tensors="pt")
        inputs = dict(ids) if hasattr(ids, "keys") else {"input_ids": ids}
        return {key: value.to(self.model.device) for key, value in inputs.items()}

    def _gen_kwargs(self, body: dict) -> dict:
        temp = float(body.get("temperature", 0.7) or 0)
        kw = {"max_new_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024),
              "do_sample": temp > 0, "pad_token_id": self.tok.eos_token_id, "max_time": 600}
        if temp > 0:
            kw["temperature"] = temp
            kw["top_p"] = float(body.get("top_p", 0.95) or 0.95)
        return kw

    def complete(self, body: dict) -> tuple[str, int, int]:
        import torch

        inputs = self._inputs(body.get("messages") or [], body)
        with torch.no_grad():
            out = self.model.generate(**inputs, **self._gen_kwargs(body))
        length = inputs["input_ids"].shape[-1]
        new = out[0][length:]
        return self.tok.decode(new, skip_special_tokens=True), length, len(new)

    def stream(self, body: dict):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        inputs = self._inputs(body.get("messages") or [], body)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True, timeout=1)
        cancelled = threading.Event()
        errors: list[BaseException] = []

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled.is_set()

        def generate():
            try:
                with torch.inference_mode():
                    self.model.generate(**inputs, streamer=streamer, stopping_criteria=StoppingCriteriaList([Stop()]),
                                        **self._gen_kwargs(body))
            except BaseException as exc:
                errors.append(exc)
                streamer.end()  # unblock the consumer even when generate fails before its first token

        t = threading.Thread(target=generate, daemon=True)
        t.start()
        deadline = time.monotonic() + 600
        try:
            while True:
                try:
                    yield next(streamer)
                except StopIteration:
                    break
                except queue.Empty:
                    if errors or not t.is_alive():
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("Generation took longer than 10 minutes. Reduce the context or output length.") from None
            if errors:
                raise RuntimeError(f"Generation failed: {errors[0]}") from errors[0]
        finally:
            cancelled.set()
            t.join()  # retain the GPU lock until the generation thread has stopped


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
            try:
                with state.gpu_lock:
                    for piece in model.stream(body):
                        if piece:
                            yield chunk({"content": piece})
            except Exception as exc:
                yield "data: " + json.dumps({"error": {"message": str(exc), "type": "generation_error"}}) + "\n\n"
                yield "data: [DONE]\n\n"
                return
            yield chunk({}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    start_loading(state, lambda: Chat(args.model, args.options))
    serve(app, args.port)


if __name__ == "__main__":
    main()
