"""OpenAI-compatible chat server on plain transformers (fallback when vLLM/Ollama don't fit the model or GPU)."""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid

from fastapi import Body
from fastapi.responses import JSONResponse, StreamingResponse

from ...capabilities import local_profile, reasoning_body
from ...toolcalls import parse_tool_calls, template_messages
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

        trust = options.get("trust_remote_code") is True
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype(options), device_map="auto",
                                                       trust_remote_code=trust)
        self.model.eval()
        self.profile = local_profile(path, text_only=True)

    def _inputs(self, messages: list[dict], body: dict | None = None):
        body = body or {}
        msgs = template_messages([{**m, "content": _content_text(m.get("content"))} for m in messages])
        normalized = reasoning_body(body, self.profile)
        extra = dict(normalized.get("chat_template_kwargs") or {})
        if body.get("tools") and body.get("tool_choice") != "none":
            if not self.tok.chat_template:
                raise ValueError("This checkpoint has no chat template, so it can't be offered tools.")
            extra["tools"] = body["tools"]
        if self.tok.chat_template:
            try:
                ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", **extra)
            except TypeError as exc:  # do not silently discard an explicit reasoning or tools choice
                if extra:
                    raise ValueError("This tokenizer could not apply the requested reasoning controls or tools.") from exc
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

    def complete(self, body: dict) -> tuple[str, int, int, str]:
        import torch

        inputs = self._inputs(body.get("messages") or [], body)
        kwargs = self._gen_kwargs(body)
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        length = inputs["input_ids"].shape[-1]
        new = out[0][length:]
        # Tool-call markers are often special tokens; keep them when tools were offered so they can be parsed.
        text = self.tok.decode(new, skip_special_tokens=not body.get("tools"))
        if body.get("tools"):
            text = text.replace(self.tok.eos_token or "\0", "").replace(self.tok.pad_token or "\0", "")
        finish = "length" if len(new) >= kwargs["max_new_tokens"] else "stop"
        return text, length, len(new), finish

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False)) if text else 0

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
        return {"object": "list", "data": [{"id": args.name, "object": "model", "owned_by": "kindle"}]}

    @app.post("/v1/chat/completions")
    def chat(body: dict = Body(...)):
        model: Chat = state.require_ready()
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        tools = body.get("tools") if body.get("tool_choice") != "none" else None

        def finished() -> tuple[dict, str, dict]:
            with state.gpu_lock:
                text, p_tok, c_tok, finish = model.complete(body)
            message: dict = {"role": "assistant", "content": text}
            if tools:
                content, calls = parse_tool_calls(text, tools)
                if calls:
                    message = {"role": "assistant", "content": content or None, "tool_calls": calls}
                    finish = "tool_calls"
            return message, finish, {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok}

        if not body.get("stream"):
            message, finish, usage = finished()
            return JSONResponse({
                "id": cid, "object": "chat.completion", "created": created, "model": args.name,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage,
            })

        def events():
            def chunk(delta: dict, finish=None, usage=None) -> str:
                data = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": args.name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage:
                    data["usage"] = usage
                return "data: " + json.dumps(data) + "\n\n"

            yield chunk({"role": "assistant", "content": ""})
            try:
                if tools:
                    # A call can only be recognised once it's complete, so tool requests are generated whole.
                    message, finish, usage = finished()
                    if message.get("content"):
                        yield chunk({"content": message["content"]})
                    for i, call in enumerate(message.get("tool_calls") or []):
                        yield chunk({"tool_calls": [{"index": i, **call}]})
                    yield chunk({}, finish, usage)
                    yield "data: [DONE]\n\n"
                    return
                pieces = []
                with state.gpu_lock:
                    for piece in model.stream(body):
                        if piece:
                            pieces.append(piece)
                            yield chunk({"content": piece})
                c_tok = model.count("".join(pieces))
                finish = "length" if c_tok >= model._gen_kwargs(body)["max_new_tokens"] else "stop"
                yield chunk({}, finish, {"prompt_tokens": 0, "completion_tokens": c_tok, "total_tokens": c_tok})
            except Exception as exc:
                yield "data: " + json.dumps({"error": {"message": str(exc), "type": "generation_error"}}) + "\n\n"
                yield "data: [DONE]\n\n"
                return
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    start_loading(state, lambda: Chat(args.model, args.options))
    serve(app, args.port)


if __name__ == "__main__":
    main()
