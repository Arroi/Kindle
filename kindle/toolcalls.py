"""Tool calling: which server-side parser a model family needs, and parsing calls out of raw text.

vLLM and SGLang only return structured ``tool_calls`` when started with a parser that matches the
model's chat template. Runners without a parser (the plain Transformers fallback, llama.cpp builds
that leave calls in the text) get theirs parsed here.
"""

from __future__ import annotations

import json
import re
import uuid

# (pattern over "model_type name", vLLM parser, SGLang parser). First match wins.
PARSERS: tuple[tuple[str, str, str], ...] = (
    (r"qwen3.*coder|qwen3_coder", "qwen3_coder", "qwen3_coder"),
    (r"gpt[-_]?oss", "openai", "gpt-oss"),
    (r"kimi[-_ ]?k2", "kimi_k2", "kimi_k2"),
    (r"deepseek[-_ ]?v3[._]?[12]", "deepseek_v31", "deepseekv31"),
    (r"deepseek_v3|deepseek[-_ ]?v3|deepseek[-_ ]?r1(?!.*distill)", "deepseek_v3", "deepseekv3"),
    (r"glm4_moe|glm[-_ ]?4[._][56]|glm[-_ ]?5", "glm45", "glm45"),
    (r"llama4", "llama4_pythonic", "pythonic"),
    (r"llama[-_ ]?3", "llama3_json", "llama3"),
    (r"mistral|mixtral|ministral|magistral|devstral", "mistral", "mistral"),
    (r"qwen|qwq|hermes", "hermes", "qwen25"),
)


def tool_parser(config: dict | None, ident: str, engine: str) -> str | None:
    """Best parser for ``engine`` ("vllm" or "sglang"), or None when the family is unknown."""
    config = config or {}
    names = " ".join([str(config.get("model_type", "")), " ".join(config.get("architectures") or []), ident]).lower()
    for pattern, vllm, sglang in PARSERS:
        if re.search(pattern, names):
            return vllm if engine == "vllm" else sglang
    return None


def parser_flags(engine: str, parser: str | None) -> list[str]:
    if not parser or parser == "none":
        return []
    if engine == "vllm":
        return ["--enable-auto-tool-choice", "--tool-call-parser", parser]
    return ["--tool-call-parser", parser]


# -- text parsing (runners that return calls inside the content) -------------------------------

_TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
_MISTRAL = re.compile(r"\[TOOL_CALLS\]\s*(\[.*\]|\{.*\})", re.S)
_PYTHON_TAG = re.compile(r"<\|python_tag\|>\s*(.*)", re.S)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def _call(name: str, arguments) -> dict:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if arguments is not None else {})
    return {"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
            "function": {"name": name, "arguments": arguments}}


def _from_json(text: str, known: set[str]) -> list[dict]:
    try:
        value = json.loads(text)
    except ValueError:
        return []
    items = value if isinstance(value, list) else [value]
    calls = []
    for item in items:
        if not isinstance(item, dict):
            return []
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if not isinstance(name, str) or (known and name not in known):
            return []
        calls.append(_call(name, fn.get("arguments", fn.get("parameters", {}))))
    return calls


def parse_tool_calls(text: str, tools: list[dict] | None) -> tuple[str, list[dict]]:
    """Split generated text into (remaining content, OpenAI ``tool_calls``).

    Understands Hermes/Qwen ``<tool_call>`` tags, Mistral ``[TOOL_CALLS]``, Llama 3.1
    ``<|python_tag|>`` and a bare JSON object or list naming one of the offered tools. Text that
    doesn't parse stays content, so a model that answers in prose is never turned into a call.
    """
    known = {t.get("function", t).get("name") for t in tools or [] if isinstance(t, dict)}
    known.discard(None)
    tagged = _TAGGED.findall(text)
    if tagged:
        calls = [c for block in tagged for c in _from_json(block, known)]
        if calls:
            return _TAGGED.sub("", text).strip(), calls
    for pattern in (_MISTRAL, _PYTHON_TAG):
        match = pattern.search(text)
        if match:
            calls = _from_json(match.group(1).strip(), known)
            if calls:
                return text[:match.start()].strip(), calls
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    candidate = fenced.group(1) if fenced else stripped
    if candidate[:1] in "[{" and known:
        calls = _from_json(candidate, known)
        if calls:
            return "", calls
    return text, []


def template_messages(messages: list[dict]) -> list[dict]:
    """OpenAI messages → the shape Hugging Face chat templates expect (arguments as objects)."""
    out = []
    for m in messages:
        msg = {k: v for k, v in m.items() if k in ("role", "content", "name", "tool_call_id")}
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args or "{}")
                    except ValueError:
                        pass
                calls.append({"id": tc.get("id"), "type": "function", "function": fn})
            msg["tool_calls"] = calls
        if msg.get("content") is None:
            msg["content"] = ""
        out.append(msg)
    return out
