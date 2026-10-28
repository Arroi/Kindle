"""Agent-facing behaviour: tool-call parsers, tool calls and thinking across API translations, model aliases."""

import json

from kindle.engines.base import UpstreamError
from kindle.gateway.translate import (
    anthropic_to_openai,
    ollama_chat_to_openai,
    openai_stream_to_anthropic,
    openai_stream_to_ollama,
    openai_to_anthropic,
    openai_to_ollama_chat,
)
from kindle.toolcalls import parse_tool_calls, parser_flags, template_messages, tool_parser

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]


async def _agen(items, fail: Exception | None = None):
    for i in items:
        yield i
    if fail:
        raise fail


def _events(raw: list[str]) -> list[dict]:
    return [json.loads(line[6:]) for block in raw for line in block.splitlines() if line.startswith("data: ")]


def test_tool_parser_matches_families():
    assert tool_parser({"model_type": "qwen3"}, "Qwen/Qwen3-8B", "vllm") == "hermes"
    assert tool_parser({"model_type": "qwen3"}, "Qwen/Qwen3-8B", "sglang") == "qwen25"
    assert tool_parser({"model_type": "qwen3_moe"}, "Qwen/Qwen3-Coder-30B-A3B-Instruct", "vllm") == "qwen3_coder"
    assert tool_parser({"model_type": "llama"}, "meta-llama/Llama-3.1-8B-Instruct", "vllm") == "llama3_json"
    assert tool_parser({"model_type": "glm4_moe"}, "zai-org/GLM-4.5-Air", "vllm") == "glm45"
    assert tool_parser({"model_type": "gpt_oss"}, "openai/gpt-oss-20b", "sglang") == "gpt-oss"
    assert tool_parser({"model_type": "deepseek_v3"}, "deepseek-ai/DeepSeek-V3.1", "vllm") == "deepseek_v31"
    # A Qwen-based R1 distill uses the Qwen template, not DeepSeek's
    assert tool_parser({"model_type": "qwen2"}, "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "vllm") == "hermes"
    assert tool_parser({"model_type": "gpt2"}, "gpt2", "vllm") is None
    assert parser_flags("vllm", "hermes") == ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    assert parser_flags("sglang", "qwen25") == ["--tool-call-parser", "qwen25"]
    assert parser_flags("vllm", "none") == []


def test_parse_tool_calls_formats():
    content, calls = parse_tool_calls('Sure.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call>', TOOLS)
    assert content == "Sure." and calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Oslo"}
    _, calls = parse_tool_calls('[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "A"}}]', TOOLS)
    assert len(calls) == 1
    _, calls = parse_tool_calls('<|python_tag|>{"name": "get_weather", "parameters": {"city": "B"}}', TOOLS)
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "B"}
    _, calls = parse_tool_calls('```json\n{"name": "get_weather", "arguments": {}}\n```', TOOLS)
    assert len(calls) == 1
    # prose and JSON naming an unknown tool stay content
    assert parse_tool_calls("It is sunny.", TOOLS) == ("It is sunny.", [])
    assert parse_tool_calls('{"name": "rm_rf", "arguments": {}}', TOOLS)[1] == []


def test_template_messages_decode_arguments():
    msgs = template_messages([{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"}])
    assert msgs[0]["content"] == "" and msgs[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    assert msgs[1] == {"role": "tool", "tool_call_id": "c1", "content": "ok"}


def test_tool_result_images_reach_vision_models():
    out = anthropic_to_openai({"messages": [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "text", "text": "screenshot"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}}]}]}],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}})
    tool, user = out["messages"]
    assert tool == {"role": "tool", "tool_call_id": "t1", "content": "screenshot"}
    assert user["role"] == "user" and user["content"][1]["image_url"]["url"] == "data:image/png;base64,AAA"
    assert out["parallel_tool_calls"] is False


async def test_anthropic_stream_thinking_only_when_requested():
    chunks = [{"choices": [{"delta": {"reasoning_content": "hmm"}}]},
              {"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 12, "completion_tokens": 3}}]
    events = _events([e async for e in openai_stream_to_anthropic(_agen(chunks), "m", thinking=True)])
    starts = [e["content_block"]["type"] for e in events if e["type"] == "content_block_start"]
    assert starts == ["thinking", "text"]
    assert any(e.get("delta", {}).get("thinking") == "hmm" for e in events)
    assert any(e.get("delta", {}).get("type") == "signature_delta" for e in events)
    delta = next(e for e in events if e["type"] == "message_delta")
    assert delta["usage"] == {"input_tokens": 12, "output_tokens": 3}
    plain = _events([e async for e in openai_stream_to_anthropic(_agen(chunks), "m")])
    assert [e["content_block"]["type"] for e in plain if e["type"] == "content_block_start"] == ["text"]


async def test_anthropic_stream_reports_upstream_failure_in_band():
    chunks = [{"choices": [{"delta": {"content": "par"}}]}]
    raw = [e async for e in openai_stream_to_anthropic(_agen(chunks, UpstreamError(500, "CUDA out of memory")), "m")]
    events = _events(raw)
    assert events[-1]["type"] == "error" and "CUDA out of memory" in events[-1]["error"]["message"]
    types = [e["type"] for e in events]
    assert types.count("content_block_start") == types.count("content_block_stop")


def test_non_stream_thinking_block():
    resp = {"choices": [{"message": {"content": "4", "reasoning_content": "2+2"}, "finish_reason": "stop"}]}
    assert openai_to_anthropic(resp, "m", thinking=True)["content"][0] == {"type": "thinking", "thinking": "2+2", "signature": ""}
    assert openai_to_anthropic(resp, "m")["content"][0]["type"] == "text"


async def test_ollama_stream_carries_tool_calls_and_thinking():
    chunks = [{"choices": [{"delta": {"reasoning_content": "need weather"}}]},
              {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "get_weather", "arguments": '{"ci'}}]}}]},
              {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "Rome"}'}}]}, "finish_reason": "tool_calls"}]}]
    lines = [json.loads(line) async for line in openai_stream_to_ollama(_agen(chunks), "m")]
    assert lines[0]["message"]["thinking"] == "need weather"
    call = next(x for x in lines if x.get("message", {}).get("tool_calls"))
    assert call["message"]["tool_calls"] == [{"function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]
    assert lines[-1]["done"] is True and lines[-1]["done_reason"] == "stop"


def test_ollama_tool_results_get_call_ids():
    out = ollama_chat_to_openai({"messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "X"}}}]},
        {"role": "tool", "content": "sunny", "tool_name": "get_weather"}]})
    call_id = out["messages"][1]["tool_calls"][0]["id"]
    assert out["messages"][2] == {"role": "tool", "tool_call_id": call_id, "content": "sunny", "name": "get_weather"}
    done = openai_to_ollama_chat({"choices": [{"message": {"content": "", "tool_calls": [
        {"function": {"name": "f", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}, "m")
    assert done["done_reason"] == "stop"
