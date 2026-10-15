"""Model capabilities and reasoning controls, shared by discovery and inference.

Runner metadata takes precedence over template inspection, followed by conservative family hints.
Templates are inspected as text; discovering a model never executes its code or generates tokens.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
ALIASES = {"thinking": "reasoning", "think": "reasoning", "reasoner": "reasoning",
           "image-text-to-text": "vision", "visual-question-answering": "vision",
           "tool-use": "tools", "tool-calling": "tools", "function-calling": "tools"}
CONTROL_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

CTX_FIELDS = ("max_sequence_length", "max_position_embeddings", "n_positions", "seq_length",
              "max_seq_len", "n_ctx")
EXTENDING_ROPE = {"linear", "dynamic", "yarn", "longrope"}


def context_length(config: dict | None) -> int | None:
    """Longest sequence a text model can take, read from its config.json.

    Checks the text stack of multimodal configs, and applies the RoPE factor for checkpoints
    that only record the base window (e.g. ``max_position_embeddings: 2048`` with ``linear`` ×8).
    Configs that already store the extended length report it as-is.
    """
    config = config or {}
    stacks = [config]
    if isinstance(config.get("text_config"), dict):
        stacks.append(config["text_config"])

    def field(name: str) -> int | None:
        for cfg in stacks:
            value = cfg.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None

    declared = [value for name in CTX_FIELDS if (value := field(name)) is not None]
    if not declared:
        return None
    best = max(declared)
    pos = field("max_position_embeddings") or field("n_positions") or best
    for cfg in stacks:
        scaling = cfg.get("rope_scaling")
        if not isinstance(scaling, dict):
            continue
        factor = scaling.get("factor")
        kind = str(scaling.get("type") or scaling.get("rope_type") or "").lower()
        # With original_max_position_embeddings set, the length fields already carry the extension.
        if isinstance(factor, (int, float)) and factor > 1 and kind in EXTENDING_ROPE \
                and not isinstance(scaling.get("original_max_position_embeddings"), int):
            best = max(best, int(pos * factor))
    return best


def _reasoning(supported=None, values=(), source="unknown", toggle=None, default=None) -> dict:
    return {"supported": supported, "efforts": list(values), "source": source, "toggle": toggle,
            "default": default}


def profile(ident: str, *, tags=(), config: dict | None = None, template: str = "",
            thinking: dict | None = None, pipeline: str | None = None, authoritative: bool = False) -> dict:
    config = config or {}
    names = " ".join([ident, str(config.get("model_type", "")), str(config.get("_name_or_path", "")),
                      " ".join(config.get("architectures") or [])]).lower()
    raw = {str(tag).lower() for tag in tags}
    caps = {ALIASES.get(tag, tag) for tag in raw if tag in ALIASES or tag in
            ("reasoning", "vision", "tools", "embedding", "completion", "attached", "imported")}
    reason = _reasoning(True if "reasoning" in caps else None)
    # The runtime's capabilities describe what this loaded artifact can actually do.
    if not authoritative:
        if pipeline in ("image-text-to-text", "visual-question-answering") or config.get("vision_config"):
            caps.add("vision")
        if re.search(r"qwen[23][._-]?vl|qwen3[._][568]|llava|pixtral|internvl|gemma[ -]?[34]", names):
            caps.add("vision")
        if re.search(r"thinking|reasoning|reasoner|deepseek[ -]?r1|qwq|magistral|gpt-oss|qwen3|glm[ -]?5|deepseek[ -]?v4", names):
            caps.add("reasoning")
        if re.search(r"qwen3.*instruct[ -]?2507", names) and "thinking" not in names:
            caps.discard("reasoning")
        if re.search(r"qwen3[._]8", names):
            reason = _reasoning(True, ("none", "low", "medium", "xhigh"), "family", "enable_thinking", "xhigh")
        elif "gpt-oss" in names:
            reason = _reasoning(True, ("low", "medium", "high"), "family", default="medium")
        elif re.search(r"glm[ -]?5[._-]2", names):
            reason = _reasoning(True, ("none", "high", "max"), "family", "enable_thinking", "max")
        elif "qwen3" in names and "reasoning" in caps and not re.search(r"thinking[ -]?2507", names):
            reason = _reasoning(True, ("none", "on"), "family", "enable_thinking")
        elif "reasoning" in caps:
            reason = _reasoning(True, source="family")

    # Named values in a model's template are real controls, not a generic effort scale.
    if template:
        toggle = "enable_thinking" if re.search(r"\benable_thinking\b", template) else None
        if not toggle and re.search(r"(?:\bthinking\b\s*(?:is|==|\|)|\bif\s+(?:not\s+)?thinking\b)", template):
            toggle = "thinking"
        statements = re.findall(r"{%[-+]?([\s\S]*?)[-+]?%}", template)
        effort_statements = " ".join(part for part in statements if "reasoning_effort" in part)
        levels = [value for value in EFFORTS if re.search(r"['\"]" + value + r"['\"]", effort_statements)]
        if levels or toggle:
            if not levels and reason["source"] == "family" and reason["toggle"] is None:
                levels = [value for value in reason["efforts"] if value in EFFORTS]
            values = (["none"] if toggle else []) + (levels or (["on"] if toggle else []))
            default = re.search(r"reasoning_effort\s*\|\s*default\(['\"](\w+)['\"]\)", template)
            reason = _reasoning(True, values, "template", toggle, default.group(1) if default else None)
            caps.add("reasoning")
        elif "<think>" in template or "reasoning_content" in template:
            caps.add("reasoning")
            if reason["supported"] is None:
                reason = _reasoning(True, source="template")
        if re.search(r"\btools\b", " ".join(statements)):
            caps.add("tools")

    declared = thinking if isinstance(thinking, dict) else config.get("thinking")
    if not isinstance(declared, dict):
        for key in ("supported_reasoning_efforts", "reasoning_efforts", "reasoning_effort_options"):
            if isinstance(config.get(key), list):
                declared = {"values": config[key]}
                break
    if isinstance(declared, dict) and isinstance(declared.get("values"), list):
        values, native_values = [], {}
        for value in declared["values"]:
            normalized = "none" if value is False else "on" if value is True else str(value).lower()
            if CONTROL_RE.fullmatch(normalized) and normalized not in values:
                values.append(normalized)
                native_values[normalized] = value
        supported = any(value != "none" for value in values)
        reason = _reasoning(supported, values, "metadata", reason.get("toggle"), declared.get("default"))
        reason["native_values"] = native_values
        if supported:
            caps.add("reasoning")
        else:
            caps.discard("reasoning")
    elif authoritative:
        # Older Ollama builds have capabilities but no thinking.values. Do not fabricate levels.
        reason = _reasoning("reasoning" in caps, source="metadata")

    if reason["supported"] is None and "reasoning" in caps:
        reason["supported"] = True
    return {"capabilities": sorted(caps), "reasoning": reason}


def local_profile(path: str, ident: str = "", *, text_only: bool = False) -> dict:
    root = Path(path)
    config = {}
    template = ""
    for name in ("config.json", "tokenizer_config.json", "processor_config.json", "chat_template.json", "chat_template.jinja"):
        file = root / name
        if not file.is_file() or file.stat().st_size > 2 * 1024**2:
            continue
        try:
            text = file.read_text()
            if name.endswith(".jinja"):
                template = text
            else:
                data = json.loads(text)
                if not isinstance(data, dict):
                    continue
                config.update({key: value for key, value in data.items() if key != "chat_template"})
                chat = data.get("chat_template")
                if isinstance(chat, str):
                    template = chat
                elif isinstance(chat, list):
                    template = next((item["template"] for item in chat if isinstance(item, dict) and item.get("name") == "default"), "")
        except (OSError, ValueError, KeyError, TypeError):
            continue
    result = profile(ident or path, config=config, template=template)
    if not template:
        result["reasoning"]["efforts"] = []  # no template: this runner cannot apply prompt-level controls
    if text_only:
        result["capabilities"] = [cap for cap in result["capabilities"] if cap not in ("vision", "tools")]
    return result


def reasoning_body(body: dict, detected: dict, *, ollama: bool = False) -> dict:
    """Validate an explicit choice and forward it using the runner's actual controls."""
    out = dict(body)
    extra = out.get("chat_template_kwargs") or {}
    if not isinstance(extra, dict):
        raise ValueError("chat_template_kwargs must be an object.")
    kwargs = dict(extra)
    requested = out.get("reasoning_effort", kwargs.get("reasoning_effort"))
    if out.get("reasoning_effort") is not None and kwargs.get("reasoning_effort") is not None:
        if str(out["reasoning_effort"]).lower() != str(kwargs["reasoning_effort"]).lower():
            raise ValueError("reasoning_effort conflicts with chat_template_kwargs.reasoning_effort. Send one choice.")
    if requested is None and "think" in out:
        requested = "on" if out["think"] is True else "none" if out["think"] is False else out["think"]
    if requested is None or requested in ("auto", "default"):
        out.pop("reasoning_effort", None)
        return out
    requested = str(requested).lower()
    if requested == "x-high":
        requested = "xhigh"
    reason = detected.get("reasoning") or {}
    values = reason.get("efforts", [])
    if requested == "none" and reason.get("supported") is False:
        out.pop("reasoning_effort", None)
        return out
    if requested not in values:
        choices = ", ".join(values) or "Auto (model default)"
        raise ValueError(f"Reasoning effort '{requested}' is not supported by this model/runner. Available: {choices}.")
    if ollama:
        out.pop("reasoning_effort", None)
        out["think"] = reason.get("native_values", {}).get(requested,
                            False if requested == "none" else True if requested == "on" else requested)
        return out
    toggle = reason.get("toggle")
    if toggle:
        if toggle in kwargs and kwargs[toggle] is not (requested != "none"):
            raise ValueError(f"reasoning_effort conflicts with chat_template_kwargs.{toggle}. Send one choice.")
        kwargs[toggle] = requested != "none"
    out.pop("think", None)
    if requested in ("none", "on") and toggle:
        out.pop("reasoning_effort", None)
        kwargs.pop("reasoning_effort", None)
    else:
        kwargs["reasoning_effort"] = requested
        out["reasoning_effort"] = requested
    if kwargs:
        out["chat_template_kwargs"] = kwargs
    return out
