"""Optional credentials: from Kaggle Secrets (Add-ons → Secrets), the environment, or entered in Kindle.

Values live in os.environ (so engines and subprocesses inherit them). Ones entered in Kindle are also
kept in ``$KINDLE_HOME/secrets.json`` (mode 600, on the ephemeral disk), so they survive a server
restart within the session. They are never written to /kaggle/working, which Kaggle saves as output.
Values are never returned to the browser or logged; only whether each is set and where it came from.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SECRET_NAMES = (
    "HF_TOKEN",
    "NGROK_AUTHTOKEN",
    "NGROK_DOMAIN",
    "CF_TUNNEL_TOKEN",
    "CF_TUNNEL_HOSTNAME",
    "BURROW_TOKEN",
    "BURROW_HOSTNAME",
    "KINDLE_PASSWORD",
    "KINDLE_ADMIN_TOKEN",
    "KINDLE_API_KEY",
    "KINDLE_NOTIFY_URL",
)

# Secrets that can be set from the panel. Credentials for Kindle itself are managed on the Access page.
EDITABLE = ("HF_TOKEN", "NGROK_AUTHTOKEN", "NGROK_DOMAIN", "CF_TUNNEL_TOKEN", "CF_TUNNEL_HOSTNAME",
            "BURROW_TOKEN", "BURROW_HOSTNAME", "KINDLE_NOTIFY_URL")

# Kaggle secret labels are case-sensitive and people name them differently.
ALIASES: dict[str, tuple[str, ...]] = {
    "HF_TOKEN": ("HF_TOKEN", "hf_token", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
                 "HF_API_TOKEN", "HUGGINGFACE_API_KEY", "huggingface_token", "huggingface", "HF"),
    "NGROK_AUTHTOKEN": ("NGROK_AUTHTOKEN", "NGROK_AUTH_TOKEN", "NGROK_TOKEN", "ngrok_token", "ngrok"),
    "KINDLE_NOTIFY_URL": ("KINDLE_NOTIFY_URL", "NTFY_URL", "DISCORD_WEBHOOK", "DISCORD_WEBHOOK_URL"),
}

_sources: dict[str, str] = {}  # name -> "kaggle" | "env" | "kindle"
_kaggle_status: dict[str, str | None] = {"state": "unknown", "error": None}
_local_path: Path | None = None


def _apply(name: str, value: str, source: str) -> None:
    os.environ[name] = value
    _sources[name] = source
    if name == "HF_TOKEN":
        os.environ["HUGGING_FACE_HUB_TOKEN"] = value


def load_kaggle_secrets(refresh: bool = False) -> dict[str, bool]:
    """Copy attached Kaggle secrets into os.environ. ``refresh`` re-reads ones already set from Kaggle."""
    from_kernel = set(filter(None, os.environ.get("KINDLE_KAGGLE_SECRETS", "").split(",")))
    for name in SECRET_NAMES:
        if os.environ.get(name) and name not in _sources:
            _sources[name] = "kaggle" if name in from_kernel else "env"
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except Exception:
        _kaggle_status.update(state="unavailable", error="Not running in a Kaggle notebook")
        return secret_status()
    try:
        client = UserSecretsClient()
    except Exception as e:  # noqa: BLE001
        _kaggle_status.update(state="error", error=f"{type(e).__name__}: {e}"[:200])
        return secret_status()
    _kaggle_status.update(state="available", error=None)
    for name in SECRET_NAMES:
        if os.environ.get(name) and not (refresh and _sources.get(name) == "kaggle"):
            continue
        for label in ALIASES.get(name, (name,)):
            try:
                value = client.get_secret(label)
            except Exception:
                continue
            if value and value.strip():
                _apply(name, value.strip(), "kaggle")
                break
    return secret_status()


def init_local(home: Path) -> None:
    """Load secrets saved from the panel earlier in this session."""
    global _local_path
    _local_path = home / "secrets.json"
    try:
        saved = json.loads(_local_path.read_text())
    except Exception:
        return
    for name, value in saved.items():
        if name in EDITABLE and value and not os.environ.get(name):
            _apply(name, value, "kindle")


def _persist() -> None:
    if not _local_path:
        return
    data = {n: os.environ[n] for n in EDITABLE if _sources.get(n) == "kindle" and os.environ.get(n)}
    tmp = _local_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.chmod(tmp, 0o600)
    tmp.replace(_local_path)


def set_secret(name: str, value: str, source: str = "kindle") -> None:
    if name not in EDITABLE:
        raise KeyError(name)
    _apply(name, value.strip(), source)
    _persist()


def clear_secret(name: str) -> None:
    if name not in EDITABLE:
        raise KeyError(name)
    os.environ.pop(name, None)
    if name == "HF_TOKEN":
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    _sources.pop(name, None)
    _persist()


def kaggle_sourced() -> list[str]:
    return [n for n, src in _sources.items() if src == "kaggle" and os.environ.get(n)]


def secret_status() -> dict[str, bool]:
    return {name: bool(os.environ.get(name)) for name in SECRET_NAMES}


def secret_details() -> dict:
    return {
        "items": [{"name": n, "set": bool(os.environ.get(n)), "source": _sources.get(n) if os.environ.get(n) else None,
                   "editable": n in EDITABLE} for n in SECRET_NAMES],
        "kaggle": dict(_kaggle_status),
    }
