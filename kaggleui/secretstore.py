"""Read optional credentials from Kaggle Secrets (Add-ons → Secrets) into the environment."""

from __future__ import annotations

import os

SECRET_NAMES = (
    "HF_TOKEN",
    "NGROK_AUTHTOKEN",
    "NGROK_DOMAIN",
    "CF_TUNNEL_TOKEN",
    "CF_TUNNEL_HOSTNAME",
    "KAGGLEUI_PASSWORD",
    "KAGGLEUI_ADMIN_TOKEN",
    "KAGGLEUI_API_KEY",
    "KAGGLEUI_NOTIFY_URL",
)


def load_kaggle_secrets() -> dict[str, bool]:
    """Copy attached Kaggle secrets into os.environ. Returns which names are available.

    Values are never returned or logged — only whether each one is set.
    """
    found = {name: bool(os.environ.get(name)) for name in SECRET_NAMES}
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except Exception:
        return found
    try:
        client = UserSecretsClient()
    except Exception:
        return found
    for name in SECRET_NAMES:
        if found[name]:
            continue
        try:
            value = client.get_secret(name)
        except Exception:
            continue
        if value:
            os.environ[name] = value
            found[name] = True
    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
    return found


def secret_status() -> dict[str, bool]:
    return {name: bool(os.environ.get(name)) for name in SECRET_NAMES}
