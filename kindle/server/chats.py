"""Studio conversations, saved on the session's disk so they survive page reloads and other devices see them.

They live as long as the notebook's disk does; Studio offers Markdown/JSON export for keeping them.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MAX_BYTES = 32 * 1024 * 1024  # images are stored inline as data URLs


class ChatStore:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, chat_id: str) -> Path:
        if not ID_RE.fullmatch(chat_id):
            raise ValueError("Invalid conversation id")
        return self.root / f"{chat_id}.json"

    def list(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        out = []
        for file in self.root.glob("*.json"):
            try:
                data = json.loads(file.read_text())
            except (OSError, ValueError):
                continue
            out.append({"id": data.get("id", file.stem), "title": data.get("title") or "Untitled",
                        "model": data.get("model"), "updated": data.get("updated") or file.stat().st_mtime,
                        "messages": len(data.get("messages") or [])})
        return sorted(out, key=lambda c: -c["updated"])

    def get(self, chat_id: str) -> dict | None:
        path = self._path(chat_id)
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None

    def put(self, chat_id: str, data: dict) -> dict:
        path = self._path(chat_id)
        if not isinstance(data, dict) or not isinstance(data.get("messages", []), list):
            raise ValueError("A conversation is an object with a messages list")
        data = {**data, "id": chat_id, "updated": time.time()}
        text = json.dumps(data)
        if len(text) > MAX_BYTES:
            raise ValueError("This conversation is too large to save (over 32 MB). Remove some images or start a new one.")
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        tmp.replace(path)
        return {"id": chat_id, "updated": data["updated"]}

    def delete(self, chat_id: str) -> None:
        self._path(chat_id).unlink(missing_ok=True)
