"""One-cell start on molab (marimo's hosted notebooks).

``kindle.molab.start("head")`` in the session that runs the model and serves the API;
``kindle.molab.start("helper")`` in each session that lends its GPU to a pool. The cell keeps
running and prints a status line every ``heartbeat`` seconds, because molab stops servers whose
cell shows no new output.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import time

import httpx

from .config import Paths
from .launcher import _read_state, _running, _status_line, launch

GB = 1000**3


def _admin(state: dict, method: str, path: str, **kwargs) -> httpx.Response:
    return httpx.request(method, f"http://127.0.0.1:{state['port']}{path}", timeout=15,
                         headers={"Authorization": f"Bearer {state['admin_token']}"}, **kwargs)


def _gradio_package_tunnel(port: int) -> str:
    """Fallback: the share tunnel from the gradio package itself (the same relay, opened in this kernel)."""
    try:
        from gradio import networking
    except ImportError:
        print("Installing gradio for its share tunnel…", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
        from gradio import networking
    # A fresh secret per tunnel: a shared fixed token would let others claim the same tunnel.
    return networking.setup_tunnel(local_host="127.0.0.1", local_port=port, share_token=secrets.token_urlsafe(32),
                                   share_server_address=None, share_server_tls_certificate=None).rstrip("/")


def _machine(state: dict, home: str) -> list[str]:
    lines = []
    try:
        system = _admin(state, "GET", "/_kui/api/system").json()
        for g in system.get("gpus") or []:
            lines.append(f" GPU:      {g.get('name')} · {g.get('vram_total', 0) / GB:.0f} GB VRAM")
        if not system.get("gpus"):
            lines.append(" GPU:      none detected (check the notebook's compute settings)")
        lines.append(f" RAM:      {system.get('ram_total', 0) / GB:.0f} GB")
    except (httpx.HTTPError, ValueError):
        pass
    try:
        free = shutil.disk_usage(home).free
        lines.append(f" Disk:     {free / GB:.0f} GB free at {home}")
    except OSError:
        pass
    return lines


def _card(role: str, url: str, state: dict, home: str) -> None:
    rule = "=" * 72
    lines = ["", rule]
    if role == "helper":
        lines += [" Kindle HELPER: this session lends its GPU to a pool", rule,
                  f" Link:     {url}",
                  f" API key:  {state.get('api_key')}",
                  *_machine(state, home), rule,
                  " In the HEAD session's panel: Linked sessions → paste this link and API key → Link session.",
                  " Keep this cell running. If the link changes (relays expire after ~72 h), link it again."]
    else:
        lines += [" Kindle HEAD: runs the model and serves the API", rule,
                  f" Panel:    {url}/?token={state.get('admin_token')}",
                  f" API:      {url}/v1   (Anthropic clients: {url})",
                  f" API key:  {state.get('api_key')}",
                  *_machine(state, home), rule,
                  " Pooling: start helper sessions with kindle.molab.start(\"helper\"), then link each one",
                  " under Linked sessions here, and deploy a GGUF with the llama.cpp (GPU pool) engine.",
                  " The head needs disk space for the whole GGUF file."]
    lines += [rule, ""]
    print("\n".join(lines), flush=True)


def start(role: str = "head", *, heartbeat: float = 20, home: str | None = None) -> None:
    if role not in ("head", "helper"):
        raise ValueError('role must be "head" or "helper"')
    os.environ.setdefault("KINDLE_PLATFORM", "molab")  # molab can't use Cloudflare quick tunnels
    paths = Paths(home)
    home_dir = str(paths.home)
    helper = role == "helper"

    existing = _read_state(paths)
    # Re-running the cell after a fallback start: keep that server, open a fresh package tunnel for it.
    reuse_external = _running(existing) and ((existing or {}).get("tunnel") or {}).get("provider") == "none"
    state: dict | None = None
    external = False
    if not reuse_external:
        try:
            state = launch(tunnel="gradio", block=False, share_gpu=helper, home=home)
        except RuntimeError as exc:
            print(f"Kindle's own Gradio tunnel didn't connect ({exc}). Using the gradio package's tunnel instead.",
                  flush=True)
            from .launcher import stop

            stop(home)
    if state is None:
        state = launch(tunnel="none", block=False, share_gpu=helper, home=home)
        external = True
    url = _gradio_package_tunnel(state["port"]) if external else state["url"].rstrip("/")

    if helper:  # also when an already-running server was started without sharing
        try:
            _admin(state, "PUT", "/_kui/api/settings", json={"pool": {"share_gpu": True}}).raise_for_status()
        except httpx.HTTPError as exc:
            print(f"Couldn't turn on GPU sharing: {exc}", flush=True)
    _card(role, url, state, home_dir)

    print("Serving. Interrupt this cell to get the notebook back; Kindle keeps running in the background.", flush=True)
    try:
        while True:
            time.sleep(heartbeat)
            current = _read_state(paths)
            if not _running(current):
                print("Kindle stopped. Re-run this cell to start it again.", flush=True)
                return
            if not external and current.get("url") and current["url"].rstrip("/") != url:
                url = current["url"].rstrip("/")
                print("The public link changed:", flush=True)
                _card(role, url, current, home_dir)
            line = _status_line(current)
            if helper:
                try:
                    pool = _admin(current, "GET", "/_kui/api/pool").json()
                    line = f"pool: {pool.get('status')} · {pool.get('message')}"
                except (httpx.HTTPError, ValueError):
                    pass
            print(f"[kindle {time.strftime('%H:%M:%S')}] {line}", flush=True)
    except KeyboardInterrupt:
        print("Cell released. Kindle is still running in the background.", flush=True)
