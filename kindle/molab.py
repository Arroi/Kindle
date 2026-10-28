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
from pathlib import Path

import httpx

from . import __version__
from .config import Paths
from .launcher import _read_state, _running, _status_line, launch
from .system.memory import container_memory

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
    reported: set[str] = set()
    try:
        while True:
            time.sleep(heartbeat)
            current = _read_state(paths)
            if not _running(current):
                print("Kindle stopped. Its log:", flush=True)
                print(_tail(paths.logs / "server.log", 40), flush=True)
                print("Re-run this cell to start it again, or run kindle.molab.report() for details.", flush=True)
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
            print(f"[kindle {time.strftime('%H:%M:%S')}] {_memory_line()} · {line}", flush=True)
            _report_failures(current, reported)
    except KeyboardInterrupt:
        print("Cell released. Kindle is still running in the background.", flush=True)


def _memory_line() -> str:
    used, limit = container_memory()
    warn = " (nearly full!)" if limit and used > limit * 0.8 else ""
    return f"RAM {used / GB:.1f}/{limit / GB:.0f} GB{warn}"


def _tail(path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def _report_failures(state: dict, reported: set[str]) -> None:
    """Print each failed deployment's error and last log lines here, once, so nothing fails silently."""
    try:
        deps = _admin(state, "GET", "/_kui/api/deployments").json()
    except (httpx.HTTPError, ValueError):
        return
    for d in deps:
        if d.get("status") != "error" or d["id"] in reported:
            continue
        reported.add(d["id"])
        print(f"\n!! {d['name']} failed: {d.get('error')}", flush=True)
        try:
            lines = _admin(state, "GET", f"/_kui/api/deployments/{d['id']}/logs").json().get("lines", [])
            print("   last log lines:\n   " + "\n   ".join(lines[-15:]) + "\n", flush=True)
        except (httpx.HTTPError, ValueError):
            pass


def report(home: str | None = None) -> None:
    """Print what's needed to diagnose a problem: machine, memory, disks, server and deployment logs.

    Run it in a new cell (``import kindle.molab; kindle.molab.report()``) and share the output.
    """
    import getpass
    import platform

    from .system.memory import fs_type

    paths = Paths(home)
    print("=" * 72)
    print(f"Kindle {__version__} · Python {platform.python_version()} · user {getpass.getuser()} (uid {os.getuid() if hasattr(os, 'getuid') else '?'})")
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi, "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        print(f"GPU: {out or '(nvidia-smi printed nothing)'}")
    else:
        print("GPU: nvidia-smi not found")
    used, limit = container_memory()
    print(f"Memory: {used / GB:.1f} GB used of a {limit / GB:.0f} GB limit")
    for root in sorted({"/", "/tmp", str(paths.home), os.path.expanduser("~"), os.getcwd()}):
        try:
            u = shutil.disk_usage(root)
            print(f"Disk {root}: {u.free / GB:.0f} GB free of {u.total / GB:.0f} GB ({fs_type(Path(root)) or '?'})")
        except OSError:
            pass
    print(f"Kindle home: {paths.home}")
    state = _read_state(paths)
    print(f"Server: {'running' if _running(state) else 'not running'} · {(state or {}).get('url')}")
    print("-" * 72 + "\nserver.log (last 40 lines):\n" + _tail(paths.logs / "server.log", 40))
    if _running(state):
        reported: set[str] = set()
        _report_failures(state, reported)
        if not reported:
            print("No failed deployments.")
    print("=" * 72, flush=True)
