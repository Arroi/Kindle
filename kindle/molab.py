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
from .config import Paths, disk_budget, molab_tunnel
from .launcher import _read_state, _running, _status_line, launch, stop
from .notify import panel_link
from .system.memory import container_memory
from .tunnels.manager import PROVIDER_LABELS
from .tunnels.persistent import read_record

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
    budget = disk_budget()
    if budget:
        lines.append(f" Budget:   {budget / GB:.0f} GB for models and installs. molab stops sandboxes that write "
                     "more (KINDLE_DISK_BUDGET_GB)")
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
                  f" Panel:    {panel_link(url, state.get('admin_token'), _link_key(state))}",
                  f" API:      {url}/v1   (Anthropic clients: {url})",
                  f" API key:  {state.get('api_key')}",
                  *_machine(state, home), rule,
                  " Pooling: start helper sessions with kindle.molab.start(\"helper\"), then link each one",
                  " under Linked sessions here, and deploy a GGUF with the llama.cpp (GPU pool) engine.",
                  " The head needs disk space for the whole GGUF file."]
    if (state.get("tunnel") or {}).get("provider") == "shayef":
        api = (os.environ.get("SHAYEF_API") or "https://shayef.me").rstrip("/")
        if _link_key(state):
            lines += [rule, " Only requests carrying the API key get through shayef. API clients already send it;",
                      " browsers get it once from the Panel link."]
        lines += [rule, f" Traffic:  {api}/   (shayef's dashboard: who is reaching this link)"]
    lines += [rule, ""]
    print("\n".join(lines), flush=True)


def _link_key(state: dict) -> str | None:
    return (state.get("tunnel") or {}).get("link_key")


TUNNEL_SECRETS = {"shayef": ("SHAYEF_TOKEN",),
                  "burrow": ("BURROW_TOKEN", "BURROW_HOSTNAME"),
                  "cloudflare_named": ("CF_TUNNEL_TOKEN", "CF_TUNNEL_HOSTNAME"),
                  "gradio": ()}


def start(role: str = "head", *, heartbeat: float = 20, home: str | None = None, tunnel: str | None = None) -> None:
    """Where the public link comes from:

    - ``"shayef"``: your address on shayef.me, SHAYEF_NAME.shayef.me (default kindle). Needs SHAYEF_TOKEN.
    - ``"burrow"``: your own domain through burrow. Needs BURROW_TOKEN (the key from burrow's
      dashboard) and BURROW_HOSTNAME, and the tunnel must point at http://127.0.0.1:8890 there.
    - ``"cloudflare_named"``: your own domain through a Cloudflare named tunnel (CF_TUNNEL_TOKEN and
      CF_TUNNEL_HOSTNAME, its public hostname pointed at http://localhost:8890).
    - ``"gradio"``: a gradio.live share link. No account needed.

    By default the head uses shayef when SHAYEF_TOKEN is set, else burrow when its two variables are,
    else Gradio. Helpers default to Gradio: two sessions on one burrow key would split the hostname's
    requests between them, and a second session on one shayef name takes the hostname over. If the
    chosen tunnel doesn't connect, Kindle falls back to a Gradio share link, then to the gradio
    package's own tunnel.
    """
    if role not in ("head", "helper"):
        raise ValueError('role must be "head" or "helper"')
    tunnel = tunnel or (molab_tunnel() if role == "head" else "gradio")
    if tunnel not in TUNNEL_SECRETS:
        raise ValueError('tunnel must be "shayef", "burrow", "gradio" or "cloudflare_named"')
    missing = [name for name in TUNNEL_SECRETS[tunnel] if not os.environ.get(name)]
    if missing:
        raise ValueError(f"The {PROVIDER_LABELS[tunnel]} needs {' and '.join(missing)} set first")
    for folder in old_data_folders():
        from .util import dir_size

        print(f"!! {folder} holds {dir_size(folder) / GB:.1f} GB from an earlier Kindle version. It still counts "
              "toward this sandbox's storage limit. Free it with: import kindle.molab as _k; _k.clean_old_data()",
              flush=True)
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
        for provider in dict.fromkeys((tunnel, "gradio")):
            try:
                state = launch(tunnel=provider, block=False, share_gpu=helper, home=home)
                break
            except RuntimeError as exc:
                failed = _read_state(paths) or {}
                if failed.get("status") != "tunnel_error":
                    raise  # the server itself didn't start: another tunnel won't help
                reason = (failed.get("tunnel") or {}).get("error") or "it never reported a connection"
                fallback = "a Gradio share link" if provider != "gradio" else "the gradio package's tunnel"
                print(f"The {PROVIDER_LABELS[provider]} didn't connect ({reason}). Using {fallback} instead.",
                      flush=True)
                log = _tail(paths.logs / "tunnel-main.log", 25, start=int(read_record(paths).get("log_start", 0)))
                if log.strip() and log != "(no log)":
                    print(f"Its log:\n{log}\n", flush=True)
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
    last_print = 0.0
    watch_until = 0.0
    ready_seen: set[str] = set()
    try:
        while True:
            time.sleep(1)
            # While a deployment works, report every few seconds: if the sandbox dies, the last line
            # still on screen shows which step (download, unpack, load) it was in, and the disk/RAM.
            busy, fresh = _busy(paths, ready_seen)
            if fresh:  # the first minutes after a model loads are when a sandbox has died; watch closely
                watch_until = time.monotonic() + READY_WATCH
            close = busy or time.monotonic() < watch_until
            if time.monotonic() - last_print < (BUSY_INTERVAL if close else heartbeat):
                continue
            last_print = time.monotonic()
            current = _read_state(paths)
            if not _running(current):
                print("Kindle stopped. Its log:", flush=True)
                print(_tail(paths.logs / "server.log", 40), flush=True)
                print("Re-run this cell to start it again, or run kindle.molab.report() for details.", flush=True)
                return
            if not external and current.get("url") and (current["url"].rstrip("/") != url
                                                         or _link_key(current) != _link_key(state)):
                url, state = current["url"].rstrip("/"), current
                print("The public link changed:", flush=True)  # or its key, after the API key is regenerated
                _card(role, url, current, home_dir)
            line = _status_line(current)
            if helper:
                try:
                    pool = _admin(current, "GET", "/_kui/api/pool").json()
                    line = f"pool: {pool.get('status')} · {pool.get('message')}"
                except (httpx.HTTPError, ValueError):
                    pass
            if busy:
                line = " · ".join(busy)
            print(f"[kindle {time.strftime('%H:%M:%S')}] {_memory_line()} · {_gpu_line(current)} · {_disk_line(paths)} · {line}", flush=True)
            _report_failures(current, reported)
    except KeyboardInterrupt:
        print("Cell released. Kindle is still running in the background.", flush=True)


BUSY_INTERVAL = 3
READY_WATCH = 120  # seconds of close reporting after a model becomes ready
WORKING = ("queued", "installing", "downloading", "loading")


def _busy(paths: Paths, ready_seen: set[str]) -> tuple[list[str], bool]:
    """Lines for deployments still working, and whether one just became ready."""
    state = _read_state(paths)
    if not _running(state):
        return [], False
    try:
        deps = _admin(state, "GET", "/_kui/api/deployments").json()
    except (httpx.HTTPError, ValueError):
        return [], False
    out, fresh = [], False
    for d in deps:
        if d.get("status") in WORKING:
            pct = f" {round(d['progress'] * 100)}%" if d.get("progress") else ""
            out.append(f"{d['name']} {d['status']}{pct}: {d.get('message', '')}")
        elif d.get("status") == "ready" and d["id"] not in ready_seen:
            ready_seen.add(d["id"])
            fresh = True
    return out, fresh


def _gpu_line(state: dict) -> str:
    try:
        gpus = _admin(state, "GET", "/_kui/api/system").json().get("gpus") or []
    except (httpx.HTTPError, ValueError):
        return "GPU ?"
    return " ".join(f"GPU {g.get('vram_used', 0) / GB:.1f}/{g.get('vram_total', 0) / GB:.0f} GB" for g in gpus) or "no GPU"


def _memory_line() -> str:
    used, limit = container_memory()
    warn = " (nearly full!)" if limit and used > limit * 0.8 else ""
    return f"RAM {used / GB:.1f}/{limit / GB:.0f} GB{warn}"


def _disk_line(paths: Paths) -> str:
    try:
        return f"disk {shutil.disk_usage(paths.home).free / GB:.0f} GB free"
    except OSError:
        return "disk ?"


def _tail(path, n: int = 40, start: int = 0) -> str:
    try:
        with path.open("rb") as f:
            f.seek(start)
            return "\n".join(f.read().decode(errors="replace").splitlines()[-n:])
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


def old_data_folders() -> list[Path]:
    """Data folders earlier Kindle versions left behind, named ``kindle``.

    In the notebook or home folder, Python can import one as ``kindle`` instead of the real package.
    On the scratch disks (/tmp, /var/tmp, …) they hold old downloads that still count toward the
    sandbox's storage limit, unseen by the current budget. Recognised by Kindle's state.json.
    """
    from .system.memory import SCRATCH

    found = []
    for base in sorted({Path.cwd(), Path.home(), *(Path(root) for root in SCRATCH)}, key=str):
        folder = base / "kindle"
        if (folder / "state.json").is_file() and (folder / "logs").is_dir():
            found.append(folder)
    return found


def clean_old_data() -> None:
    """Delete the old ``kindle`` data folders (models, Ollama and logs from earlier runs)."""
    folders = old_data_folders()
    for folder in folders:
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / GB
        shutil.rmtree(folder, ignore_errors=True)
        print(f"Removed {folder} ({size:.1f} GB)", flush=True)
    if not folders:
        print("No old Kindle data folders found.", flush=True)


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
