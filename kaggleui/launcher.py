"""``kaggleui.launch()`` for notebooks: start the server as a background process and show a link card."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .config import DEFAULT_PORT, Paths, default_tunnel, load_preset
from .notify import card_html, card_text
from .secretstore import kaggle_sourced, load_kaggle_secrets


def _display(html_str: str, text: str) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        if get_ipython() is not None:
            display(HTML(html_str))
            return
    except Exception:
        pass
    print(text, flush=True)


def _read_state(paths: Paths) -> dict | None:
    try:
        return json.loads(paths.state_file.read_text())
    except Exception:
        return None


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # an exited server lingers as a zombie until the notebook process reaps it
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except OSError:
        return True


def _running(state: dict | None) -> bool:
    return bool(state) and state.get("status") != "stopped" and _alive(state.get("pid"))


def installed_version() -> str:
    """Version on disk. In a notebook, re-running ``import kaggleui`` after an upgrade keeps the old
    module (and its ``__version__``) in memory, so that can't be trusted."""
    import importlib
    import importlib.metadata

    importlib.invalidate_caches()
    try:
        return importlib.metadata.version("kaggleui")
    except importlib.metadata.PackageNotFoundError:
        return __version__


def _show(state: dict) -> None:
    note = ""
    if state.get("tunnel", {}) and state["tunnel"].get("provider") == "cloudflare":
        note = "Quick-tunnel URLs change if the tunnel restarts; the new link is printed here."
    _display(card_html(state["url"], state["admin_token"], state.get("password"), state.get("api_key"), note),
             card_text(state["url"], state["admin_token"], state.get("password"), state.get("api_key")))


def stop(home: str | None = None) -> bool:
    paths = Paths(home)
    st = _read_state(paths)
    if not _running(st):
        print("KaggleUI isn't running.")
        return False
    os.kill(st["pid"], signal.SIGTERM)
    for _ in range(40):
        if not _alive(st["pid"]):
            break
        time.sleep(0.5)
    else:
        try:
            os.killpg(st["pid"], signal.SIGKILL)  # launch() starts the server in its own session
        except OSError:
            os.kill(st["pid"], signal.SIGKILL)
        time.sleep(1)
    print("KaggleUI stopped.")
    return True


def launch(preset=None, *, block: bool = True, tunnel: str | None = None, port: int | None = None,
           mock: bool = False, fake_hw: str | None = None, home: str | None = None):
    load_kaggle_secrets()  # Kaggle secrets are readable from the kernel process; pass them down via env.
    paths = Paths(home).ensure()
    existing = _read_state(paths)
    current = installed_version()
    if current != __version__:
        print(f"Note: this notebook has KaggleUI {__version__} imported but {current} is installed. The server will "
              f"run {current}; restart the kernel to update kaggleui.* helpers too.", flush=True)
    if _running(existing) and existing.get("version") != current:
        # pip upgraded the files under a running server: its Python code is old but it would serve the
        # new web UI. Restart so both match. (Servers before 0.1.4 didn't record a version.)
        old = existing.get("version") or "an older version"
        print(f"KaggleUI was updated ({old} → {current}). Restarting the server; relaunch your models "
              "afterwards.", flush=True)
        stop(home)
        existing = None
    if _running(existing) and existing.get("url"):
        print("KaggleUI is already running (kaggleui.stop() to stop it).", flush=True)
        _share_secrets(existing)
        _show(existing)
        if block:
            _block(paths, existing)
        return existing

    p = load_preset(preset)
    cmd = [sys.executable, "-m", "kaggleui", "serve", "--port", str(port or DEFAULT_PORT),
           "--tunnel", tunnel or (p.settings.exposure.provider if p else default_tunnel())]
    if p:
        preset_file = paths.home / "preset.json"
        preset_file.write_text(p.model_dump_json())
        cmd += ["--preset", str(preset_file)]
    if mock:
        cmd.append("--mock-engines")
    if fake_hw:
        cmd += ["--fake-hw", fake_hw]
    if home:
        cmd += ["--home", home]
    if paths.state_file.exists():
        paths.state_file.unlink()
    log = open(paths.logs / "server.log", "a")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PYTHONUNBUFFERED": "1",
                                 "KAGGLEUI_KAGGLE_SECRETS": ",".join(kaggle_sourced())})
    print("Starting KaggleUI…", flush=True)
    deadline = time.time() + 180
    state = None
    while time.time() < deadline:
        if proc.poll() is not None:
            print(_tail(paths.logs / "server.log"))
            raise RuntimeError("KaggleUI exited during startup; see the log above.")
        state = _read_state(paths)
        if state and state.get("status") == "running" and state.get("url"):
            break
        time.sleep(0.5)
    if not state:
        print(_tail(paths.logs / "server.log"))
        raise RuntimeError("KaggleUI didn't start within 3 minutes.")
    _show(state)
    if block:
        _block(paths, state)
    return state


def _share_secrets(state: dict) -> None:
    """Re-running the cell after adding a Kaggle secret: hand the new values to the running server."""
    names = kaggle_sourced()
    if not names:
        return
    try:
        import httpx

        r = httpx.post(f"http://127.0.0.1:{state['port']}/_kui/api/secrets/import",
                       headers={"Authorization": f"Bearer {state['admin_token']}"},
                       json={"values": {n: os.environ[n] for n in names}}, timeout=10)
        if r.status_code == 200 and r.json().get("updated"):
            print("Picked up Kaggle secrets: " + ", ".join(r.json()["updated"]), flush=True)
    except Exception:
        pass


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text().splitlines()[-n:])
    except Exception:
        return ""


def _block(paths: Paths, state: dict) -> None:
    """Keep the cell alive (needed for batch runs); reprint the card if the public URL changes."""
    last_url = state.get("url")
    print("Serving. Interrupt this cell to get the notebook back; the server keeps running.", flush=True)
    try:
        while True:
            time.sleep(5)
            st = _read_state(paths)
            if not _running(st):
                print("KaggleUI stopped.")
                print(_tail(paths.logs / "server.log"))
                return
            if st.get("url") and st["url"] != last_url:
                last_url = st["url"]
                print("Public URL changed:", flush=True)
                _show(st)
    except KeyboardInterrupt:
        print("Cell released. KaggleUI is still running in the background.", flush=True)
