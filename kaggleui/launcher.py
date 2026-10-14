"""``kaggleui.launch()`` for notebooks: start the server as a background process and show a link card."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from . import __version__
from .config import DEFAULT_PORT, Paths, default_tunnel, load_preset
from .notify import card_html, card_text
from .secretstore import kaggle_sourced, load_kaggle_secrets

STOP_TIMEOUT = 20


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
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
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


def stop(home: str | None = None, *, _preserve_pid: int | None = None) -> bool:
    paths = Paths(home)
    st = _read_state(paths)
    if not _running(st):
        print("KaggleUI isn't running.")
        return False
    try:
        server = psutil.Process(st["pid"])
        started = st.get("process_started_at")
        if started is not None:
            matches = abs(server.create_time() - started) < 0.01
        else:
            cmd = server.cmdline()
            matches = "kaggleui" in cmd and "serve" in cmd
        if not matches:
            print("The recorded KaggleUI process has already exited.", flush=True)
            return False
    except psutil.NoSuchProcess:
        print("KaggleUI isn't running.", flush=True)
        return False

    print("Stopping KaggleUI and its model processes…", flush=True)
    children: dict[tuple[int, float], psutil.Process] = {}

    def remember_children() -> None:
        try:
            for child in server.children(recursive=True):
                if child.pid == _preserve_pid:
                    continue
                children[(child.pid, child.create_time())] = child
        except psutil.NoSuchProcess:
            pass

    remember_children()
    try:
        server.terminate()
    except psutil.NoSuchProcess:
        pass
    deadline = time.monotonic() + STOP_TIMEOUT
    while _alive(server.pid) and time.monotonic() < deadline:
        remember_children()
        time.sleep(0.1)
    # Engines and tunnels each create their own session. Killing only the server's
    # process group misses them, so retain their identities before the parent exits.
    processes = [*children.values(), server]
    for proc in processes:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=2)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=2)
    remaining = [p.pid for p in processes if _alive(p.pid) and p.is_running()]
    if remaining:
        raise RuntimeError(f"KaggleUI could not stop processes {remaining}. See {paths.logs / 'server.log'}.")
    latest = _read_state(paths)
    if latest and latest.get("pid") == server.pid:
        latest["status"] = "stopped"
        tmp = paths.state_file.with_suffix(".stop.tmp")
        tmp.write_text(json.dumps(latest))
        os.chmod(tmp, 0o600)
        tmp.replace(paths.state_file)
    print("KaggleUI stopped.", flush=True)
    return True


def restart_tunnel(home: str | None = None) -> dict:
    """Create a new public link while leaving model deployments running."""
    import httpx

    paths = Paths(home)
    state = _read_state(paths)
    if not _running(state):
        raise RuntimeError("KaggleUI isn't running. Start it with kaggleui.launch().")
    requested = time.time()
    response = httpx.post(f"http://127.0.0.1:{state['port']}/_kui/api/tunnels/main/restart",
                          headers={"Authorization": f"Bearer {state['admin_token']}"}, timeout=10)
    response.raise_for_status()
    print("Reconnecting the public tunnel…", flush=True)
    deadline = time.monotonic() + 100
    while time.monotonic() < deadline:
        latest = _read_state(paths)
        if not _running(latest):
            raise RuntimeError("KaggleUI stopped while reconnecting.")
        if latest.get("updated", 0) > requested:
            if latest.get("status") == "running" and latest.get("url"):
                _show(latest)
                return latest
            if latest.get("status") == "tunnel_error":
                break
        time.sleep(0.5)
    raise RuntimeError(f"The public tunnel didn't connect. Check {paths.logs / 'tunnel-main.log'}.")


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
    if _running(existing):
        raise RuntimeError("KaggleUI is running but its tunnel isn't ready. Run kaggleui.restart_tunnel() "
                           "to reconnect, or kaggleui.stop() before launching again.")

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
    with open(paths.logs / "server.log", "a") as log:
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
        if state and state.get("status") == "tunnel_error":
            print(_tail(paths.logs / "server.log"))
            raise RuntimeError("The public tunnel didn't connect. Run kaggleui.restart_tunnel() to retry.")
        if state and state.get("status") == "running" and state.get("url"):
            break
        time.sleep(0.5)
    if not state or state.get("status") != "running" or not state.get("url"):
        proc.terminate()
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
