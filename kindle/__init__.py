"""Kindle — one-cell control panel for hosting AI models on Kaggle's free GPUs."""

from __future__ import annotations

__version__ = "0.2.16"


def launch(preset=None, *, block: bool = True, tunnel: str | None = None, port: int | None = None,
           heartbeat: float | None = 20):
    """Start Kindle from a notebook cell.

    Starts the server in a background process, prints a link card with the admin token,
    and (when ``block`` is true) keeps the cell alive so batch ("Save & Run All") sessions
    don't end while models are being served. While blocking, a status line is printed at
    least every ``heartbeat`` seconds so hosted notebooks don't treat the cell as idle.
    """
    from .launcher import launch as _launch

    return _launch(preset=preset, block=block, tunnel=tunnel, port=port, heartbeat=heartbeat)


def stop(home: str | None = None) -> bool:
    """Stop a Kindle server started with :func:`launch` (and the models it runs)."""
    from .launcher import stop as _stop

    return _stop(home)


def restart_tunnel(home: str | None = None) -> dict:
    """Reconnect the public URL without restarting running models."""
    from .launcher import restart_tunnel as _restart

    return _restart(home)
