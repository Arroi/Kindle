"""KaggleUI — one-cell control panel for hosting AI models on Kaggle's free GPUs."""

from __future__ import annotations

__version__ = "0.1.4"


def launch(preset=None, *, block: bool = True, tunnel: str | None = None, port: int | None = None):
    """Start KaggleUI from a notebook cell.

    Starts the server in a background process, prints a link card with the admin token,
    and (when ``block`` is true) keeps the cell alive so batch ("Save & Run All") sessions
    don't end while models are being served.
    """
    from .launcher import launch as _launch

    return _launch(preset=preset, block=block, tunnel=tunnel, port=port)
