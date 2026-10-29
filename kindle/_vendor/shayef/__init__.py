"""shayef - a private address for an AI chat UI running in a notebook.

Colab and Kaggle accept no inbound connections, so a chat UI started there is
reachable only from inside the notebook. shayef gives it a real address on
shayef.me, gated so only you can open it, and a dashboard that shows what is
actually happening.

One cell is enough::

    import shayef
    tunnel = shayef.start(7860, "chat")      # -> https://chat.shayef.me

`start` returns a Tunnel: it has the public URL, the key that gates it, and a
link to the dashboard. A shayef token is needed; set SHAYEF_TOKEN or run
`shayef login`.
"""

from .client import ControlPlane, Session
from .config import Config
from .errors import (ConfigError, ControlPlaneError, RelayError, ShayefError,
                     TunnelError)
from .guard import new_key
from .session import (ACCESS_KEY, ACCESS_OPEN, Tunnel, active, start,
                      stop_all)
from .version import __version__

__all__ = [
    "start",
    "serve",
    "Tunnel",
    "active",
    "stop_all",
    "Config",
    "ControlPlane",
    "Session",
    "new_key",
    "ACCESS_KEY",
    "ACCESS_OPEN",
    "ShayefError",
    "ConfigError",
    "ControlPlaneError",
    "RelayError",
    "TunnelError",
    "__version__",
]


def serve(port, name, **kwargs):
    """Open a tunnel and block until interrupted.

    Handy as the last cell of a notebook, where the kernel would otherwise
    finish and tear the tunnel down with it.
    """
    return start(port, name, **kwargs).wait()
