"""Where are we running?

shayef has to behave differently on Colab (output is proxied through a kernel
port, filesystem is wiped between sessions), on Kaggle (no inbound ports at
all, /kaggle/working survives a commit) and on a normal Jupyter or terminal
session. Everything that branches on the host lives here.
"""

import os
import platform
import sys

COLAB = "colab"
KAGGLE = "kaggle"
JUPYTER = "jupyter"
VSCODE = "vscode"
TERMINAL = "terminal"


class Environment(object):
    def __init__(self, kind, in_notebook, persist_dir=None, notes=()):
        self.kind = kind
        self.in_notebook = in_notebook
        self.persist_dir = persist_dir
        self.notes = list(notes)

    @property
    def is_colab(self):
        return self.kind == COLAB

    @property
    def is_kaggle(self):
        return self.kind == KAGGLE

    @property
    def ephemeral(self):
        """True when the filesystem is wiped when the session ends, which is
        why named tunnels need credentials that can be re-fetched rather than
        only stored on disk."""
        return self.kind in (COLAB, KAGGLE)

    def describe(self):
        return {
            "kind": self.kind,
            "in_notebook": self.in_notebook,
            "python": platform.python_version(),
            "platform": "%s-%s" % (platform.system().lower(), platform.machine().lower()),
            "persist_dir": self.persist_dir,
            "notes": self.notes,
        }

    def __repr__(self):
        return "<Environment %s notebook=%s>" % (self.kind, self.in_notebook)


def _ipython_shell():
    try:
        from IPython import get_ipython  # noqa: F401
    except Exception:
        return None
    try:
        from IPython import get_ipython
        return get_ipython()
    except Exception:
        return None


def in_notebook():
    shell = _ipython_shell()
    if shell is None:
        return False
    return shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell", "GoogleColabShell")


def detect():
    notebook = in_notebook()

    if "google.colab" in sys.modules or os.path.isdir("/content"):
        notes = []
        drive = "/content/drive/MyDrive"
        persist = drive if os.path.isdir(drive) else None
        if persist is None:
            notes.append(
                "Colab wipes the filesystem when the session ends. Mount Drive "
                "(or pass a token each run) to keep your shayef config."
            )
        return Environment(COLAB, notebook, persist_dir=persist, notes=notes)

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle"):
        persist = "/kaggle/working" if os.path.isdir("/kaggle/working") else None
        return Environment(
            KAGGLE, notebook, persist_dir=persist,
            notes=["Kaggle allows no inbound connections, so the tunnel is the "
                   "only way to reach the app. Internet must be enabled on the notebook."],
        )

    if notebook:
        kind = VSCODE if os.environ.get("VSCODE_PID") else JUPYTER
        return Environment(kind, True)

    return Environment(TERMINAL, False)


def home_dir():
    """Where shayef keeps its config and the relay connector."""
    override = os.environ.get("SHAYEF_HOME")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(os.path.expanduser("~"), ".shayef")
