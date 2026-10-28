"""Which exact Kindle is installed: the version, plus the git commit for installs from GitHub.

The version alone isn't enough: installing a newer commit with the same version number left the
old server running the old code, because launch() saw nothing to restart.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json

from . import __version__


def installed_build() -> str:
    importlib.invalidate_caches()
    try:
        dist = importlib.metadata.distribution("kindle")
    except importlib.metadata.PackageNotFoundError:
        return __version__
    try:
        commit = (json.loads(dist.read_text("direct_url.json") or "{}").get("vcs_info") or {}).get("commit_id")
    except ValueError:
        commit = None
    return f"{dist.version}+{commit[:12]}" if commit else dist.version
