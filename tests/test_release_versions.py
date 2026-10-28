"""Version stamps must agree across the server, the packaging metadata, and the web bundle.

vite bakes ``__APP_VERSION__`` from ``web/package.json`` into the bundle when ``pnpm build``
runs. Building the bundle *before* bumping the version shipped a wheel whose page reported
the previous version, which made the UI show a permanent "page is older than the server"
banner. These tests catch that class of release mistake.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from kindle import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_version_matches_package():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml has no version"
    assert match.group(1) == __version__


def test_web_package_version_matches_package():
    pkg = json.loads((ROOT / "web" / "package.json").read_text(encoding="utf-8"))
    assert pkg["version"] == __version__


def test_built_web_bundle_carries_current_version():
    bundles = sorted((ROOT / "kindle" / "static" / "assets").glob("index-*.js"))
    if not bundles:
        pytest.skip("web bundle not built (run `pnpm build` in web/ first)")
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in bundles)
    stamped = sorted(set(re.findall(r"`(\d+\.\d+\.\d+)`", text)))
    assert __version__ in stamped, (
        f"the built web bundle is stamped {stamped or ['nothing']}, not {__version__} — "
        "bump the version, then rebuild with `cd web && pnpm build` before building the wheel"
    )
