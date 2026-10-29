"""Finding, downloading and caching the relay connector.

A notebook user should not have to install anything before opening a tunnel,
so shayef fetches the connector itself and caches it under ~/.shayef/bin as
`shayef-relay`.

Where it downloads from is not compiled in: the control plane is asked. An
operator who points shayef.me at their own mirror moves the download with it.
"""

import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request

from . import env as envmod
from .errors import RelayError
from .log import log
from .version import __version__

RELAY_NAME = "shayef-relay"


def bin_dir():
    return os.path.join(envmod.home_dir(), "bin")


def relay_path():
    suffix = ".exe" if os.name == "nt" else ""
    return os.path.join(bin_dir(), RELAY_NAME + suffix)


def ensure_relay(control=None, force_download=False):
    """Return a path to a runnable relay connector.

    control        a ControlPlane, asked where to download from
    force_download ignore the cache and fetch a fresh copy
    """
    override = os.environ.get("SHAYEF_RELAY")
    if override and not force_download:
        if not os.path.exists(override):
            raise RelayError(
                "SHAYEF_RELAY points at %s, which does not exist." % override)
        log.debug("using relay from SHAYEF_RELAY", path=override)
        return override

    cached = relay_path()
    if not force_download and _runnable(cached):
        log.debug("using cached relay", path=cached)
        return cached

    if control is None:
        raise RelayError(
            "No relay connector is installed and shayef cannot ask where to "
            "get one.",
            hint=("Run `shayef install` while signed in, or point "
                  "SHAYEF_RELAY at a connector binary."))

    return _download(control)


# Connector builds are versioned by date, e.g. 2026.10.5.
_VERSION_RE = re.compile("([0-9]{4}[.][0-9]+[.][0-9]+)")


def relay_version(path=None):
    """The connector's build version, or None.

    Only the version is returned, not the binary's own banner: the banner
    names the upstream project, and nothing user-facing should.
    """
    path = path or relay_path()
    if not os.path.exists(path):
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True,
                             timeout=20)
    except Exception:
        return None
    text = (out.stdout or b"").decode("utf-8", "replace")
    text += (out.stderr or b"").decode("utf-8", "replace")
    match = _VERSION_RE.search(text)
    return match.group(1) if match else "unknown"


# -- internals -----------------------------------------------------------


def _runnable(path):
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) < 1024 * 1024:
        return False
    if os.name != "nt" and not os.access(path, os.X_OK):
        return False
    return True


def _download(control):
    from .client import platform_tag

    tag = platform_tag()
    try:
        where = control.relay_download(tag)
    except Exception as exc:
        raise RelayError(
            "shayef.me could not say where to get a relay for %s." % tag,
            hint=str(getattr(exc, "message", exc)))

    url = where.get("url")
    kind = where.get("kind", "raw")
    if not url:
        raise RelayError("shayef.me returned no relay download for %s." % tag)

    target_dir = bin_dir()
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    target = relay_path()

    log.info("installing the relay connector", platform=tag)
    handle, tmp_path = tempfile.mkstemp(dir=target_dir, prefix=".relay-")
    os.close(handle)
    staging = None
    try:
        _fetch(url, tmp_path)
        if kind == "tgz":
            extracted, staging = _extract(tmp_path, target_dir)
            os.replace(extracted, target)
            os.unlink(tmp_path)
        else:
            os.replace(tmp_path, target)
        _make_executable(target)
    except BaseException:
        for leftover in (tmp_path,):
            try:
                if os.path.exists(leftover):
                    os.unlink(leftover)
            except OSError:
                pass
        raise
    finally:
        if staging and os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)

    if not _runnable(target):
        raise RelayError(
            "The relay connector at %s is not runnable." % target,
            hint="Delete %s and try again." % target_dir)

    log.info("relay ready", version=relay_version(target) or "unknown")
    return target


def _fetch(url, dest):
    request = urllib.request.Request(
        url, headers={"User-Agent": "shayef/%s" % __version__})
    try:
        response = urllib.request.urlopen(request, timeout=180)
    except urllib.error.HTTPError as exc:
        raise RelayError(
            "Downloading the relay connector failed with HTTP %s." % exc.code,
            hint="The download address shayef.me gave is not serving a build.")
    except Exception as exc:
        raise RelayError(
            "Downloading the relay connector failed: %s" % exc,
            hint=("shayef needs outbound HTTPS. On Kaggle, switch the "
                  "notebook's Internet setting on."))
    try:
        with open(dest, "wb") as fh:
            shutil.copyfileobj(response, fh, 1024 * 256)
    finally:
        response.close()


def _extract(archive, target_dir):
    """macOS builds ship as a tarball holding a single binary."""
    staging = tempfile.mkdtemp(dir=target_dir, prefix=".unpack-")
    with tarfile.open(archive, "r:gz") as tar:
        member = None
        for candidate in tar.getmembers():
            if candidate.isfile():
                member = candidate
                break
        if member is None:
            raise RelayError("The relay archive held no file.")
        member.name = os.path.basename(member.name)
        tar.extract(member, staging)
        return os.path.join(staging, member.name), staging


def _make_executable(path):
    if os.name == "nt":
        return
    try:
        mode = os.stat(path).st_mode
        os.chmod(path, mode | 0o755)
    except OSError as exc:
        raise RelayError("Could not mark %s executable: %s" % (path, exc))
