"""Config and credential storage.

A client holds one thing: a shayef token. Everything about hostnames, DNS and
the edge lives in the control plane, so there is nothing else here to leak.

Resolution order, most specific first: explicit keyword arguments, environment
variables, then ~/.shayef/config.json. Notebook users usually live on the
environment-variable path (a secrets cell) and terminal users on the file.
"""

import json
import os
import stat

from . import env as envmod
from .errors import ConfigError

CONFIG_NAME = "config.json"

DEFAULT_API = "https://shayef.me"

_ENV_KEYS = {
    "api_token": ("SHAYEF_TOKEN",),
    "api_base": ("SHAYEF_API",),
}

_SECRET_FIELDS = ("api_token",)


class Config(object):
    fields = ("api_token", "api_base")

    def __init__(self, api_token=None, api_base=None, path=None):
        self.api_token = api_token
        self.api_base = (api_base or DEFAULT_API).rstrip("/")
        self.path = path

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path=None, **overrides):
        path = path or default_path()
        data = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as fh:
                    data = json.load(fh) or {}
            except ValueError as exc:
                raise ConfigError(
                    "%s is not valid JSON (%s)." % (path, exc),
                    hint="Delete the file and run `shayef login` again.")

        for key in cls.fields:
            for var in _ENV_KEYS.get(key, ()):
                value = os.environ.get(var)
                if value:
                    data[key] = value
                    break

        for key, value in overrides.items():
            if value is not None:
                data[key] = value

        known = dict((k, v) for k, v in data.items() if k in cls.fields)
        return cls(path=path, **known)

    # -- persistence -----------------------------------------------------

    def save(self, path=None):
        path = path or self.path or default_path()
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        payload = {}
        for key in self.fields:
            value = getattr(self, key)
            if value:
                payload[key] = value
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        _lock_down(tmp)
        os.replace(tmp, path)
        self.path = path
        return path

    # -- checks ----------------------------------------------------------

    @property
    def has_token(self):
        return bool(self.api_token)

    def require_token(self):
        if not self.api_token:
            raise ConfigError(
                "No shayef token configured.",
                hint=("Run `shayef login --token <TOKEN>`, or set "
                      "SHAYEF_TOKEN. Ask whoever runs shayef.me for a token; "
                      "they can issue one with `shayef tokens new`."))
        return True

    def redacted(self):
        out = {}
        for key in self.fields:
            value = getattr(self, key)
            if key in _SECRET_FIELDS and value:
                value = _mask(value)
            out[key] = value
        out["path"] = self.path
        return out

    def __repr__(self):
        return "<Config api=%s token=%s>" % (self.api_base, self.has_token)


def default_path():
    return os.path.join(envmod.home_dir(), CONFIG_NAME)


def _mask(value):
    value = str(value)
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def _lock_down(path):
    """Best effort 0600. On Windows this is a no-op and we do not pretend
    otherwise."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass
