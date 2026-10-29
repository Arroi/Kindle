"""The shayef.me control-plane client.

The notebook side knows nothing about how traffic is carried. It asks
shayef.me to open a session and gets back a hostname and a relay token; the
credential that provisions any of it never leaves the control plane.

Built on urllib so `pip install shayef` pulls nothing into a notebook kernel.
"""

import json
import platform
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

from .errors import ControlPlaneError
from .version import __version__

USER_AGENT = "shayef/%s" % __version__
RETRY_STATUS = (429, 500, 502, 503, 504)


class Session(object):
    """What the control plane hands back when a session is opened."""

    def __init__(self, name, hostname, url, relay_token, session_token,
                 reused=False):
        self.name = name
        self.hostname = hostname
        self.url = url
        self.relay_token = relay_token
        self.session_token = session_token
        self.reused = reused

    def describe(self):
        return {"name": self.name, "hostname": self.hostname, "url": self.url,
                "reused": self.reused}

    def __repr__(self):
        return "<Session %s>" % self.hostname


class ControlPlane(object):
    def __init__(self, base, token, timeout=30, retries=3):
        self.base = (base or "").rstrip("/")
        self.token = token
        self.timeout = timeout
        self.retries = retries

    # -- transport -------------------------------------------------------

    def request(self, method, path, body=None, token=None, params=None,
                retries=None):
        url = self.base + path
        if params:
            clean = dict((k, v) for k, v in params.items() if v is not None)
            if clean:
                url += "?" + urllib.parse.urlencode(clean)

        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")

        attempts = self.retries if retries is None else retries
        last = None
        for attempt in range(max(1, attempts)):
            request = urllib.request.Request(url, data=payload, method=method)
            request.add_header("Authorization", "Bearer " + (token or self.token))
            request.add_header("Content-Type", "application/json")
            request.add_header("Accept", "application/json")
            request.add_header("User-Agent", USER_AGENT)
            try:
                response = urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                raw = exc.read()
                if exc.code in RETRY_STATUS and attempt < attempts - 1:
                    last = exc
                    time.sleep(_backoff(attempt))
                    continue
                raise _error(exc.code, raw, method, path)
            except (urllib.error.URLError, socket.timeout) as exc:
                last = exc
                if attempt < attempts - 1:
                    time.sleep(_backoff(attempt))
                    continue
                raise ControlPlaneError(
                    "Could not reach %s (%s)." % (self.base, _reason(exc)),
                    hint=("Check the address and network access. On Kaggle "
                          "the notebook's Internet setting has to be on."))
            try:
                return _decode(response.read())
            finally:
                response.close()

        raise ControlPlaneError("Request to %s failed: %s"
                                % (self.base, _reason(last)))

    # -- endpoints -------------------------------------------------------

    def health(self):
        return self.request("GET", "/v1/health")

    def open_session(self, name, service, label=None, unlock_key=None,
                     recreate=False):
        payload = {"name": name, "service": service, "recreate": bool(recreate)}
        if label:
            payload["label"] = label
        if unlock_key:
            payload["unlock_key"] = unlock_key
        result = self.request("POST", "/v1/sessions", body=payload)
        return Session(
            name=name,
            hostname=result["hostname"],
            url=result["url"],
            relay_token=result["relay_token"],
            session_token=result["session_token"],
            reused=bool(result.get("reused")),
        )

    def close_session(self, name):
        return self.request(
            "DELETE", "/v1/sessions/%s" % urllib.parse.quote(name, safe=""))

    def list_sessions(self):
        return self.request("GET", "/v1/sessions").get("sessions", [])

    def get_session(self, name):
        return self.request(
            "GET", "/v1/sessions/%s" % urllib.parse.quote(name, safe="")).get("session")

    def relay_download(self, platform_tag):
        """Where to fetch the relay connector for this machine."""
        return self.request("GET", "/v1/relay",
                            params={"platform": platform_tag})

    def beat(self, name, session_token, telemetry, requests=None,
             unlock_key=None):
        """Push telemetry. Never retried: a stale beat is worth nothing, and
        the next one is a second away."""
        payload = {"telemetry": telemetry}
        if requests:
            payload["requests"] = requests
        if unlock_key is not None:
            payload["unlock_key"] = unlock_key
        return self.request(
            "POST", "/v1/sessions/%s/beat" % urllib.parse.quote(name, safe=""),
            body=payload, token=session_token, retries=1)

    # -- admin -----------------------------------------------------------

    def issue_token(self, kind="client", label=None):
        return self.request("POST", "/v1/tokens",
                            body={"kind": kind, "label": label})

    def list_tokens(self):
        return self.request("GET", "/v1/tokens").get("tokens", [])

    def revoke_token(self, token_id):
        return self.request(
            "DELETE", "/v1/tokens/%s" % urllib.parse.quote(token_id, safe=""))

    def __repr__(self):
        return "<ControlPlane %s>" % self.base


# -- helpers -------------------------------------------------------------


def platform_tag():
    """The platform string the control plane uses to pick a relay build."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "amd64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64"
    elif machine in ("armv7l", "armv6l", "arm"):
        arch = "arm"
    elif machine in ("i386", "i686", "x86"):
        arch = "386"
    else:
        arch = machine or "unknown"
    return "%s-%s" % (system, arch)


def _decode(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError:
        raise ControlPlaneError(
            "shayef.me returned something that is not JSON.",
            hint="Check that SHAYEF_API points at the control plane.")


def _error(status, raw, method, path):
    body = {}
    try:
        body = json.loads(raw.decode("utf-8")) if raw else {}
    except ValueError:
        body = {}
    message = body.get("error") or "shayef.me rejected %s %s (HTTP %s)." % (
        method, path, status)
    hint = body.get("hint")
    if status in (401, 403) and not hint:
        hint = ("The token was rejected. Check SHAYEF_TOKEN, or ask for a new "
                "one.")
    if status == 409 and not hint:
        hint = "Pick a different name, or close the existing session first."
    return ControlPlaneError(message, status=status, hint=hint)


def _backoff(attempt):
    return min(2 ** attempt * 0.5, 4.0)


def _reason(exc):
    if exc is None:
        return "unknown"
    return str(getattr(exc, "reason", None) or exc)
