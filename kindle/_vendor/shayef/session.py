"""A running tunnel: guard + relay + heartbeat, as one object.

This is what `shayef.start(...)` hands back, and what the CLI drives. It owns
the lifecycle so a notebook cell can be re-run without leaking a relay
process, a guard thread or a claimed hostname.

The chain it sets up:

    visitor -> <name>.shayef.me -> edge -> relay -> guard -> your app

The guard is shayef's, and it is in the path on purpose: it gates access and
it is the only place request traffic can actually be observed.
"""

import atexit
import re
import socket
import threading
import time
import weakref

from . import env as envmod
from . import relay as relay_mod
from .beacon import Beacon
from .client import ControlPlane
from .config import Config
from .errors import ConfigError, ShayefError
from .guard import MODE_KEY, MODE_OPEN, Guard, new_key
from .log import log
from .metrics import Metrics
from .supervisor import CONNECTED, FAILED, PROTOCOL_HTTP2, PROTOCOL_QUIC, Supervisor

ACCESS_KEY = "key"
ACCESS_OPEN = "open"

_live = weakref.WeakSet()
_live_lock = threading.Lock()


class Tunnel(object):
    def __init__(self, port, name, access=ACCESS_KEY, key=None, config=None,
                 token=None, api=None, host="127.0.0.1", guard=True,
                 recreate=False, label=None, share_key=False,
                 beat_interval=5.0, app_label=None):
        self.app_port = int(port)
        self.app_host = host
        self.name = _check_name(name)
        self.access = _normalize_access(access)
        self.app_label = app_label or label or ("the app on port %d" % self.app_port)
        self.label = label
        self.use_guard = bool(guard)
        self.recreate = recreate
        self.share_key = bool(share_key)
        self.beat_interval = beat_interval
        self.environment = envmod.detect()

        self.config = config or Config.load(api_token=token, api_base=api)
        self.config.require_token()
        self.control = ControlPlane(self.config.api_base, self.config.api_token)

        self.metrics = Metrics()
        self.guard = None
        self.supervisor = None
        self.beacon = None
        self.session = None

        self.started_at = None
        self.stopped = False
        self._explicit_key = key
        self._stop_lock = threading.Lock()

    # -- properties ------------------------------------------------------

    @property
    def url(self):
        return self.session.url if self.session else None

    @property
    def hostname(self):
        return self.session.hostname if self.session else None

    @property
    def key(self):
        return self.guard.key if self.guard else None

    @property
    def status(self):
        if self.stopped:
            return "stopped"
        return self.supervisor.state if self.supervisor else "stopped"

    @property
    def unlock_url(self):
        """The link to actually hand to a person: address plus key."""
        if not self.url:
            return None
        if self.guard and self.guard.mode == MODE_KEY:
            return "%s/?k=%s" % (self.url, self.guard.key)
        return self.url

    @property
    def dashboard_url(self):
        return self.config.api_base + "/"

    @property
    def target(self):
        return "%s:%d" % (self.app_host, self.app_port)

    # -- lifecycle -------------------------------------------------------

    def start(self, wait=True, timeout=75):
        if self.started_at:
            return self
        self.started_at = time.time()
        with _live_lock:
            _live.add(self)

        log.info("opening session", name=self.name, target=self.target,
                 host=self.environment.kind)
        for note in self.environment.notes:
            log.warn(note)

        if not _port_open(self.app_host, self.app_port):
            log.warn("nothing is listening on %s yet" % self.target,
                     detail="the tunnel will serve a holding page until it is")

        service_port = self._start_guard()
        service_url = "http://127.0.0.1:%d" % service_port

        # The control plane owns the hostname; it hands back a relay token.
        self.session = self.control.open_session(
            self.name, service_url, label=self.label,
            unlock_key=self.guard.key if (self.share_key and self.guard) else None,
            recreate=self.recreate)
        log.info("hostname ready", url=self.session.url,
                 reused=self.session.reused)

        binary = relay_mod.ensure_relay(control=self.control)

        self.supervisor = Supervisor(
            binary=binary,
            target_url=service_url,
            token=self.session.relay_token,
            metrics_port=_free_port(),
            protocol=self._protocol(),
            on_state=self._on_relay_state,
        ).start()

        self.beacon = Beacon(
            self.control, self.name, self.session.session_token,
            self.metrics, self.supervisor, guard=self.guard,
            interval=self.beat_interval, share_key=self.share_key,
            host=self.environment.kind,
        ).start()

        if wait:
            self._wait_for_connection(timeout=timeout)

        self._banner()
        return self

    def _protocol(self):
        if self.environment.is_kaggle:
            # Kaggle's egress blocks UDP 7844 often enough that starting on
            # HTTP/2 saves everyone a minute of confused retrying.
            return PROTOCOL_HTTP2
        return PROTOCOL_QUIC

    def _start_guard(self):
        if not self.use_guard:
            log.warn("guard disabled: no access control and no request log")
            return self.app_port
        mode = MODE_KEY if self.access == ACCESS_KEY else MODE_OPEN
        key = self._explicit_key or new_key()
        self.guard = Guard(target_port=self.app_port, target_host=self.app_host,
                           mode=mode, key=key, metrics=self.metrics,
                           app_label=self.app_label).start()
        return self.guard.port

    def _wait_for_connection(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.supervisor.state == FAILED:
                log.error("the relay failed to come up",
                          detail=self.supervisor.last_error or "unknown")
                self._explain_failure()
                return False
            if self.supervisor.state == CONNECTED:
                return True
            time.sleep(0.25)
        log.warn("the relay has not connected after %ss" % timeout)
        self._explain_failure()
        return False

    def _explain_failure(self):
        """Say why, in the relay's own words.

        A hostname with no connector behind it serves a 1033 from the edge,
        which tells a visitor nothing. Printing the relay's own account of
        the failure here is the difference between a diagnosable problem and
        a mystery.
        """
        detail = self.supervisor.explain()
        print("")
        print("    The address exists but nothing is behind it yet, so it will")
        print("    return an edge error until the relay connects.")
        print("")
        print("      state     %s" % detail["state"])
        print("      protocol  %s" % detail["protocol"])
        if detail["precheck"]:
            print("      network   %s" % detail["precheck"])
        if detail["last_error"]:
            print("      error     %s" % str(detail["last_error"])[:150])
        if detail["recent"]:
            print("")
            print("    Last lines from the relay:")
            for line in detail["recent"]:
                print("      %s" % str(line)[:160])
        print("")
        if detail["precheck"] == "udp blocked":
            print("    UDP to the edge is blocked here. shayef has already retried")
            print("    over HTTP/2; if it still will not connect, outbound 443 is")
            print("    blocked too and this network cannot carry a tunnel.")
        else:
            print("    Most often this is a sandbox blocking outbound traffic.")
            print("    Run `shayef doctor`, and check the host allows outbound")
            print("    UDP 7844 or TCP 443 to the edge.")
        print("")

    def stop(self, release=False):
        """Stop serving. `release` also gives up the hostname."""
        with self._stop_lock:
            if self.stopped:
                return self
            self.stopped = True

        log.info("closing session", name=self.name)
        if self.beacon:
            try:
                self.beacon.stop(final=True)
            except Exception:
                pass
        if self.supervisor:
            try:
                self.supervisor.stop()
            except Exception:
                pass
        if self.guard:
            try:
                self.guard.stop()
            except Exception:
                pass
        if release:
            try:
                result = self.control.close_session(self.name)
                log.info("hostname released",
                         removed=",".join(result.get("removed") or []) or "nothing")
            except ShayefError as exc:
                log.warn("could not release the hostname", detail=exc.message)

        with _live_lock:
            _live.discard(self)
        log.info("session closed")
        return self

    def release(self):
        """Stop and give up the hostname entirely."""
        return self.stop(release=True)

    def restart_relay(self):
        if self.supervisor:
            self.supervisor.restart()
        return self

    def rotate_key(self):
        if self.guard is None:
            raise ShayefError("There is no guard, so there is no key to rotate.")
        if self.guard.mode != MODE_KEY:
            self.guard.mode = MODE_KEY
        return self.guard.rotate_key()

    def wait(self, timeout=None):
        """Block while the tunnel runs. Ctrl-C or kernel interrupt stops it."""
        start = time.time()
        try:
            while not self.stopped:
                if timeout is not None and time.time() - start > timeout:
                    return self
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("")
            self.stop()
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def __repr__(self):
        return "<Tunnel %s %s -> %s>" % (self.status, self.url or "(no url)",
                                         self.target)

    # -- local view ------------------------------------------------------

    def snapshot(self):
        """Everything shayef knows locally. The dashboard shows the control
        plane's view of this, which is this plus staleness."""
        relay = self.supervisor.describe() if self.supervisor else {
            "state": "stopped", "connection_count": 0}
        guard = self.guard.describe() if self.guard else {
            "mode": "off", "protected": False}
        return {
            "name": self.name,
            "url": self.url,
            "hostname": self.hostname,
            "target": self.target,
            "access": self.access,
            "stopped": self.stopped,
            "relay": relay,
            "relay_detail": (self.supervisor.explain()
                             if self.supervisor and self.status != "connected"
                             else None),
            "guard": guard,
            "beacon": self.beacon.describe() if self.beacon else None,
            "metrics": self.metrics.summary(),
            "series": self.metrics.series(90),
            "environment": self.environment.describe(),
            "dashboard": self.dashboard_url,
        }

    def dashboard(self, inline=True, height=760):
        """Show where the dashboard is. It is served by shayef.me, so in a
        notebook this is a link or an iframe, not a local server."""
        from . import notebook
        return notebook.show_dashboard(self, inline=inline, height=height)

    # -- callbacks -------------------------------------------------------

    def _on_relay_state(self, state, previous):
        if state == CONNECTED and previous != CONNECTED and self.url:
            log.info("tunnel is live", url=self.url)

    # -- output ----------------------------------------------------------

    def _banner(self):
        lines = [("Public", self.url or "not assigned (%s)" % self.status),
                 ("Forwards to", "http://%s" % self.target)]
        if self.guard and self.guard.mode == MODE_KEY:
            lines.append(("Key", self.guard.key))
            if self.url:
                lines.append(("Unlock link", self.unlock_url))
        else:
            lines.append(("Access", "OPEN - anyone with the address"))
        lines.append(("Dashboard", self.dashboard_url))

        width = max(len(label) for label, _ in lines)
        print("")
        print("  shayef.me  ---  %s  ---  %s" % (self.name, self.status))
        print("")
        for label, value in lines:
            print("    %s  %s" % (label.rjust(width), value))
        print("")
        if self.access == ACCESS_OPEN:
            print("    ! This address is not gated. Anyone who finds it "
                  "reaches your app.")
            print("")


# -- module level helpers -------------------------------------------------


def start(port, name, wait=True, timeout=75, **kwargs):
    """Open a tunnel to a local port. See Tunnel for the options."""
    return Tunnel(port, name, **kwargs).start(wait=wait, timeout=timeout)


def active():
    with _live_lock:
        return [t for t in _live if not t.stopped]


def stop_all():
    for tunnel in active():
        try:
            tunnel.stop()
        except Exception:
            pass


def _check_name(name):
    value = str(name or "").strip().lower()
    if not value:
        raise ConfigError(
            "A name is required; it becomes the hostname.",
            hint="For example name='chat' serves at chat.shayef.me.")
    if not re.match(r"^[a-z0-9]([a-z0-9-]{0,48}[a-z0-9])?$", value):
        raise ConfigError(
            "%r cannot be used as a hostname." % name,
            hint=("Lowercase letters, digits and hyphens, starting and ending "
                  "with a letter or digit. For example: chat, llama-3, demo2."))
    return value


def _normalize_access(access):
    value = str(access or ACCESS_KEY).lower()
    if value in ("none", "public", "off"):
        value = ACCESS_OPEN
    if value not in (ACCESS_KEY, ACCESS_OPEN):
        raise ConfigError(
            "Unknown access mode %r." % access,
            hint="Use key (the default) or open.")
    return value


def _free_port():
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _port_open(host, port, timeout=0.4):
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((host, int(port)))
        return True
    except Exception:
        return False
    finally:
        sock.close()


atexit.register(stop_all)
