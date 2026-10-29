"""Running and babysitting the relay connector.

Responsibilities, in order of how often they matter in a notebook:

  * Parse the connector's log stream into a state machine the dashboard can
    show (starting / connected / degraded / reconnecting / failed).
  * Fall back from QUIC to HTTP/2 automatically. Colab and Kaggle often block
    outbound UDP on 7844, and the stock failure mode is a connector that
    retries forever with no explanation.
  * Restart with backoff, because a notebook session can be suspended and
    resumed and the relay has to come back on its own.
"""

import collections
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request

from .errors import TunnelError
from .log import log

STARTING = "starting"
CONNECTED = "connected"
DEGRADED = "degraded"
RECONNECTING = "reconnecting"
STOPPED = "stopped"
FAILED = "failed"

PROTOCOL_QUIC = "quic"
PROTOCOL_HTTP2 = "http2"

CONN_INDEX_RE = re.compile(r"connIndex=(\d+)")
_SUGGESTED_RE = re.compile("suggested_protocol=([A-Za-z0-9_]+)")

# Substrings that mean "the UDP path to the edge is not working".
_QUIC_SYMPTOMS = (
    "failed to dial to edge with quic",
    "quic: no recent network activity",
    "failed to create quic connection",
    "icmp proxy feature is disabled",
)
_QUIC_HINTS = ("quic", "udp")

MAX_BACKOFF = 30.0

# How long to wait for a QUIC connection before trying HTTP/2 instead.
#
# This exists because the relay does not exit when it cannot reach the edge:
# it retries internally, forever. Waiting for the process to die before
# reconsidering the transport means never reconsidering it. A sandbox that
# blocks UDP but allows TCP 443 -- which is most of them, since the client
# reached the control plane over 443 to get here -- would otherwise hang
# until the caller's timeout with no fallback ever attempted.
FALLBACK_AFTER = 15.0


class Supervisor(object):
    """Owns one relay connector child process."""

    def __init__(self, binary, target_url, token, metrics_port=None,
                 protocol=PROTOCOL_QUIC, extra_args=(), on_state=None,
                 max_restarts=None, fallback_after=FALLBACK_AFTER):
        self.binary = binary
        self.target_url = target_url
        self.token = token
        self.metrics_port = metrics_port
        self.protocol = protocol
        self.extra_args = list(extra_args)
        self.on_state = on_state
        self.fallback_after = fallback_after
        self.max_restarts = max_restarts

        self.state = STOPPED
        self.connections = set()
        self.restarts = 0
        self.last_error = None
        self.started_at = None
        self.connected_at = None

        # The relay's own output is the only evidence of why it will not
        # connect, so a window of it is kept rather than discarded.
        self.recent = collections.deque(maxlen=60)
        self.precheck = None
        self.tcp_ok = None
        self.suggested_protocol = None

        self._process = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._quic_failures = 0
        self._protocol_switched = False
        self._switch_pending = False
        self._reported_errors = 0
        self._reported_errors_lock = threading.Lock()
        self._first_connect = threading.Event()
        self._watchdog = None

    # -- lifecycle -------------------------------------------------------

    def start(self):
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._supervise,
                                        name="shayef-connector", daemon=True)
        self._thread.start()
        if self.protocol == PROTOCOL_QUIC and self.fallback_after:
            self._watchdog = threading.Thread(
                target=self._watch_for_stall, name="shayef-fallback",
                daemon=True)
            self._watchdog.start()
        return self

    def _watch_for_stall(self):
        """Fall back to HTTP/2 if QUIC has not connected in time.

        Deliberately not conditional on recognising an error string: the
        failure that matters here produced no message this code knew how to
        match, and a transport that has not connected in fifteen seconds is
        not working regardless of what it did or did not say about why.
        """
        if self._stop.wait(self.fallback_after):
            return
        if self._first_connect.is_set() or self.connections:
            return
        if self.protocol != PROTOCOL_QUIC or self._protocol_switched:
            return
        self._request_switch(
            reason="no QUIC connection after %gs" % self.fallback_after)

    def _request_switch(self, reason=None):
        """Switch transport and bounce the relay so it takes effect."""
        if self._protocol_switched:
            return
        self._switch_protocol(reason=reason)
        self._switch_pending = True
        self._terminate()

    def wait_connected(self, timeout=60):
        """Block until the connector registers at least one edge connection."""
        if self._first_connect.wait(timeout):
            return True
        if self.state == FAILED:
            raise TunnelError(
                "The relay could not connect: %s" % (self.last_error or "unknown"),
                hint=_failure_hint(self.last_error))
        return False

    def stop(self, timeout=10):
        self._stop.set()
        self._terminate()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._thread = None
        self._set_state(STOPPED)

    def restart(self):
        """Bounce the child; the supervise loop brings it straight back."""
        log.info("restarting connector")
        self._terminate()

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def describe(self):
        return {
            "state": self.state,
            "protocol": self.protocol,
            "connections": sorted(self.connections),
            "connection_count": len(self.connections),
            "restarts": self.restarts,
            "last_error": self.last_error,
            "precheck": self.precheck,
            "uptime": (time.time() - self.started_at) if self.started_at else 0,
            "connected_for": (time.time() - self.connected_at) if self.connected_at else 0,
        }

    # -- the loop --------------------------------------------------------

    def _supervise(self):
        backoff = 1.0
        while not self._stop.is_set():
            if self.max_restarts is not None and self.restarts > self.max_restarts:
                self.last_error = "gave up after %d restarts" % self.restarts
                self._set_state(FAILED)
                return

            self.connections = set()
            self._set_state(STARTING if self.restarts == 0 else RECONNECTING)
            self.started_at = self.started_at or time.time()

            code = self._run_once()
            if self._stop.is_set():
                break

            self.connected_at = None
            self.restarts += 1

            if self._switch_pending:
                # We stopped it on purpose to change transport.
                self._switch_pending = False
                self.restarts -= 1
                backoff = 1.0
                log.info("restarting the relay", protocol=self.protocol)
                continue

            if self._should_switch_protocol():
                self._switch_protocol()
                backoff = 1.0
                continue

            log.warn("relay exited, restarting",
                     code=code, attempt=self.restarts)
            self._set_state(RECONNECTING)
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, MAX_BACKOFF)

        self._set_state(STOPPED)

    def _run_once(self):
        argv = self._argv()
        log.debug("spawning the relay", argv=" ".join(_redact(argv)))
        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                errors="replace",
            )
        except OSError as exc:
            self.last_error = "could not launch the relay: %s" % exc
            self._set_state(FAILED)
            return -1

        with self._lock:
            self._process = process

        try:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                self._consume(line.rstrip("\n"))
                if self._stop.is_set():
                    break
        except Exception as exc:
            log.debug("relay log reader stopped", error=str(exc))
        finally:
            try:
                process.stdout.close()
            except Exception:
                pass

        try:
            return process.wait(timeout=10)
        except Exception:
            self._terminate()
            return -1

    def _argv(self):
        argv = [self.binary, "--no-autoupdate", "--loglevel", "info"]
        if self.metrics_port:
            argv += ["--metrics", "127.0.0.1:%d" % self.metrics_port]
        argv += ["--protocol", self.protocol]
        argv += ["tunnel", "run", "--token", self.token]
        argv += self.extra_args
        return argv

    def _terminate(self):
        with self._lock:
            process = self._process
            self._process = None
        if process is None:
            return
        try:
            process.terminate()
        except Exception:
            return
        try:
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    # -- log parsing -----------------------------------------------------

    def _consume(self, line):
        if not line.strip():
            return
        message = _message_of(line)
        lowered = message.lower()
        self.recent.append(message)

        # Order matters: "unregistered tunnel connection" contains
        # "registered tunnel connection", so the negative case is tested
        # first or every disconnect reads as a connect.
        if "unregistered tunnel connection" in lowered:
            index = CONN_INDEX_RE.search(message)
            self.connections.discard(int(index.group(1)) if index else 0)
            self._set_state(CONNECTED if self.connections else RECONNECTING)
            return

        if "registered tunnel connection" in lowered:
            index = CONN_INDEX_RE.search(message)
            self.connections.add(int(index.group(1)) if index else 0)
            if self.connected_at is None:
                self.connected_at = time.time()
            self._quic_failures = 0
            self._set_state(CONNECTED)
            self._first_connect.set()
            return

        # The relay's precheck ends by naming the transport it thinks will
        # work. Taking its word for it beats both guessing from error strings
        # and waiting out a timer -- it has just measured both paths.
        if "precheck complete" in lowered:
            match = _SUGGESTED_RE.search(message)
            if match:
                suggested = match.group(1).strip().lower()
                self.suggested_protocol = suggested
                if (suggested in (PROTOCOL_QUIC, PROTOCOL_HTTP2)
                        and suggested != self.protocol
                        and not self.connections
                        and not self._protocol_switched):
                    self._switch_protocol(
                        to=suggested,
                        reason="the relay's own precheck recommends it")
                    self._switch_pending = True
                    self._terminate()
            return

        if "tcp connectivity" in lowered and ("fail" in lowered or "pass" in lowered):
            self.tcp_ok = "pass" in lowered
            if not self.tcp_ok:
                self.last_error = (
                    "the relay's own check says TCP to the edge is blocked too")
                log.warn("TCP to the shayef edge is blocked here as well")
            return

        # The relay runs its own connectivity precheck and prints the
        # verdict. Acting on it beats waiting for dial errors that may never
        # match a known phrase.
        if "udp connectivity" in lowered and ("fail" in lowered or "pass" in lowered):
            passed = "pass" in lowered
            self.precheck = "udp ok" if passed else "udp blocked"
            if not passed:
                self._quic_failures += 1
                self.last_error = (
                    "the relay's own check says UDP to the edge is blocked")
                # Its own check already said UDP will not work, so there is
                # nothing to wait for.
                self._request_switch(
                    reason="the relay's own check says UDP is blocked")
            return

        if "summary: environment is healthy" in lowered:
            self.precheck = self.precheck or "healthy"
            return

        if any(symptom in lowered for symptom in _QUIC_SYMPTOMS):
            # Counted, not acted on. One dial failure is not proof the path is
            # unusable, and permanently downgrading the transport on a single
            # hiccup would be worse than waiting. The authoritative signals --
            # the relay's own precheck and its suggested protocol -- switch
            # straight away; this leaves the ambiguous case to the watchdog.
            self._quic_failures += 1
            self.last_error = message
            log.debug("quic symptom observed", count=self._quic_failures)
            return

        if _is_error(line, lowered):
            # A retry notice is a symptom, not the cause; letting it
            # overwrite last_error buries the dial error that explains the
            # failure.
            if "retrying connection" not in lowered:
                self.last_error = message
            if "no such host" in lowered or "lookup" in lowered:
                log.warn("relay dns problem", detail=message[:160])
            elif "token" in lowered or "unauthorized" in lowered or "401" in lowered:
                log.error("the edge rejected the relay", detail=message[:160])
                self._set_state(FAILED)
            else:
                # Visible, not debug-only: a relay that cannot connect is the
                # single most important thing to be able to explain, and
                # burying this is what made it undiagnosable.
                self._reported_errors += 1
                if self._reported_errors <= 3:
                    log.warn("relay error", detail=message[:160])
                else:
                    log.debug("relay error", detail=message[:160])

    def _should_switch_protocol(self):
        return (self.protocol == PROTOCOL_QUIC
                and not self._protocol_switched
                and not self.connections
                and (self._quic_failures > 0
                     or _looks_like_quic(self.last_error)))

    def _switch_protocol(self, to=PROTOCOL_HTTP2, reason=None):
        self.protocol = to
        self._protocol_switched = True
        self._quic_failures = 0
        log.warn("switching the relay to %s" % to,
                 reason=reason or "the UDP path to the edge looks blocked")

    def _set_state(self, state):
        if state == self.state:
            return
        previous, self.state = self.state, state
        log.debug("relay state", **{"from": previous, "to": state})
        if self.on_state:
            try:
                self.on_state(state, previous)
            except Exception:
                pass

    # -- metrics ---------------------------------------------------------

    def explain(self, lines=12):
        """Why the relay is not connected, in the relay's own words."""
        return {
            "state": self.state,
            "protocol": self.protocol,
            "precheck": self.precheck,
            "tcp_ok": self.tcp_ok,
            "suggested_protocol": self.suggested_protocol,
            "last_error": self.last_error,
            "restarts": self.restarts,
            "recent": list(self.recent)[-lines:],
        }

    def readiness(self):
        """The relay's own readiness endpoint, when metrics are enabled."""
        if not self.metrics_port:
            return None
        url = "http://127.0.0.1:%d/ready" % self.metrics_port
        try:
            resp = urllib.request.urlopen(url, timeout=2)
        except (urllib.error.URLError, OSError):
            return None
        try:
            return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
        finally:
            resp.close()


# -- helpers -------------------------------------------------------------


def _message_of(line):
    """The relay logs text by default but JSON under some builds; accept both."""
    stripped = line.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
        except ValueError:
            return line
        parts = [str(payload.get("message", ""))]
        for key in ("error", "url", "connIndex"):
            if payload.get(key):
                parts.append("%s=%s" % (key, payload[key]))
        return " ".join(p for p in parts if p)
    return line


def _is_error(line, lowered):
    return ("ERR" in line or "error" in lowered or "failed" in lowered
            or "fatal" in lowered)


def _looks_like_quic(text):
    if not text:
        return False
    lowered = str(text).lower()
    return any(hint in lowered for hint in _QUIC_HINTS)


def _redact(argv):
    """Never log a relay token."""
    out = []
    skip = False
    for item in argv:
        if skip:
            out.append("<token>")
            skip = False
            continue
        out.append(item)
        if item == "--token":
            skip = True
    return out


def _failure_hint(error):
    lowered = str(error or "").lower()
    if "token" in lowered or "unauthorized" in lowered:
        return ("The relay token was rejected. Reopen the session with "
                "`shayef up --recreate`.")
    if _looks_like_quic(lowered):
        return ("The network is blocking the shayef edge. shayef already "
                "retried over HTTP/2; if it still fails, outbound 443 is "
                "blocked too.")
    return "Run `shayef doctor` to check the relay and the network."
