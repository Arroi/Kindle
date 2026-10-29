"""The heartbeat.

The dashboard lives in the control plane, but the facts it shows -- which
requests arrived, how fast, how many were turned away -- are only observable
next to the app. This pushes them up on a timer.

Failures are deliberately quiet. A tunnel that is working must not be taken
down, or spam a notebook with tracebacks, because telemetry could not be
delivered. The dashboard shows a session as stale instead, which is the
honest reading: shayef.me genuinely does not know how it is doing.
"""

import threading
import time

from .log import log

DEFAULT_INTERVAL = 5.0
MAX_ROWS_PER_BEAT = 50


class Beacon(object):
    def __init__(self, control, name, session_token, metrics, supervisor,
                 guard=None, interval=DEFAULT_INTERVAL, share_key=False,
                 host=None):
        self.control = control
        self.name = name
        self.session_token = session_token
        self.metrics = metrics
        self.supervisor = supervisor
        self.guard = guard
        self.interval = max(1.0, float(interval))
        self.share_key = bool(share_key)
        self.host = host

        self.sent = 0
        self.failures = 0
        self.last_sent_at = None
        self.last_error = None

        self._cursor = 0
        self._stop = threading.Event()
        self._thread = None

    # -- lifecycle -------------------------------------------------------

    def start(self):
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="shayef-beacon",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self, final=True):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 2)
        self._thread = None
        if final:
            # One last beat so the dashboard shows "stopped" rather than
            # waiting for the session to go stale on its own.
            try:
                self._send(state_override="stopped")
            except Exception:
                pass

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def describe(self):
        return {
            "interval": self.interval,
            "sent": self.sent,
            "failures": self.failures,
            "last_sent_at": self.last_sent_at,
            "last_error": self.last_error,
        }

    # -- the loop --------------------------------------------------------

    def _loop(self):
        # Beat once straight away so the dashboard does not sit empty for the
        # first interval.
        while True:
            try:
                self._send()
            except Exception as exc:
                self.failures += 1
                self.last_error = str(exc)
                if self.failures in (1, 10) or self.failures % 50 == 0:
                    log.debug("could not report telemetry", error=str(exc)[:120])
            if self._stop.wait(self.interval):
                return

    def _send(self, state_override=None):
        telemetry = self._telemetry(state_override)
        rows = self._rows()
        unlock_key = None
        if self.share_key and self.guard is not None:
            unlock_key = self.guard.key

        self.control.beat(self.name, self.session_token, telemetry,
                          requests=rows, unlock_key=unlock_key)
        self.sent += 1
        self.last_sent_at = time.time()
        self.last_error = None
        if rows:
            self._cursor = rows[-1]["_seq"]
            for row in rows:
                row.pop("_seq", None)

    def _telemetry(self, state_override=None):
        summary = self.metrics.summary()
        connector = self.supervisor.describe() if self.supervisor else {}
        return {
            "state": state_override or connector.get("state", "unknown"),
            "connections": connector.get("connection_count", 0),
            "protocol": connector.get("protocol"),
            "uptime": connector.get("uptime", summary.get("uptime", 0)),
            "requests": summary.get("total", 0),
            "blocked": summary.get("denied", 0),
            "errors": summary.get("errors", 0),
            "p50_ms": summary.get("p50_ms"),
            "p95_ms": summary.get("p95_ms"),
            "series": self.metrics.series(90),
            "host": self.host,
        }

    def _rows(self):
        fresh = self.metrics.recent(since=self._cursor, limit=MAX_ROWS_PER_BEAT)
        out = []
        for row in fresh:
            out.append({
                "_seq": row["seq"],
                "ts": row["ts"],
                "method": row["method"],
                "path": row["path"],
                "status": row["status"],
                "ms": row["ms"],
                "blocked": bool(row["denied"]),
                "note": row.get("note"),
            })
        return out
