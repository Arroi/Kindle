"""A structured event log that is both printed and kept in memory.

The dashboard reads the in-memory ring buffer, so anything worth showing a
user has to go through here rather than straight to print().
"""

import collections
import sys
import threading
import time

LEVELS = ("debug", "info", "warn", "error")

_QUIET_PREFIX = "  "
_COLOR = {
    "debug": "\033[90m",
    "info": "\033[36m",
    "warn": "\033[33m",
    "error": "\033[31m",
}
_RESET = "\033[0m"
_DIM = "\033[90m"


class EventLog:
    def __init__(self, capacity=500):
        self._events = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = 0
        self.min_level = "info"
        self.use_color = _supports_color()
        self.echo = True

    def _enabled(self, level):
        return LEVELS.index(level) >= LEVELS.index(self.min_level)

    def emit(self, level, message, **fields):
        if level not in LEVELS:
            raise ValueError("unknown level: %r" % level)
        with self._lock:
            self._seq += 1
            event = {
                "seq": self._seq,
                "ts": time.time(),
                "level": level,
                "message": message,
                "fields": fields,
            }
            self._events.append(event)
        if self.echo and self._enabled(level):
            self._print(event)
        return event

    def _print(self, event):
        level = event["level"]
        tag = {"debug": "    ", "info": "  - ", "warn": "  ! ", "error": "  x "}[level]
        extra = ""
        if event["fields"]:
            extra = " " + " ".join(
                "%s=%s" % (k, v) for k, v in sorted(event["fields"].items())
            )
        line = "%s%s%s" % (tag, event["message"], extra)
        if self.use_color:
            color = _COLOR[level]
            line = "%s%s%s%s%s" % (color, tag, _RESET, event["message"],
                                   _DIM + extra + _RESET if extra else "")
        stream = sys.stderr if level in ("warn", "error") else sys.stdout
        try:
            stream.write(line + "\n")
            stream.flush()
        except Exception:
            pass

    def debug(self, message, **f):
        return self.emit("debug", message, **f)

    def info(self, message, **f):
        return self.emit("info", message, **f)

    def warn(self, message, **f):
        return self.emit("warn", message, **f)

    def error(self, message, **f):
        return self.emit("error", message, **f)

    def since(self, seq=0, limit=200):
        with self._lock:
            return [e for e in self._events if e["seq"] > seq][-limit:]

    def all(self):
        with self._lock:
            return list(self._events)


def _supports_color():
    if not hasattr(sys.stdout, "isatty"):
        return False
    try:
        if sys.stdout.isatty():
            return True
    except Exception:
        return False
    return False


log = EventLog()
