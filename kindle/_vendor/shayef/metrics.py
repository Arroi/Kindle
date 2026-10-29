"""Traffic bookkeeping.

The guard records every request here; the dashboard reads it. Kept
deliberately small and lock-guarded rather than clever, because it is written
from the guard's event loop thread and read from the dashboard's HTTP threads.
"""

import collections
import threading
import time

BUCKET_SECONDS = 1
BUCKET_COUNT = 120


class Metrics(object):
    def __init__(self, capacity=300):
        self._lock = threading.Lock()
        self._requests = collections.deque(maxlen=capacity)
        self._buckets = collections.deque(maxlen=BUCKET_COUNT)
        self._seq = 0
        self.started_at = time.time()
        self.total = 0
        self.denied = 0
        self.errors = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self._latencies = collections.deque(maxlen=capacity)

    # -- writing ---------------------------------------------------------

    def record(self, method, path, status, duration_ms, bytes_in=0,
               bytes_out=0, denied=False, note=None, client=None):
        now = time.time()
        with self._lock:
            self._seq += 1
            entry = {
                "seq": self._seq,
                "ts": now,
                "method": method,
                "path": path,
                "status": status,
                "ms": round(duration_ms, 1),
                "bytes_in": bytes_in,
                "bytes_out": bytes_out,
                "denied": bool(denied),
                "note": note,
                "client": client,
            }
            self._requests.append(entry)
            self.total += 1
            self.bytes_in += bytes_in
            self.bytes_out += bytes_out
            if denied:
                self.denied += 1
            if isinstance(status, int) and status >= 500:
                self.errors += 1
            if duration_ms:
                self._latencies.append(duration_ms)
            self._bump_bucket(now, denied=denied, status=status)
        return entry

    def add_traffic(self, bytes_in=0, bytes_out=0):
        """Connection-level byte totals.

        Proxied requests share one socket, so bytes cannot be split per
        request without reframing the stream -- which would cost the
        streaming fidelity chat UIs depend on. Totals are counted here
        instead, and per-request byte fields stay exact only for responses
        the guard writes itself.
        """
        with self._lock:
            self.bytes_in += bytes_in
            self.bytes_out += bytes_out

    def _bump_bucket(self, now, denied=False, status=None):
        slot = int(now // BUCKET_SECONDS)
        if self._buckets and self._buckets[-1]["slot"] == slot:
            bucket = self._buckets[-1]
        else:
            bucket = {"slot": slot, "count": 0, "denied": 0, "error": 0}
            self._buckets.append(bucket)
        bucket["count"] += 1
        if denied:
            bucket["denied"] += 1
        if isinstance(status, int) and status >= 500:
            bucket["error"] += 1

    # -- reading ---------------------------------------------------------

    def recent(self, since=0, limit=100):
        with self._lock:
            items = [r for r in self._requests if r["seq"] > since]
        return items[-limit:]

    def series(self, seconds=90):
        """A dense per-second series, zero-filled, oldest first. Zero-filling
        here keeps the dashboard sparkline honest about idle gaps."""
        now = int(time.time() // BUCKET_SECONDS)
        with self._lock:
            by_slot = dict((b["slot"], b) for b in self._buckets)
        out = []
        for slot in range(now - seconds + 1, now + 1):
            bucket = by_slot.get(slot)
            out.append(bucket["count"] if bucket else 0)
        return out

    def summary(self):
        with self._lock:
            latencies = sorted(self._latencies)
            total = self.total
            denied = self.denied
            errors = self.errors
            bytes_in = self.bytes_in
            bytes_out = self.bytes_out
            started = self.started_at
        return {
            "total": total,
            "denied": denied,
            "errors": errors,
            "bytes_in": bytes_in,
            "bytes_out": bytes_out,
            "uptime": max(0.0, time.time() - started),
            "p50_ms": _percentile(latencies, 50),
            "p95_ms": _percentile(latencies, 95),
        }

    def reset(self):
        with self._lock:
            self._requests.clear()
            self._buckets.clear()
            self._latencies.clear()
            self.total = 0
            self.denied = 0
            self.errors = 0
            self.bytes_in = 0
            self.bytes_out = 0
            self.started_at = time.time()


def _percentile(sorted_values, pct):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return round(sorted_values[0], 1)
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    value = sorted_values[low] * (1 - weight) + sorted_values[high] * weight
    return round(value, 1)
