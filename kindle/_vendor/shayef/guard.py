"""The guard: a localhost reverse proxy in front of the app being tunnelled.

The relay points at the guard instead of at the app directly, which buys
two things that matter for a chat UI on a public hostname:

  1. Access control. A notebook chat UI has no login of its own, and leaving
     one reachable by anyone who guesses the hostname is not an acceptable
     default. The guard requires a key before any byte reaches the app.
  2. Visibility. Every request is recorded, which is what the dashboard shows.

Why the two directions are handled differently
----------------------------------------------

The *request* direction is framed properly: each request head is parsed, its
key checked, and its body forwarded according to Content-Length or chunked
encoding before the next request is read. Every request is authorized, not
just the first on a connection.

That matters specifically because the relay sits in front. A relay pools
keep-alive connections to the origin and reuses them across unrelated
visitors, so "this socket already presented a valid key" says nothing about
who is sending the current request. Deciding once per socket would let an
unauthenticated visitor ride a connection somebody else unlocked, and would
leave a rotated key still working.

The *response* direction is a raw byte pipe for the life of the connection.
Responses are never parsed, buffered or reframed, which is what keeps SSE,
chunked output and streamed tokens behaving exactly as the app intended.
HTTP/1.1 responses come back in request order, so piping them verbatim stays
correct however far along the request loop is.

On `Upgrade` (WebSocket), both directions become raw pipes after the handshake
is forwarded, because the bytes are no longer HTTP messages.
"""

import asyncio
import collections
import secrets
import threading
import time
import urllib.parse

from .log import log
from .metrics import Metrics

COOKIE_NAME = "shayef_auth"
KEY_PARAMS = ("shayef_key", "k")
HEAD_LIMIT = 256 * 1024
PIPE_CHUNK = 64 * 1024

MODE_KEY = "key"
MODE_OPEN = "open"


def new_key(nbytes=16):
    """A URL-safe key short enough to paste into a phone."""
    return secrets.token_urlsafe(nbytes)


class Guard(object):
    def __init__(self, target_port, target_host="127.0.0.1", listen_port=0,
                 listen_host="127.0.0.1", mode=MODE_KEY, key=None,
                 metrics=None, app_label=None):
        self.target_port = int(target_port)
        self.target_host = target_host
        self.listen_host = listen_host
        self.requested_port = int(listen_port)
        self.port = None
        self.mode = mode
        self.key = key or (new_key() if mode == MODE_KEY else None)
        self.metrics = metrics or Metrics()
        self.app_label = app_label or "your app"

        self._loop = None
        self._server = None
        self._thread = None
        self._ready = threading.Event()
        self._error = None

    # -- lifecycle -------------------------------------------------------

    def start(self, timeout=10):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="shayef-guard",
                                        daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("The guard proxy did not start within %ss." % timeout)
        if self._error:
            raise self._error
        log.debug("guard listening", port=self.port, mode=self.mode)
        return self

    def stop(self):
        loop, self._loop = self._loop, None
        if loop is None:
            return
        # Cancel in-flight connections before stopping the loop, otherwise
        # asyncio prints "Task was destroyed but it is pending" into the
        # notebook output, which looks like a crash and is not one.
        try:
            future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
            future.result(timeout=5)
        except Exception:
            pass
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None

    async def _shutdown(self):
        server, self._server = self._server, None
        if server is not None:
            server.close()
            try:
                await server.wait_closed()
            except Exception:
                pass
        current = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks() if task is not current]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def rotate_key(self):
        self.key = new_key()
        log.info("guard key rotated")
        return self.key

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def describe(self):
        return {
            "mode": self.mode,
            "port": self.port,
            "target": "%s:%s" % (self.target_host, self.target_port),
            "protected": self.mode == MODE_KEY,
        }

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            coro = asyncio.start_server(
                self._handle, host=self.listen_host, port=self.requested_port,
                limit=HEAD_LIMIT)
            self._server = loop.run_until_complete(coro)
            self.port = self._server.sockets[0].getsockname()[1]
            self._ready.set()
            loop.run_forever()
        except Exception as exc:  # surfaced to start()
            self._error = exc
            self._ready.set()
        finally:
            try:
                if self._server is not None:
                    self._server.close()
                    loop.run_until_complete(self._server.wait_closed())
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    # -- connection handling ---------------------------------------------

    async def _handle(self, client_reader, client_writer):
        peer = _peer_of(client_writer)
        upstream_reader = None
        upstream_writer = None
        response_pump = None
        conn = None

        try:
            while True:
                try:
                    head = await client_reader.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError,
                        ConnectionResetError, BrokenPipeError):
                    return
                if not head:
                    return

                started = time.time()
                request = _parse_head(head)

                refusal = self._authorize(request)
                if refusal is not None:
                    body, status, note = refusal
                    # A proper page is only safe on the first request of a
                    # connection; later on, an upstream response may be in
                    # flight on the same socket, so close instead and let the
                    # client retry on a fresh connection, where it gets the
                    # page.
                    if upstream_writer is None:
                        client_writer.write(body)
                        await client_writer.drain()
                    self.metrics.record(
                        request.method, request.path, status,
                        (time.time() - started) * 1000.0,
                        bytes_in=len(head),
                        bytes_out=len(body) if upstream_writer is None else 0,
                        denied=status in (401, 403), note=note, client=peer)
                    return

                if upstream_writer is None:
                    try:
                        upstream_reader, upstream_writer = \
                            await asyncio.open_connection(self.target_host,
                                                          self.target_port)
                    except (ConnectionRefusedError, OSError):
                        body = _not_up_page(self.app_label, self.target_port)
                        client_writer.write(body)
                        await client_writer.drain()
                        self.metrics.record(
                            request.method, request.path, 502,
                            (time.time() - started) * 1000.0,
                            bytes_in=len(head), bytes_out=len(body),
                            note="app not listening on port %s"
                                 % self.target_port,
                            client=peer)
                        return
                    conn = _Connection(self.metrics, peer)
                    response_pump = asyncio.ensure_future(
                        _pump(upstream_reader, client_writer,
                              conn.saw_upstream_bytes))

                conn.expect(request.method, request.path, started)
                conn.bytes_in += len(head)
                upstream_writer.write(head)

                upgraded = await _forward_body(request, client_reader,
                                               upstream_writer, conn)
                await upstream_writer.drain()

                if upgraded:
                    # No more HTTP messages on this socket; pipe it raw. The
                    # `upgraded` flag is left to the response pump, which sets
                    # it when it sees the 101 -- setting it here would race the
                    # pump and lose the 101 from the request log.
                    await _pump(client_reader, upstream_writer,
                                conn.saw_client_bytes)
                    return

        except asyncio.CancelledError:
            raise
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        except Exception as exc:
            log.debug("guard connection error", error=str(exc))
        finally:
            # Every write was drained as it was made, so cancelling the pump
            # loses nothing still owed to the client; closing the transport
            # flushes whatever it already holds.
            if response_pump is not None and not response_pump.done():
                response_pump.cancel()
                try:
                    await asyncio.gather(response_pump, return_exceptions=True)
                except Exception:
                    pass
            if upstream_writer is not None:
                _close(upstream_writer)
            if conn is not None:
                conn.finish()
            _close(client_writer)

    def _authorize(self, request):
        """None means allow. Otherwise (response_bytes, status, note)."""
        if self.mode == MODE_OPEN:
            return None

        presented = request.presented_key()
        if presented and secrets.compare_digest(str(presented), str(self.key)):
            if request.key_in_query:
                # Strip the key from the URL and keep it in a cookie, so the
                # secret stops travelling in referrers, logs and the address bar.
                return (_redirect(request.path_without_key(), self.key), 303,
                        "key accepted, moved to cookie")
            return None

        return (_unlock_page(request.path, bool(presented), self.app_label),
                401, "no valid key")


def raise_if_cancelled():
    """Let cancellation propagate; swallow ordinary disconnects."""
    import sys
    exc = sys.exc_info()[1]
    if isinstance(exc, asyncio.CancelledError):
        raise exc


class _Connection(object):
    """Per-socket bookkeeping.

    Requests are enqueued by the request loop as each head is authorized, and
    dequeued as each response starts, so a request that streams for minutes is
    recorded the moment its response begins rather than when the socket
    closes.
    """

    def __init__(self, metrics, peer):
        self.metrics = metrics
        self.peer = peer
        self.pending = collections.deque()
        self.upgraded = False
        self.recorded = 0
        self.bytes_in = 0
        self.bytes_out = 0

    def expect(self, method, path, at=None):
        self.pending.append({"method": method, "path": path,
                             "at": at or time.time()})

    def saw_client_bytes(self, chunk):
        self.bytes_in += len(chunk)

    def saw_upstream_bytes(self, chunk):
        self.bytes_out += len(chunk)
        if self.upgraded:
            return
        status = _sniff_status(chunk)
        if status is None:
            return
        if status == 101:
            self.upgraded = True
        waiting = self.pending.popleft() if self.pending else {
            "method": "?", "path": "?", "at": time.time()}
        self.metrics.record(waiting["method"], waiting["path"], status,
                            (time.time() - waiting["at"]) * 1000.0,
                            client=self.peer)
        self.recorded += 1

    def finish(self):
        self.metrics.add_traffic(self.bytes_in, self.bytes_out)
        if self.recorded or not self.pending:
            return
        waiting = self.pending[0]
        self.metrics.record(waiting["method"], waiting["path"], 0,
                            (time.time() - waiting["at"]) * 1000.0,
                            note="no response from the app", client=self.peer)


# -- request parsing -----------------------------------------------------


class Request(object):
    def __init__(self, method, target, headers):
        self.method = method
        self.target = target
        self.headers = headers
        parsed = urllib.parse.urlsplit(target)
        self.path = parsed.path or "/"
        self.query = parsed.query
        self._params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        self.key_in_query = any(p in self._params for p in KEY_PARAMS)

    def presented_key(self):
        for param in KEY_PARAMS:
            values = self._params.get(param)
            if values and values[0]:
                return values[0]
        cookie = self.headers.get("cookie", "")
        for part in cookie.split(";"):
            name, _, value = part.strip().partition("=")
            if name == COOKIE_NAME and value:
                return urllib.parse.unquote(value)
        header_key = self.headers.get("x-shayef-key")
        if header_key:
            return header_key.strip()
        # API clients only know how to send an API key. When the app behind
        # the guard uses the same key, they pass both checks with one header,
        # which the guard forwards untouched for the app to verify too.
        bearer = self.headers.get("authorization", "")
        if bearer[:7].lower() == "bearer " and bearer[7:].strip():
            return bearer[7:].strip()
        api_key = self.headers.get("x-api-key")
        if api_key:
            return api_key.strip()
        return None

    def path_without_key(self):
        remaining = []
        for key, values in self._params.items():
            if key in KEY_PARAMS:
                continue
            for value in values:
                remaining.append((key, value))
        query = urllib.parse.urlencode(remaining)
        return self.path + ("?" + query if query else "")

    def is_upgrade(self):
        if self.method == "CONNECT":
            return True
        if not self.headers.get("upgrade"):
            return False
        return "upgrade" in self.headers.get("connection", "").lower()

    def body_length(self):
        """(kind, value): ("chunked", None), ("length", n) or ("none", 0)."""
        if "chunked" in self.headers.get("transfer-encoding", "").lower():
            return "chunked", None
        raw = self.headers.get("content-length")
        if raw:
            try:
                return "length", max(0, int(raw.strip()))
            except ValueError:
                return "unknown", None
        return "none", 0


def _parse_head(head):
    try:
        text = head.split(b"\r\n\r\n", 1)[0].decode("latin-1")
    except Exception:
        text = ""
    lines = text.split("\r\n")
    request_line = lines[0] if lines else ""
    parts = request_line.split(" ")
    method = parts[0] if parts else "?"
    target = parts[1] if len(parts) > 1 else "/"
    headers = {}
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if sep:
            headers[name.strip().lower()] = value.strip()
    return Request(method, target, headers)


async def _forward_body(request, reader, writer, conn):
    """Forward one request body. True means the connection is now raw.

    Framing the request side is what lets the guard reach the next request's
    headers and authorize it, instead of trusting the socket.
    """
    if request.is_upgrade():
        return True

    kind, length = request.body_length()

    if kind == "none":
        return False

    if kind == "unknown":
        # An unparseable framing header: go raw rather than risk desyncing and
        # corrupting the stream.
        return True

    if kind == "length":
        remaining = length
        while remaining > 0:
            chunk = await reader.read(min(PIPE_CHUNK, remaining))
            if not chunk:
                return True
            conn.bytes_in += len(chunk)
            writer.write(chunk)
            remaining -= len(chunk)
        return False

    # chunked
    while True:
        try:
            line = await reader.readuntil(b"\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            return True
        conn.bytes_in += len(line)
        writer.write(line)
        try:
            size = int(line.strip().split(b";")[0], 16)
        except ValueError:
            return True
        if size == 0:
            # Trailers, up to the terminating blank line.
            while True:
                try:
                    trailer = await reader.readuntil(b"\r\n")
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                    return True
                conn.bytes_in += len(trailer)
                writer.write(trailer)
                if trailer == b"\r\n":
                    return False
        remaining = size + 2          # the chunk data plus its trailing CRLF
        while remaining > 0:
            chunk = await reader.read(min(PIPE_CHUNK, remaining))
            if not chunk:
                return True
            conn.bytes_in += len(chunk)
            writer.write(chunk)
            remaining -= len(chunk)


async def _pump(reader, writer, observe=None):
    """Copy one direction verbatim, telling `observe` about each chunk.

    Bytes are never altered or buffered beyond the copy, which is what keeps
    SSE and WebSocket traffic behaving as the app intended.
    """
    try:
        while True:
            chunk = await reader.read(PIPE_CHUNK)
            if not chunk:
                break
            if observe is not None:
                try:
                    observe(chunk)
                except Exception:
                    pass
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        try:
            if writer.can_write_eof():
                writer.write_eof()
        except Exception:
            pass


def _sniff_status(chunk):
    """The status code, if this chunk begins an HTTP response."""
    if not chunk.startswith(b"HTTP/"):
        return None
    try:
        line = chunk.split(b"\r\n", 1)[0].decode("latin-1")
        parts = line.split(" ")
        if len(parts) > 1:
            return int(parts[1])
    except Exception:
        pass
    return None


def _peer_of(writer):
    try:
        peer = writer.get_extra_info("peername")
        return peer[0] if peer else None
    except Exception:
        return None


def _close(writer):
    try:
        writer.close()
    except Exception:
        pass


# -- responses -----------------------------------------------------------


def _response(status, reason, body_html, extra_headers=()):
    body = body_html.encode("utf-8")
    lines = ["HTTP/1.1 %d %s" % (status, reason),
             "Content-Type: text/html; charset=utf-8",
             "Content-Length: %d" % len(body),
             "Cache-Control: no-store",
             "Connection: close"]
    lines.extend(extra_headers)
    return ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8") + body


def _redirect(location, key):
    cookie = "%s=%s; Path=/; HttpOnly; SameSite=Lax; Secure; Max-Age=604800" % (
        COOKIE_NAME, urllib.parse.quote(key))
    return _response(303, "See Other",
                     _page("Unlocked", "Taking you through&hellip;"),
                     ["Location: " + (location or "/"), "Set-Cookie: " + cookie])


def _unlock_page(path, wrong_key, app_label):
    message = "That key is not right." if wrong_key else "This tunnel is private."
    detail = ("Check the key shayef printed when the tunnel started."
              if wrong_key else
              "Enter the key shayef printed when the tunnel started.")
    form = (
        '<form method="get" action="%s" class="row">'
        '<input type="password" name="k" placeholder="tunnel key" autofocus '
        'autocomplete="current-password" spellcheck="false">'
        '<button type="submit">Unlock</button>'
        "</form>"
        '<p class="meta">Protecting %s</p>'
    ) % (_escape(path or "/"), _escape(app_label))
    return _response(401, "Unauthorized", _page(message, detail + form))


def _not_up_page(app_label, port):
    return _response(
        502, "Bad Gateway",
        _page("Nothing is listening yet",
              "The tunnel is up, but %s is not answering on port %d. "
              "If a model is still loading, give it a moment and reload."
              % (_escape(app_label), port)))


def _page(title, body_html):
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>%s &middot; shayef</title><style>"
        ":root{color-scheme:dark light}"
        "*{box-sizing:border-box}"
        "body{margin:0;min-height:100vh;display:grid;place-items:center;"
        "padding:16px;background:#0d0d0d;color:#f0efec;"
        "font:15px/1.55 system-ui,-apple-system,'Segoe UI',sans-serif}"
        ".card{width:100%%;max-width:27rem;background:#1a1a19;"
        "border:1px solid rgba(255,255,255,.1);border-radius:14px;padding:28px}"
        ".mark{font-weight:650;letter-spacing:-.01em;font-size:13px;"
        "margin:0 0 20px;color:#f0efec}"
        ".mark span{color:#f38020}"
        "h1{font-size:19px;line-height:1.3;margin:0 0 8px;font-weight:620}"
        "p{margin:0 0 18px;color:#c3c2b7}"
        ".meta{font-size:12px;color:#898781;margin:16px 0 0}"
        ".row{display:flex;gap:8px}"
        "input{flex:1;min-width:0;padding:10px 12px;border-radius:9px;"
        "border:1px solid #383835;background:#0d0d0d;color:#f0efec;font:inherit}"
        "input:focus{outline:2px solid #f38020;outline-offset:1px;border-color:#f38020}"
        "button{padding:10px 16px;border-radius:9px;border:0;background:#f38020;"
        "color:#1a1205;font:inherit;font-weight:600;cursor:pointer}"
        "button:hover{background:#ff9140}"
        "@media(prefers-color-scheme:light){"
        "body{background:#f9f9f7;color:#0b0b0b}"
        ".card{background:#fcfcfb;border-color:rgba(11,11,11,.1)}"
        ".mark{color:#0b0b0b}h1{color:#0b0b0b}p{color:#52514e}"
        "input{background:#fcfcfb;border-color:#c3c2b7;color:#0b0b0b}}"
        "</style></head><body><div class=\"card\">"
        "<p class=\"mark\">shayef<span>.</span>me</p>"
        "<h1>%s</h1><p>%s</p></div></body></html>"
    ) % (_escape(title), _escape(title), body_html)


def _escape(text):
    return (str("" if text is None else text).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))
