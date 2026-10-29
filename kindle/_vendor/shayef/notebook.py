"""Showing the dashboard from inside a notebook.

The dashboard is served by shayef.me, not by the notebook, which makes this
much simpler than it used to be: there is no kernel port to proxy, so a plain
iframe works on every host, Kaggle included.

Everything degrades to a printed summary rather than raising, because losing
the dashboard must never take the tunnel down with it.
"""

from .log import log


def show_dashboard(tunnel, inline=True, height=760):
    """Display the dashboard for a tunnel. Returns what was displayed, or the
    dashboard URL when nothing could be rendered."""
    url = tunnel.dashboard_url

    if not inline or not tunnel.environment.in_notebook:
        print("  Dashboard: %s" % url)
        return url

    html = _display_html()
    if html is None:
        print("  Dashboard: %s" % url)
        return url

    try:
        return html(
            '<div style="font:14px system-ui,-apple-system,\'Segoe UI\','
            'sans-serif;margin:0 0 8px">'
            '<a href="%s" target="_blank" rel="noopener" '
            'style="color:#f38020;text-decoration:none;font-weight:600">'
            'Open the shayef.me dashboard &nearr;</a></div>'
            '<iframe src="%s" style="width:100%%;height:%dpx;border:1px solid '
            'rgba(128,128,128,.3);border-radius:12px" '
            'title="shayef dashboard"></iframe>' % (url, url, height))
    except Exception as exc:
        log.debug("could not render the dashboard inline", error=str(exc))
        print("  Dashboard: %s" % url)
        return url


def summary(tunnel):
    """A static inline card, for when the dashboard is not wanted."""
    html = _display_html()
    rows = [
        ("Address", tunnel.url or "waiting"),
        ("Status", tunnel.status),
        ("Forwards to", "http://" + tunnel.target),
    ]
    if tunnel.guard and tunnel.guard.mode == "key":
        rows.append(("Key", tunnel.key))

    if html is None:
        print("")
        for label, value in rows:
            print("    %-12s %s" % (label, value))
        print("")
        return None

    cells = "".join(
        '<div style="display:flex;gap:10px;padding:6px 0;border-bottom:1px '
        'solid rgba(128,128,128,.18)"><span style="min-width:7.5rem;'
        'color:#898781;font-size:12px;text-transform:uppercase;'
        'letter-spacing:.02em">%s</span><span style="font-size:13px;'
        'word-break:break-all">%s</span></div>'
        % (_escape(label), _escape(value)) for label, value in rows)

    return html(
        '<div style="font:14px system-ui,-apple-system,\'Segoe UI\',sans-serif;'
        'border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:16px;'
        'max-width:40rem">'
        '<p style="margin:0 0 10px;font-weight:650;font-size:13px">'
        'shayef<span style="color:#f38020">.</span>me</p>%s</div>' % cells)


def _display_html():
    """IPython's HTML display function, or None outside IPython."""
    try:
        from IPython.display import HTML, display
    except Exception:
        return None

    def render(markup):
        node = HTML(markup)
        display(node)
        return node

    return render


def _escape(text):
    return (str("" if text is None else text).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))
