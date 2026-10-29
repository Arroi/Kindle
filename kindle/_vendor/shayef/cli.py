"""The shayef command line.

Mirrors the Python API so the same things can be driven from a terminal or
from a notebook shell cell (`!shayef ls`).
"""

import argparse
import json
import os
import sys
import time

from . import env as envmod
from . import relay as relay_mod
from .client import ControlPlane, platform_tag
from .config import Config
from .errors import ShayefError
from .log import log
from .session import ACCESS_KEY, ACCESS_OPEN, Tunnel
from .version import __version__

PROGRAM = "shayef"


def build_parser():
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="A private address and dashboard for AI chat UIs in notebooks.",
        epilog="Run `%s up 7860 chat` to serve a local app at chat.shayef.me."
               % PROGRAM)
    parser.add_argument("--version", action="version",
                        version="%s %s" % (PROGRAM, __version__))
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="include debug events")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output where it makes sense")
    parser.add_argument("--api", help="control plane base URL")

    subs = parser.add_subparsers(dest="command", metavar="<command>")

    up = subs.add_parser("up", help="open a tunnel to a local port")
    up.add_argument("port", type=int, help="the local port your app listens on")
    up.add_argument("name", help="the subdomain to serve it at, e.g. chat")
    up.add_argument("--host", default="127.0.0.1", help="local host to forward to")
    up.add_argument("--access", default=ACCESS_KEY,
                    choices=[ACCESS_KEY, ACCESS_OPEN],
                    help="how the hostname is gated (default: key)")
    up.add_argument("--key", default=os.environ.get("SHAYEF_KEY"),
                    help="use this key instead of generating one (default: "
                         "SHAYEF_KEY, which keeps it off the command line)")
    up.add_argument("--label", help="friendly name, shown on the dashboard")
    up.add_argument("--recreate", action="store_true",
                    help="rebuild the relay before starting")
    up.add_argument("--no-guard", action="store_true",
                    help="forward straight to the app: no gate, no request log")
    up.add_argument("--share-key", action="store_true",
                    help="let the dashboard show the unlock link (sends the "
                         "key to the control plane)")
    up.add_argument("--beat", type=float, default=5.0,
                    help="seconds between telemetry reports (default: 5)")
    up.add_argument("--timeout", type=int, default=75,
                    help="seconds to wait for the relay (default: 75)")

    down = subs.add_parser("down", help="close a session and release its name")
    down.add_argument("name", help="the subdomain to release")

    subs.add_parser("ls", help="list your sessions")

    status = subs.add_parser("status", help="show one session")
    status.add_argument("name", nargs="?", help="subdomain; omit to list all")

    login = subs.add_parser("login", help="store a shayef token")
    login.add_argument("--token", help="the token (prompted if omitted)")
    login.add_argument("--show", action="store_true",
                       help="print the stored config instead of changing it")

    subs.add_parser("doctor", help="check the environment and the token")

    install = subs.add_parser("install", help="pre-fetch the relay connector")
    install.add_argument("--force", action="store_true",
                         help="re-download even if cached")

    tokens = subs.add_parser("tokens", help="manage tokens (admin)")
    token_subs = tokens.add_subparsers(dest="token_command", metavar="<action>")
    token_new = token_subs.add_parser("new", help="issue a token")
    token_new.add_argument("--kind", default="client",
                           choices=["client", "admin"])
    token_new.add_argument("--label", help="what this token is for")
    token_subs.add_parser("ls", help="list tokens")
    token_rm = token_subs.add_parser("rm", help="revoke a token")
    token_rm.add_argument("id", help="the token id from `tokens ls`")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        log.min_level = "debug"

    if not args.command:
        parser.print_help()
        return 0

    handler = {
        "up": cmd_up,
        "down": cmd_down,
        "ls": cmd_ls,
        "status": cmd_status,
        "login": cmd_login,
        "doctor": cmd_doctor,
        "install": cmd_install,
        "tokens": cmd_tokens,
    }[args.command]

    try:
        return handler(args) or 0
    except ShayefError as exc:
        print("")
        print("  error: %s" % exc.message, file=sys.stderr)
        if exc.hint:
            print("  hint:  %s" % exc.hint, file=sys.stderr)
        print("")
        return 1
    except KeyboardInterrupt:
        print("")
        return 130


def _control(args):
    config = Config.load(api_base=getattr(args, "api", None))
    config.require_token()
    return ControlPlane(config.api_base, config.api_token), config


# -- commands ------------------------------------------------------------


def cmd_up(args):
    tunnel = Tunnel(
        args.port,
        args.name,
        host=args.host,
        access=args.access,
        key=args.key,
        label=args.label,
        api=args.api,
        guard=not args.no_guard,
        recreate=args.recreate,
        share_key=args.share_key,
        beat_interval=args.beat,
    )
    tunnel.start(wait=True, timeout=args.timeout)
    if args.json:
        print(json.dumps(tunnel.snapshot(), indent=2, default=str))
    print("  Ctrl-C to stop. The name stays claimed; `shayef down %s` "
          "releases it." % tunnel.name)
    tunnel.wait()
    return 0


def cmd_down(args):
    control, _ = _control(args)
    result = control.close_session(args.name)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    removed = result.get("removed") or []
    if removed:
        print("  released %s for %s" % (", ".join(removed), result["hostname"]))
    else:
        print("  nothing to release for %s" % result.get("hostname", args.name))
    return 0


def cmd_ls(args):
    control, config = _control(args)
    rows = control.list_sessions()
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0
    if not rows:
        print("  no sessions yet")
        print("  open one with: %s up 7860 chat" % PROGRAM)
        return 0
    width = max(len(r["hostname"]) for r in rows)
    print("")
    print("  %s  %-12s  %s" % ("HOSTNAME".ljust(width), "STATE", "FORWARDS TO"))
    for row in rows:
        print("  %s  %-12s  %s" % (row["hostname"].ljust(width),
                                   row.get("state") or "unknown",
                                   row.get("service") or ""))
    print("")
    print("  dashboard: %s/" % config.api_base)
    print("")
    return 0


def cmd_status(args):
    if not args.name:
        return cmd_ls(args)
    control, _ = _control(args)
    session = control.get_session(args.name)
    if session is None:
        print("  no session named %s" % args.name)
        return 1
    if args.json:
        print(json.dumps(session, indent=2, default=str))
        return 0
    telemetry = session.get("telemetry") or {}
    print("")
    for label, value in (
        ("hostname", session.get("hostname")),
        ("state", session.get("state")),
        ("forwards to", session.get("service")),
        ("requests", telemetry.get("requests", 0)),
        ("blocked", telemetry.get("blocked", 0)),
        ("edge links", telemetry.get("connections", 0)),
        ("last seen", _ago(session.get("last_seen"))),
    ):
        print("    %-12s %s" % (label, value))
    print("")
    return 0


def cmd_login(args):
    config = Config.load(api_base=args.api)

    if args.show:
        print(json.dumps(config.redacted(), indent=2, default=str))
        return 0

    token = args.token or os.environ.get("SHAYEF_TOKEN")
    if not token:
        try:
            import getpass
            token = getpass.getpass("  shayef token: ").strip()
        except Exception:
            token = ""
    if not token:
        print("  no token given; nothing saved", file=sys.stderr)
        return 1

    config.api_token = token
    control = ControlPlane(config.api_base, token)

    # Prove the token works before writing it to disk.
    health = control.health()
    control.list_sessions()

    path = config.save()
    print("  signed in to %s" % config.api_base)
    print("  zone:   %s" % health.get("zone", "unknown"))
    print("  saved:  %s" % path)
    if os.name != "nt":
        print("  (file mode 0600)")
    return 0


def cmd_doctor(args):
    environment = envmod.detect()
    checks = []

    checks.append(("host", environment.kind, True,
                   "notebook" if environment.in_notebook else "terminal"))
    described = environment.describe()
    checks.append(("python", described["python"],
                   sys.version_info >= (3, 8), ""))
    checks.append(("platform", platform_tag(), True, ""))

    config = Config.load(api_base=args.api)
    checks.append(("control plane", config.api_base, True, ""))
    checks.append(("config", config.path or "(none)", True, ""))

    installed = relay_mod.relay_version()
    if installed:
        checks.append(("relay", installed, True, relay_mod.relay_path()))
    else:
        checks.append(("relay", "not installed", False,
                       "run `%s install`" % PROGRAM))

    if config.has_token:
        try:
            control = ControlPlane(config.api_base, config.api_token)
            health = control.health()
            checks.append(("reachable", "yes", True,
                           "zone " + str(health.get("zone", "?"))))
            sessions = control.list_sessions()
            checks.append(("token", "accepted", True,
                           "%d session(s)" % len(sessions)))
        except ShayefError as exc:
            checks.append(("token", "rejected", False, exc.message))
    else:
        checks.append(("token", "not set", False,
                       "run `%s login`" % PROGRAM))

    if args.json:
        print(json.dumps([
            {"check": c[0], "value": c[1], "ok": c[2], "detail": c[3]}
            for c in checks], indent=2, default=str))
        return 0

    print("")
    width = max(len(c[0]) for c in checks)
    failures = 0
    for name, value, ok, detail in checks:
        mark = "ok " if ok else "XX "
        if not ok:
            failures += 1
        line = "  %s %s  %s" % (mark, name.ljust(width), value)
        if detail:
            line += "   %s" % detail
        print(line)
    print("")
    if failures:
        print("  %d check%s attention." % (
            failures, " needs" if failures == 1 else "s need"))
    else:
        print("  Everything checks out.")
    print("")
    return 1 if failures else 0


def cmd_install(args):
    control, _ = _control(args)
    started = time.time()
    path = relay_mod.ensure_relay(control=control, force_download=args.force)
    print("  relay %s" % (relay_mod.relay_version(path) or "unknown"))
    print("  %s" % path)
    print("  (%.1fs)" % (time.time() - started))
    return 0


def cmd_tokens(args):
    control, _ = _control(args)
    action = getattr(args, "token_command", None)

    if action == "new":
        issued = control.issue_token(kind=args.kind, label=args.label)
        if args.json:
            print(json.dumps(issued, indent=2, default=str))
            return 0
        print("")
        print("  %s" % issued["token"])
        print("")
        print("  Copy it now: shayef.me stores only a hash and cannot show it")
        print("  again. Revoke it with `%s tokens rm %s`."
              % (PROGRAM, issued["record"]["id"]))
        print("")
        return 0

    if action == "rm":
        control.revoke_token(args.id)
        print("  revoked %s" % args.id)
        return 0

    rows = control.list_tokens()
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0
    if not rows:
        print("  no tokens")
        return 0
    print("")
    print("  %-38s %-7s %-10s %s" % ("ID", "KIND", "PREFIX", "LABEL"))
    for row in rows:
        print("  %-38s %-7s %-10s %s" % (
            row.get("id", ""), row.get("kind", ""),
            row.get("fingerprint", ""), row.get("label") or ""))
    print("")
    return 0


def _ago(milliseconds):
    if not milliseconds:
        return "never"
    seconds = max(0, time.time() - (milliseconds / 1000.0))
    if seconds < 60:
        return "%ds ago" % int(seconds)
    if seconds < 3600:
        return "%dm ago" % int(seconds / 60)
    return "%dh ago" % int(seconds / 3600)


if __name__ == "__main__":
    sys.exit(main())
