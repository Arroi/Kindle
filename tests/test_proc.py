"""ManagedProcess keeps its whole interface (every engine starts its runners through it)."""

import sys

from kindle.proc import ManagedProcess, sacrifice


def test_managed_process_has_its_methods():
    # A module-level function once landed in the middle of the class and swallowed these methods,
    # breaking every download and runner. The shutdown tests that use them only run on Linux.
    for name in ("start", "_pump", "running", "returncode", "wait", "stop"):
        assert hasattr(ManagedProcess, name), name


async def test_managed_process_runs_and_captures_output(tmp_path):
    lines = []
    proc = ManagedProcess("echo", [sys.executable, "-c", "print('hello from child')"],
                          log_path=tmp_path / "out.log", on_line=lines.append)
    await proc.start()
    assert await proc.wait() == 0
    assert "hello from child" in lines and not proc.running
    assert "hello from child" in (tmp_path / "out.log").read_text()


def test_sacrifice_ignores_platforms_without_proc():
    sacrifice(999_999_999, 1000)  # no such process (or no /proc): must not raise
