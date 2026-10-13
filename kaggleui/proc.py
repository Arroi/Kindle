"""Async subprocess helpers: supervised long-running processes and one-shot commands with live output."""

from __future__ import annotations

import asyncio
import os
import signal
from collections import deque
from collections.abc import Callable
from pathlib import Path

LineHandler = Callable[[str], None]


class ManagedProcess:
    """A long-running child process in its own process group, with line-by-line log capture."""

    def __init__(self, name: str, cmd: list[str], *, env: dict[str, str] | None = None,
                 cwd: str | Path | None = None, log_path: Path | None = None,
                 on_line: LineHandler | None = None):
        self.name = name
        self.cmd = cmd
        self.env = env
        self.cwd = str(cwd) if cwd else None
        self.log_path = log_path
        self.on_line = on_line
        self.proc: asyncio.subprocess.Process | None = None
        self.tail: deque[str] = deque(maxlen=400)
        self._pumps: list[asyncio.Task] = []

    async def start(self) -> None:
        env = {**os.environ, **(self.env or {})}
        self.proc = await asyncio.create_subprocess_exec(
            *self.cmd, env=env, cwd=self.cwd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
        )
        self._pumps = [asyncio.create_task(self._pump())]

    async def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        logf = open(self.log_path, "a", buffering=1) if self.log_path else None
        try:
            while True:
                raw = await self.proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode(errors="replace").rstrip("\r\n")
                # tqdm-style progress bars use \r; keep only the latest segment.
                if "\r" in line:
                    line = line.split("\r")[-1]
                self.tail.append(line)
                if logf:
                    logf.write(line + "\n")
                if self.on_line:
                    try:
                        self.on_line(line)
                    except Exception:
                        pass
        finally:
            if logf:
                logf.close()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    @property
    def returncode(self) -> int | None:
        return self.proc.returncode if self.proc else None

    async def wait(self) -> int:
        assert self.proc
        rc = await self.proc.wait()
        for t in self._pumps:
            try:
                await asyncio.wait_for(t, timeout=2)
            except Exception:
                pass
        return rc

    async def stop(self, timeout: float = 10) -> None:
        if not self.running:
            return
        assert self.proc
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self.proc.wait()


async def run_cmd(cmd: list[str] | str, *, on_line: LineHandler | None = None,
                  env: dict[str, str] | None = None, cwd: str | Path | None = None,
                  timeout: float | None = None) -> int:
    """Run a command to completion, streaming output lines. ``cmd`` as str runs through bash."""
    full_env = {**os.environ, **(env or {})}
    if isinstance(cmd, str):
        proc = await asyncio.create_subprocess_exec(
            "bash", "-lc", cmd, env=full_env, cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
        )
    else:
        proc = await asyncio.create_subprocess_exec(
            *cmd, env=full_env, cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
        )

    async def pump() -> None:
        assert proc.stdout
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode(errors="replace").rstrip("\r\n").split("\r")[-1]
            if on_line and line:
                on_line(line)

    try:
        await asyncio.wait_for(asyncio.gather(pump(), proc.wait()), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()
        raise
    return proc.returncode or 0
