"""The memory this notebook may actually use, and whether a folder lives in RAM.

Hosted notebooks (molab, Colab) run in containers. ``psutil.virtual_memory()`` there reports the whole
host machine (often hundreds of GB), while the container is killed at its own limit, taking the
notebook with it. The cgroup files hold the real limit and usage.
"""

from __future__ import annotations

import os
from pathlib import Path

import psutil

NO_LIMIT = 1 << 60  # cgroup v1 reports "unlimited" as a huge number
MEMORY_FS = {"tmpfs", "ramfs"}


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _stat(path: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in (_read(path) or "").splitlines():
        key, _, value = line.partition(" ")
        if value.isdigit():
            out[key] = int(value)
    return out


def container_memory() -> tuple[int, int]:
    """(bytes in use that can't be reclaimed, limit in bytes) for this container.

    "In use" counts process memory and RAM-backed files (tmpfs), not the page cache of files on disk,
    which the kernel drops under pressure; counting it would flag every large download.
    """
    vm = psutil.virtual_memory()
    limit, used = vm.total, vm.total - vm.available
    v2 = _read("/sys/fs/cgroup/memory.max")
    if v2 is not None:
        if v2.isdigit():
            limit = min(limit, int(v2))
        stat = _stat("/sys/fs/cgroup/memory.stat")
        if stat:
            used = stat.get("anon", 0) + stat.get("shmem", 0) + stat.get("kernel", 0)
        return used, limit
    v1 = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1 is not None and v1.isdigit():
        if int(v1) < NO_LIMIT:
            limit = min(limit, int(v1))
        stat = _stat("/sys/fs/cgroup/memory/memory.stat")
        if stat:
            used = stat.get("total_rss", stat.get("rss", 0)) + stat.get("total_shmem", stat.get("shmem", 0))
    return used, limit


def fs_type(path: Path) -> str:
    """Filesystem type of the mount holding ``path`` ("tmpfs" means it is stored in RAM)."""
    try:
        resolved = str(path.resolve())
    except OSError:
        return ""
    best, kind = "", ""
    try:
        parts = psutil.disk_partitions(all=True)
    except OSError:
        return ""
    for part in parts:
        mount = part.mountpoint.rstrip("/") or "/"
        if (resolved == mount or resolved.startswith(mount + "/") or mount == "/") and len(mount) >= len(best):
            best, kind = mount, part.fstype
    return kind


def in_memory(path: Path) -> bool:
    return fs_type(path) in MEMORY_FS


NETWORK_FS = {"nfs", "nfs4", "cifs", "smb3", "smbfs", "9p", "ceph", "glusterfs", "lustre", "afs"}


def on_network(path: Path) -> bool:
    """Network and FUSE mounts often report huge free space but stall (or hit a quota) under GB of writes."""
    kind = fs_type(path)
    return kind in NETWORK_FS or kind.startswith("fuse")


def roomiest_home(name: str = "kindle") -> Path:
    """The writable, local, disk-backed place with the most free space, for models and caches.

    Local disks come before network mounts, whatever their free space. A previous run's folder wins,
    so every helper (``kindle.stop()``, the server, re-runs) agrees.
    """
    roots = ["/tmp", str(Path.home()), os.getcwd(), "/mnt", "/data", "/workspace", "/scratch", "/var/tmp"]
    seen: set[str] = set()
    options: list[tuple[bool, int, Path]] = []
    for root in roots:
        base = Path(root)
        if str(base) in seen or not base.is_dir() or not os.access(base, os.W_OK):
            continue
        seen.add(str(base))
        home = base / name
        if (home / "state.json").exists():
            return home
        if in_memory(base):
            continue
        try:
            options.append((not on_network(base), psutil.disk_usage(str(base)).free, home))
        except OSError:
            continue
    return max(options)[2] if options else Path("/tmp") / name
