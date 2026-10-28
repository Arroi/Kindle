"""Hosted notebooks: rootless Ollama install, real container memory, memory guard, context defaults."""

import asyncio
import io
import json
import pathlib
import tarfile
import time
from types import SimpleNamespace

import httpx
import pytest

from kindle.deploy import manager as manager_mod
from kindle.deploy.manager import Deployment
from kindle.engines import ollama
from kindle.engines.context import EngineContext
from kindle.gateway.common import context_of, fit_reply
from kindle.schemas import DeploymentSpec, ModelRef, Settings
from kindle.server.state import AppState
from kindle.system import memory

zstandard = pytest.importorskip("zstandard")


def fake_ollama_archive() -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in (("bin/ollama", b"#!/bin/sh\necho ollama\n"), ("lib/ollama/libggml.so", b"\x7fELF")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return zstandard.ZstdCompressor().compress(raw.getvalue())


def ctx_for(paths) -> EngineContext:
    return EngineContext(paths=paths, dep_id="d1", served_name="m", gpus=[0], hw=None, log=lambda line: None,
                         progress=lambda *a: None, internal_key="k", main_port=1)


async def test_ollama_installs_without_root(paths, monkeypatch):
    body = fake_ollama_archive()
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=body, headers={"content-length": str(len(body))})

    real = httpx.AsyncClient
    monkeypatch.setattr(ollama.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(ollama.platform, "machine", lambda: "x86_64")
    exe = await ollama.install_ollama(ctx_for(paths))
    assert seen == ["https://ollama.com/download/ollama-linux-amd64.tar.zst"]
    assert exe.endswith("ollama-dist/bin/ollama".replace("/", "\\") if "\\" in exe else "ollama-dist/bin/ollama")
    assert (paths.bin / "ollama-dist" / "lib" / "ollama" / "libggml.so").exists()
    assert not (paths.cache / "ollama-linux-amd64.tar.zst").exists()  # archive removed after unpacking


async def test_ollama_install_refuses_when_disk_is_short(paths, monkeypatch):
    monkeypatch.setattr(ollama.shutil, "disk_usage", lambda p: SimpleNamespace(free=10**9))
    monkeypatch.setattr(ollama.platform, "machine", lambda: "x86_64")
    with pytest.raises(ollama.DeployError, match="KINDLE_HOME"):
        await ollama.install_ollama(ctx_for(paths))


def test_ollama_unpack_keeps_only_the_drivers_cuda(tmp_path):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name in ("bin/ollama", "lib/ollama/libggml-base.so", "lib/ollama/cuda_v12/libcublas.so",
                     "lib/ollama/cuda_v13/libcublas.so"):
            info = tarfile.TarInfo(name)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    archive = tmp_path / "o.tar.zst"
    archive.write_bytes(zstandard.ZstdCompressor().compress(raw.getvalue()))
    ollama._unpack_zst(archive, tmp_path / "out", "13")
    lib = tmp_path / "out" / "lib" / "ollama"
    assert (lib / "cuda_v13" / "libcublas.so").exists() and not (lib / "cuda_v12").exists()
    assert (lib / "libggml-base.so").exists() and (tmp_path / "out" / "bin" / "ollama").exists()
    ollama._unpack_zst(archive, tmp_path / "all", None)  # driver unknown: keep everything
    assert (tmp_path / "all" / "lib" / "ollama" / "cuda_v12").exists()


def _scratch(tmp_path, monkeypatch, sizes: dict, network=()):
    roots = [tmp_path / name for name in sizes]
    for r in roots:
        r.mkdir()
    monkeypatch.setattr(memory, "SCRATCH", tuple(str(r) for r in roots))
    monkeypatch.setattr(memory, "in_memory", lambda p: False)
    monkeypatch.setattr(memory, "on_network", lambda p: p.name in network)
    monkeypatch.setattr(memory.psutil, "disk_usage", lambda p: SimpleNamespace(free=sizes[pathlib.Path(p).name]))
    return {r.name: r for r in roots}


def test_home_prefers_local_disk_over_a_roomier_network_mount(tmp_path, monkeypatch):
    roots = _scratch(tmp_path, monkeypatch, {"local": 50, "net": 10**6}, network={"net"})
    assert memory.roomiest_home() == roots["local"] / "kindle-data"


def test_home_reuses_a_previous_run(tmp_path, monkeypatch):
    roots = _scratch(tmp_path, monkeypatch, {"big": 10**6, "old": 5})
    (roots["old"] / "kindle-data").mkdir()
    (roots["old"] / "kindle-data" / "state.json").write_text("{}")
    assert memory.roomiest_home() == roots["old"] / "kindle-data"


def test_home_is_never_importable_or_in_the_notebook_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    home = memory.roomiest_home()
    assert home.name.isidentifier() is False  # "kindle-data": Python can't import it as "kindle"
    assert tmp_path not in home.parents


def test_old_kindle_folders_are_found_and_cleaned(tmp_path, monkeypatch, capsys):
    from kindle import molab

    old = tmp_path / "kindle"
    (old / "logs").mkdir(parents=True)
    (old / "state.json").write_text("{}")
    (old / "models.bin").write_bytes(b"x" * 1000)
    unrelated = tmp_path / "other" / "kindle"
    unrelated.mkdir(parents=True)  # no state.json: not ours, never touched
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(molab.Path, "home", staticmethod(lambda: tmp_path / "other"))
    assert molab.old_data_folders() == [old]
    molab.clean_old_data()
    assert not old.exists() and unrelated.exists()
    assert "Removed" in capsys.readouterr().out


def test_container_memory_reads_cgroup_v2(monkeypatch):
    files = {"/sys/fs/cgroup/memory.max": "34359738368",
             "/sys/fs/cgroup/memory.stat": "anon 8000000000\nfile 50000000000\nshmem 2000000000\nkernel 100\n"}
    monkeypatch.setattr(memory, "_read", lambda path: files.get(path))
    host = 1024 * 10**9  # what psutil sees inside a container: the whole host
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: SimpleNamespace(total=host, available=host // 2))
    used, limit = memory.container_memory()
    assert limit == 34359738368  # the container's limit, not the host's RAM
    assert used == 8000000000 + 2000000000 + 100  # page cache ("file") doesn't count


def test_container_memory_cgroup_v1_unlimited_falls_back(monkeypatch):
    files = {"/sys/fs/cgroup/memory/memory.limit_in_bytes": str(1 << 62),
             "/sys/fs/cgroup/memory/memory.stat": "total_rss 5\ntotal_shmem 7\n"}
    monkeypatch.setattr(memory, "_read", lambda path: files.get(path))
    used, limit = memory.container_memory()
    assert used == 12 and limit < (1 << 62)


def test_reply_length_is_capped_to_the_window():
    spec = DeploymentSpec(modality="text", model=ModelRef(id="m", engine="vllm"), options={"ctx": 32768})
    dep = Deployment(id="d", spec=spec, served_name="m")
    body = {"messages": [{"role": "user", "content": "x" * 32000}], "max_tokens": 32000}
    capped = fit_reply(body, dep)["max_tokens"]
    assert 20000 < capped < 32768 - 9000  # the prompt's ~10K tokens are left room for
    assert fit_reply({"messages": [], "max_tokens": 1000}, dep)["max_tokens"] == 1000
    assert context_of(dep) == 32768
    linked = Deployment(id="l", spec=DeploymentSpec(modality="text", model=ModelRef(id="m", engine="linked")),
                        served_name="m")
    assert fit_reply(body, linked) is body  # a linked session knows its own window


def test_default_context_is_the_longest_that_fits(paths):
    from kindle.catalog.service import CatalogService
    from kindle.system.probe import Probe

    hw = Probe(paths, fake="t4x2").snapshot()
    svc = CatalogService(lambda: hw)
    small = {"modality": "text", "engine": "vllm", "source": "hf", "id": "a/small", "size": 1_000_000_000,
             "params": 0.5e9, "context_length": 131072, "options": {}}
    f = svc._apply_fit(small, hw)
    assert f.verdict in ("best", "good") and small["spec"]["options"]["ctx"] == 131072
    capped = {**small, "id": "a/short", "context_length": 32768, "options": {}}
    svc._apply_fit(capped, hw)
    assert capped["spec"]["options"]["ctx"] == 32768  # never past the model's own window
    chosen = {**small, "id": "a/chosen", "options": {"ctx": 4096}}
    svc._apply_fit(chosen, hw)
    assert chosen["spec"]["options"]["ctx"] == 4096  # an explicit choice is kept


def test_memory_guard_stops_the_hungriest_deployment(paths, monkeypatch):
    s = Settings()
    s.exposure.provider = "none"
    st = AppState(paths=paths, settings=s, port=8999, mock=False, fake_hw="t4x2")
    monkeypatch.setattr(manager_mod, "MEMORY_CHECK_SECONDS", 0.01)
    monkeypatch.setattr(manager_mod, "container_memory", lambda: (31 * 10**9, 32 * 10**9))
    spec = DeploymentSpec(modality="text", name="big", model=ModelRef(id="m", engine="mock"))
    dep = Deployment(id="big", spec=spec, served_name="big", status="loading", created_at=time.time())
    st.manager.deployments[dep.id] = dep

    async def run():
        task = asyncio.create_task(st.manager.guard_memory())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if dep.status == "error":
                break
        task.cancel()

    asyncio.run(run())
    assert dep.status == "error" and "run out of memory" in dep.error
    assert json.loads((paths.home / "deployments.json").read_text()) == []  # not restored into the same crash
