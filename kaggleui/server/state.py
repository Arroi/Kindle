"""Process-wide state shared by every route."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..catalog.imports import ModelImports
from ..catalog.service import CatalogService
from ..config import Paths
from ..deploy.manager import DeploymentManager
from ..interfaces.manager import InterfaceManager
from ..links import LinkManager
from ..schemas import Preset, Settings
from ..system.probe import Probe
from ..tunnels.manager import TunnelManager
from .auth import Auth
from .events import EventBus
from .updates import Updates


@dataclass
class AppState:
    paths: Paths
    settings: Settings
    port: int
    listen_host: str = "127.0.0.1"
    mock: bool = False
    fake_hw: str | None = None
    preset: Preset | None = None
    secrets: dict[str, bool] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    video_jobs: dict[str, dict] = field(default_factory=dict)
    world_jobs: dict[str, dict] = field(default_factory=dict)

    auth: Auth = field(init=False)
    bus: EventBus = field(init=False)
    probe: Probe = field(init=False)
    catalog: CatalogService = field(init=False)
    imports: ModelImports = field(init=False)
    tunnels: TunnelManager = field(init=False)
    manager: DeploymentManager = field(init=False)
    interfaces: InterfaceManager = field(init=False)
    updates: Updates = field(init=False)
    links: LinkManager = field(init=False)

    def __post_init__(self) -> None:
        self.auth = Auth(self.paths.home / "session.key")
        self.bus = EventBus()
        self.probe = Probe(self.paths, fake=self.fake_hw)
        self.imports = ModelImports(self.paths.models / "imports", self.paths.envs)
        self.catalog = CatalogService(self.probe.snapshot, imports_root=self.imports.root)
        self.tunnels = TunnelManager(self.paths, on_change=lambda t: self.bus.publish("tunnel", t.info()))
        self.manager = DeploymentManager(self)
        self.interfaces = InterfaceManager(self)
        self.updates = Updates(self)
        self.links = LinkManager(self)

    def main_url(self) -> str:
        t = self.tunnels.get("main")
        return t.url.rstrip("/") if t and t.url else f"http://127.0.0.1:{self.port}"

    def local_url(self) -> str:
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(self.listen_host, self.listen_host)
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}"
