"""Package and safely expose scene exports produced by native world runners."""

from __future__ import annotations

import json
import mimetypes
import re
import shutil
import zipfile
from pathlib import Path

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".ply", ".glb", ".gltf", ".obj", ".mtl", ".spz", ".splat", ".mp4", ".json"}


def package_scene(folder: Path, output: Path) -> str:
    # Keep full exports for download; smaller mesh previews are practical in a browser.
    for path in list(folder.rglob("*.ply")):
        if path.is_symlink() or "preview" in path.parts:
            continue
        with path.open("rb") as stream:
            header = stream.read(8192)
        count = re.search(rb"element face (\d+)", header)
        if count and int(count.group(1)) > 200_000:
            try:
                import open3d as o3d

                mesh = o3d.io.read_triangle_mesh(str(path))
                preview = mesh.simplify_quadric_decimation(100_000)
                preview.compute_vertex_normals()
                target = folder / "preview" / path.relative_to(folder)
                target.parent.mkdir(parents=True, exist_ok=True)
                o3d.io.write_triangle_mesh(str(target), preview)
            except Exception as exc:
                print(f"Mesh preview unavailable for {path.name}: {type(exc).__name__}", flush=True)
    files = [path for path in folder.rglob("*") if path.is_file() and not path.is_symlink()
             and path.suffix.lower() in EXTENSIONS and path.resolve().is_relative_to(folder.resolve())]
    if sum(path.stat().st_size for path in files) > shutil.disk_usage(folder).free:
        raise ValueError("Not enough free disk space to package the world exports")
    assets = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for path in files:
            relative = path.relative_to(folder).as_posix()
            bundle.write(path, relative)
            kind = "image"
            if path.suffix.lower() in (".spz", ".splat"):
                kind = "splat"
            elif path.suffix.lower() == ".ply":
                with path.open("rb") as stream:
                    header = stream.read(8192)
                faces = re.search(rb"element face (\d+)", header)
                kind = "splat" if b"f_dc_0" in header or b"scale_0" in header else "mesh" if faces and int(faces.group(1)) else "points"
            elif path.suffix.lower() in (".glb", ".gltf", ".obj"):
                kind = "mesh"
            elif path.suffix.lower() == ".mp4":
                kind = "video"
            elif path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                kind = "data"
            assets.append({"path": relative, "size": path.stat().st_size, "kind": kind,
                           "media_type": mimetypes.guess_type(relative)[0] or "application/octet-stream"})
    output.with_suffix(".assets.json").write_text(json.dumps({"root": str(folder), "assets": assets}))
    return str(output)


def manifest(result: str) -> dict:
    path = Path(result).with_suffix(".assets.json")
    return json.loads(path.read_text()) if path.exists() else {"assets": []}


def asset_path(result: str, name: str) -> tuple[Path, str]:
    data = manifest(result)
    asset = next((item for item in data["assets"] if item["path"] == name), None)
    if not asset:
        raise KeyError(name)
    root = Path(data["root"]).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise KeyError(name)
    return path, asset["media_type"]
