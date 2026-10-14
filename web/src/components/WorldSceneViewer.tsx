import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

import { Button } from "./ui";

export interface WorldAsset { path: string; kind: "image" | "mesh" | "points" | "splat" | "video" | "data"; size: number; }
export const worldAssetUrl = (job: string, path: string) => `/_kui/api/studio/worlds/${job}/assets/${path.split("/").map(encodeURIComponent).join("/")}`;

export function WorldSceneViewer({ job, assets }: { job: string; assets: WorldAsset[] }) {
  const host = useRef<HTMLDivElement>(null);
  const reset = useRef<(inside: boolean) => void>(() => {});
  const [message, setMessage] = useState("Loading scene…");
  const [error, setError] = useState("");
  const signature = assets.map(asset => asset.path).join("|");
  useEffect(() => {
    const mount = host.current!;
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: false }); }
    catch { setError("This browser could not create a WebGL2 viewer. You can still download the scene."); return; }
    setError(""); setMessage("Loading scene…");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#11181c");
    const camera = new THREE.PerspectiveCamera(65, 1, 0.001, 10000);
    camera.position.set(0, 1, 4);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.domElement.tabIndex = 0;
    renderer.domElement.setAttribute("aria-label", "3D world viewer. Drag to look, scroll to zoom, use W A S D to move.");
    mount.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const group = new THREE.Group(); scene.add(group);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x4b6470, 2));
    let disposed = false, frame = 0, extent = 5;
    const cleanups: (() => void)[] = [];
    const keys = new Set<string>();
    const onKey = (event: KeyboardEvent) => {
      if ("wasdqe".includes(event.key.toLowerCase()) && event.key.length === 1) {
        event.preventDefault(); keys.add(event.key.toLowerCase());
      }
    };
    const offKey = (event: KeyboardEvent) => keys.delete(event.key.toLowerCase());
    const blur = () => keys.clear();
    const focus = () => renderer.domElement.focus({ preventScroll: true });
    renderer.domElement.addEventListener("keydown", onKey);
    renderer.domElement.addEventListener("keyup", offKey);
    renderer.domElement.addEventListener("blur", blur);
    renderer.domElement.addEventListener("pointerdown", focus);
    const box = new THREE.Box3();
    const fit = (inside: boolean) => {
      const center = box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
      if (inside || assets[0]?.kind === "image") {
        camera.position.copy(assets[0]?.kind === "image" ? new THREE.Vector3() : center);
        controls.target.copy(camera.position).add(new THREE.Vector3(0, 0, -1));
      } else {
        controls.target.copy(center);
        camera.position.copy(center).add(new THREE.Vector3(extent * .7, extent * .35, extent * .9));
      }
      controls.update();
    };
    reset.current = fit;
    const resize = () => {
      const width = mount.clientWidth, height = mount.clientHeight;
      if (!width || !height) return;
      renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize); observer.observe(mount); resize();
    const clock = new THREE.Clock();
    const animate = () => {
      if (disposed) return;
      const dt = Math.min(clock.getDelta(), .05);
      const forward = camera.getWorldDirection(new THREE.Vector3()); forward.y = 0; forward.normalize();
      const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
      const movement = new THREE.Vector3();
      if (keys.has("w")) movement.add(forward); if (keys.has("s")) movement.sub(forward);
      if (keys.has("d")) movement.add(right); if (keys.has("a")) movement.sub(right);
      if (keys.has("e")) movement.y += 1; if (keys.has("q")) movement.y -= 1;
      movement.multiplyScalar(dt * extent * .3); camera.position.add(movement); controls.target.add(movement);
      controls.update(); renderer.render(scene, camera); frame = requestAnimationFrame(animate);
    };
    animate();
    void (async () => {
      try {
        for (const asset of assets) {
          if (disposed) return;
          const url = worldAssetUrl(job, asset.path);
          if (asset.kind === "splat") {
            const { SparkRenderer, SplatMesh } = await import("@sparkjsdev/spark");
            if (disposed) return;
            const spark = new SparkRenderer({ renderer }); scene.add(spark); cleanups.push(() => spark.dispose());
            const splats = new SplatMesh({ url, withCredentials: true, lod: true }); group.add(splats); cleanups.push(() => splats.dispose());
            await splats.initialized;
            box.union(splats.getBoundingBox());
          } else if (asset.kind === "image") {
            const texture = await new THREE.TextureLoader().loadAsync(url);
            if (disposed) { texture.dispose(); return; }
            texture.colorSpace = THREE.SRGBColorSpace;
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(100, 64, 32), new THREE.MeshBasicMaterial({ map: texture, side: THREE.BackSide }));
            group.add(mesh); cleanups.push(() => texture.dispose());
          } else if (asset.path.toLowerCase().endsWith(".glb")) {
            const object = await new GLTFLoader().loadAsync(url);
            if (disposed) return;
            group.add(object.scene); box.expandByObject(object.scene);
          } else if (asset.path.toLowerCase().endsWith(".obj")) {
            const object = await new OBJLoader().loadAsync(url);
            if (disposed) return;
            group.add(object); box.expandByObject(object);
          } else if (asset.path.toLowerCase().endsWith(".ply")) {
            const geometry = await new PLYLoader().loadAsync(url);
            if (disposed) { geometry.dispose(); return; }
            geometry.computeVertexNormals();
            const colors = !!geometry.getAttribute("color");
            const object = asset.kind === "points"
              ? new THREE.Points(geometry, new THREE.PointsMaterial({ size: .025, vertexColors: colors, color: colors ? 0xffffff : 0x94c5be }))
              : new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ vertexColors: colors, color: colors ? 0xffffff : 0x94c5be, side: THREE.DoubleSide, roughness: 1 }));
            group.add(object); box.expandByObject(object);
          }
        }
        if (disposed) return;
        extent = Math.max(1, box.isEmpty() ? 5 : box.getSize(new THREE.Vector3()).length());
        camera.far = Math.max(10000, extent * 20); camera.updateProjectionMatrix();
        fit(assets[0]?.kind === "image"); setMessage("");
      } catch (err) { if (!disposed) setError(err instanceof Error ? err.message : "Could not load this scene"); }
    })();
    return () => {
      disposed = true; cancelAnimationFrame(frame); observer.disconnect(); controls.dispose();
      renderer.domElement.removeEventListener("keydown", onKey); renderer.domElement.removeEventListener("keyup", offKey);
      renderer.domElement.removeEventListener("blur", blur); renderer.domElement.removeEventListener("pointerdown", focus);
      cleanups.forEach(cleanup => cleanup());
      group.traverse(object => {
        const mesh = object as THREE.Mesh;
        mesh.geometry?.dispose();
        if (mesh.material) (Array.isArray(mesh.material) ? mesh.material : [mesh.material]).forEach(material => material.dispose());
      });
      renderer.dispose(); renderer.domElement.remove();
    };
  }, [job, signature]);
  return <div>
    <div className="relative h-[440px] overflow-hidden rounded-lg border border-line bg-[#11181c]" ref={host}>
      {(message || error) && <div className="absolute left-4 right-4 top-4 rounded-lg bg-bg/90 p-3 text-[13px]" role="status">{error || message}</div>}
    </div>
    <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
      <p className="text-[12px] text-fg-3">Drag to look · Scroll to zoom · WASD to move · Q/E for height</p>
      <div className="flex gap-2"><Button size="sm" variant="ghost" onClick={() => reset.current(false)}>Frame scene</Button><Button size="sm" variant="ghost" onClick={() => reset.current(true)}>Inside view</Button></div>
    </div>
  </div>;
}
