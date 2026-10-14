import { useEffect, useRef, useState } from "react";

import { Button, Field, Input, Segmented, Select, Textarea, useToast } from "../../components/ui";
import { WorldSceneViewer, worldAssetUrl, type WorldAsset } from "../../components/WorldSceneViewer";
import { gateway } from "../../lib/api";
import { bytes } from "../../lib/format";
import type { StudioModel } from "../../lib/types";
import { fileToB64, ImagePick, JobProgress } from "./Media";

type Job = { id: string; model: string; prompt?: string; status: string; progress: number; message?: string; error?: string };
type InputMode = "text" | "image" | "panorama";
function meshLayers(files: WorldAsset[]) {
  const all = files.filter(file => /mesh_layer\d+\.ply$/.test(file.path));
  const previews = all.filter(file => file.path.startsWith("preview/"));
  return previews.length ? previews : all;
}

export default function WorldSceneStudio({ model }: { model: StudioModel }) {
  const [mode, setMode] = useState<InputMode>("text");
  const [prompt, setPrompt] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [seed, setSeed] = useState("");
  const [sceneClass, setSceneClass] = useState("outdoor");
  const [labels1, setLabels1] = useState(""); const [labels2, setLabels2] = useState("");
  const [vlm, setVlm] = useState(""); const [vlmModel, setVlmModel] = useState("Qwen/Qwen3-VL-8B-Instruct"); const [vlmKey, setVlmKey] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]); const [selected, setSelected] = useState("");
  const [assets, setAssets] = useState<WorldAsset[]>([]); const [asset, setAsset] = useState("");
  const [busy, setBusy] = useState(false); const toast = useToast();
  const [connectionError, setConnectionError] = useState("");
  const poll = useRef(0);
  const active = jobs.find(job => ["queued", "in_progress"].includes(job.status));
  useEffect(() => {
    let alive = true;
    const update = async () => {
      try {
        const list = await (await gateway.get("/v1/worlds")).json();
        const mine: Job[] = list.data.filter((job: Job) => job.model === model.name);
        const current = await Promise.all(mine.map(async job => {
          if (!["queued", "in_progress"].includes(job.status)) return job;
          return await (await gateway.get(`/v1/worlds/${job.id}`)).json() as Job;
        }));
        if (alive) { setJobs(current); setConnectionError(""); }
      } catch (error) { if (alive) setConnectionError(error instanceof Error ? error.message : "Couldn't refresh generation progress"); }
      if (alive) poll.current = window.setTimeout(update, 1500);
    };
    void update();
    return () => { alive = false; window.clearTimeout(poll.current); };
  }, [model.name]);
  useEffect(() => {
    const completed = jobs.find(job => job.status === "completed");
    if (!selected && completed) setSelected(completed.id);
  }, [jobs, selected]);
  useEffect(() => {
    let alive = true;
    setAssets([]); setAsset("");
    if (selected) void gateway.get(`/v1/worlds/${selected}/assets`).then(reply => reply.json()).then(data => {
      if (!alive) return;
      const files = data.assets as WorldAsset[]; setAssets(files);
      const layers = meshLayers(files);
      const first = files.find(file => file.kind === "splat") ?? files.find(file => file.kind === "mesh") ?? files.find(file => file.path.endsWith("panorama.png"));
      setAsset(layers.length ? "layers" : first?.path ?? "");
    }).catch(err => { if (alive) toast(err.message, "error"); });
    return () => { alive = false; };
  }, [selected]);
  const picked = assets.find(file => file.path === asset);
  const viewer = asset === "layers" ? meshLayers(assets) : picked ? [picked] : [];
  return <div className="grid gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
    <form className="panel space-y-4 p-5" onSubmit={async event => {
      event.preventDefault(); setBusy(true);
      try {
        const body = { model: model.name, prompt, input_mode: mode, input_reference_b64: file && mode !== "text" ? await fileToB64(file) : undefined,
          seed: seed ? Number(seed) : 42, scene_class: sceneClass, labels_fg1: labels1, labels_fg2: labels2,
          vlm_url: vlm, vlm_model: vlmModel, vlm_key: vlmKey || undefined };
        const job = await (await gateway.post("/v1/worlds", body)).json() as Job;
        setJobs(old => [job, ...old]); toast("World generation started");
      } catch (err) { toast(err instanceof Error ? err.message : "Generation failed", "error"); }
      finally { setBusy(false); }
    }}>
      <div><h2 className="text-[17px] font-semibold">Build a 3D world</h2><p className="mt-1 text-[12px] text-fg-3">Generates a panorama and a downloadable 3D scene. These pipelines can take a long time.</p></div>
      <Segmented<InputMode> value={mode} onChange={setMode} options={[{value:"text",label:"Text"},{value:"image",label:"Image"},{value:"panorama",label:"Panorama"}]} />
      <Field label={mode === "text" ? "Describe your world" : "Scene description (optional)"}><Textarea rows={4} value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="A quiet coastal village with narrow streets, stone houses and a bright blue sea…" /></Field>
      {mode !== "text" && <ImagePick file={file} onFile={setFile} label={mode === "panorama" ? "360° equirectangular panorama" : "Starting image"} />}
      <Field label="Seed"><Input value={seed} onChange={e => setSeed(e.target.value.replace(/\D/g,""))} placeholder="42" /></Field>
      {!model.needs_vlm && <>
        <Field label="Scene type"><Select value={sceneClass} onChange={e => setSceneClass(e.target.value)}><option value="outdoor">Outdoor</option><option value="indoor">Indoor</option></Select></Field>
        <Field label="Foreground layer 1 (optional)" hint="Object names separated by commas"><Input value={labels1} onChange={e => setLabels1(e.target.value)} placeholder="stones, flowers" /></Field>
        <Field label="Foreground layer 2 (optional)"><Input value={labels2} onChange={e => setLabels2(e.target.value)} placeholder="trees, buildings" /></Field>
      </>}
      {model.needs_vlm && <div className="space-y-3 border-t border-line pt-4">
        <p className="text-[12px] text-fg-2">HY-World 2.0 uses a vision-language server to plan routes and describe views. Provide a server that supports image inputs.</p>
        <Field label="Vision-language API URL"><Input type="url" required value={vlm} onChange={e => setVlm(e.target.value)} placeholder="http://server:8000/v1" /></Field>
        <Field label="Vision-language model"><Input required value={vlmModel} onChange={e => setVlmModel(e.target.value)} /></Field>
        <Field label="API key (optional)"><Input type="password" value={vlmKey} onChange={e => setVlmKey(e.target.value)} autoComplete="off" /></Field>
      </div>}
      <Button type="submit" variant="primary" className="w-full" busy={busy || !!active} disabled={mode === "text" ? !prompt.trim() : !file}>Generate world</Button>
      {active && <JobProgress job={{ progress: active.progress / 100, message: active.message || active.status }} />}
      {connectionError && <p role="alert" className="text-[12px] text-tight">{connectionError} · retrying…</p>}
      {jobs.find(job => job.status === "failed") && <p className="break-words text-[12px] text-worst" role="alert">{jobs.find(job => job.status === "failed")?.error}</p>}
    </form>
    <div className="panel min-w-0 space-y-4 p-5">
      {!selected ? <div className="grid min-h-[440px] place-items-center text-center text-[13px] text-fg-3">Your generated world and its downloads will appear here.</div> : <>
        <div className="flex flex-wrap items-center gap-3">
          <Select aria-label="Generated world" value={selected} onChange={e => setSelected(e.target.value)} className="min-w-0 flex-1">{jobs.filter(job => job.status === "completed").map(job => <option key={job.id} value={job.id}>{job.prompt || job.id}</option>)}</Select>
          <a className="text-[13px] font-medium text-signal hover:underline" href={`/_kui/api/studio/worlds/${selected}/content`} download>Download world ZIP</a>
        </div>
        <Select aria-label="Scene asset" value={asset} onChange={e => setAsset(e.target.value)}>
          {assets.some(file => /mesh_layer\d+\.ply$/.test(file.path)) && <option value="layers">All 3D mesh layers</option>}
          {assets.filter(file => file.kind !== "data").map(file => <option key={file.path} value={file.path}>{file.path} · {bytes(file.size)}</option>)}
        </Select>
        {picked?.kind === "video" ? <video src={worldAssetUrl(selected, picked.path)} controls className="w-full rounded-lg" /> :
          picked?.kind === "image" && !picked.path.includes("panorama") ? <img src={worldAssetUrl(selected, picked.path)} alt="Generated scene output" className="max-h-[440px] w-full object-contain" /> :
          viewer.length ? <WorldSceneViewer key={`${selected}:${asset}`} job={selected} assets={viewer} /> : null}
        <details className="border-t border-line pt-3"><summary className="cursor-pointer text-[13px] text-fg-2">Exported files ({assets.length})</summary><div className="mt-3 max-h-48 space-y-1 overflow-auto">{assets.map(file => <a key={file.path} href={worldAssetUrl(selected,file.path)} download className="flex justify-between gap-3 break-all text-[12px] text-fg-2 hover:text-signal"><span>{file.path}</span><span className="shrink-0 text-fg-3">{bytes(file.size)}</span></a>)}</div></details>
      </>}
    </div>
  </div>;
}
