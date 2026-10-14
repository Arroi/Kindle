import { ImageIcon, Paintbrush, Wand2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router";

import { Button, Field, Input, Segmented, Select, Slider, Textarea, useToast } from "../../components/ui";
import { gateway } from "../../lib/api";
import type { StudioModel } from "../../lib/types";
import { ActionBuilder, ClipList, fileToB64, ImagePick, JobProgress, runVideoJob, type Segment } from "./Media";

type Start = "text" | "describe" | "upload" | "default";

/**
 * World models (Oasis, Matrix-Game) are driven by a starting frame plus actions, not text.
 * "Describe it" paints that frame with a running image model first.
 */
export default function WorldStudio({ model, imageModels, admin }: { model: StudioModel; imageModels: StudioModel[]; admin: boolean }) {
  const o = model.options ?? {};
  const toast = useToast();
  const minecraft = model.controls === "minecraft";
  const canPaint = imageModels.length > 0;
  const [start, setStart] = useState<Start>(model.text_input ? "text" : canPaint ? "describe" : "default");
  const [prompt, setPrompt] = useState("");
  const [motionPrompt, setMotionPrompt] = useState("");
  const [painter, setPainter] = useState(imageModels[0]?.name ?? "");
  const [painted, setPainted] = useState<string | null>(null);
  const [painting, setPainting] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [frames, setFrames] = useState<number>(Number(o.frames ?? 32));
  const [steps, setSteps] = useState<number>(Number(o.steps ?? 10));
  const [seed, setSeed] = useState("");
  const [segments, setSegments] = useState<Segment[]>([]);
  const [chunks, setChunks] = useState<number>(Number(o.chunks ?? 1));
  const [pose, setPose] = useState<string>(String(o.pose ?? "w-4"));
  const [job, setJob] = useState<{ id: string; progress: number; message: string } | null>(null);
  const [videos, setVideos] = useState<{ url: string; prompt: string; frame?: string }[]>([]);
  const segFrames = segments.reduce((sum, x) => sum + x.frames, 0);

  useEffect(() => {
    if (minecraft && segFrames > 0) setFrames(segFrames);
  }, [minecraft, segFrames]);
  useEffect(() => {
    if (!imageModels.some((m) => m.name === painter)) setPainter(imageModels[0]?.name ?? "");
  }, [imageModels, painter]);
  useEffect(() => setPainted(null), [prompt, painter]);

  async function paint(): Promise<string> {
    setPainting(true);
    try {
      const res = await gateway.post("/v1/start-frames", {
        model: model.name, prompt, start_frame_model: painter || undefined, seed: seed ? Number(seed) : undefined,
      });
      const data = await res.json();
      const url = `data:image/png;base64,${data.b64_json}`;
      setPainted(url);
      return url;
    } finally {
      setPainting(false);
    }
  }

  async function generate() {
    try {
      const body: Record<string, unknown> = { model: model.name, frames, steps, seed: seed ? Number(seed) : undefined };
      if (model.prompt_conditioning) body.prompt = motionPrompt || "A coherent scene with natural motion.";
      if (start === "text") body.prompt = prompt;
      if (model.controls === "pose") { body.chunks = chunks; body.pose = pose; }
      if (minecraft) body.actions = segments.map(({ keys, camera, frames }) => ({ keys, camera, frames }));
      let frame: string | undefined;
      if (start === "describe") {
        frame = painted ?? (await paint());
        body.input_reference_b64 = frame.split(",", 2)[1];
      } else if (start === "upload" && file) {
        body.input_reference_b64 = await fileToB64(file);
        frame = URL.createObjectURL(file);
      }
      const label = start === "text" || start === "describe" ? prompt : minecraft ? segments.map((x) => x.label).join(", ") : motionPrompt;
      const url = await runVideoJob(body, setJob);
      setVideos((vs) => [{ url, prompt: label || "default start frame", frame }, ...vs]);
    } catch (e) {
      toast(e instanceof Error ? e.message : "Generation failed", "error");
    } finally {
      setJob(null);
    }
  }

  const blocked = start === "text" ? !prompt.trim() : start === "describe" ? !prompt.trim() || !canPaint : start === "upload" ? !file : false;

  return (
    <div className="grid gap-4 lg:grid-cols-[400px_minmax(0,1fr)]">
      <div className="panel space-y-4 p-4">
        <div>
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="readout">{model.text_input ? "World input" : "Starting frame"}</span>
            <Segmented<Start>
              size="sm"
              value={start}
              onChange={setStart}
              options={model.text_input ? [
                { value: "text", label: "Text" }, { value: "upload", label: "Image" },
              ] : [
                { value: "describe", label: "Describe it" },
                { value: "upload", label: "Upload" },
                { value: "default", label: "Default" },
              ]}
            />
          </div>
          {start === "text" && <Field label="Describe the scene"><Textarea rows={4} value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="First-person view walking through a quiet mountain village…" /></Field>}
          {start === "describe" &&
            (canPaint ? (
              <div className="space-y-2.5">
                <Textarea
                  rows={3}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder={minecraft ? "A snowy taiga with a frozen lake at sunset" : "A neon city street at night, wet asphalt"}
                />
                <div className="flex items-end gap-2">
                  {imageModels.length > 1 && (
                    <Field label="Paint with">
                      <Select value={painter} onChange={(e) => setPainter(e.target.value)} className="h-9">
                        {imageModels.map((m) => (
                          <option key={m.id}>{m.name}</option>
                        ))}
                      </Select>
                    </Field>
                  )}
                  <Button size="sm" onClick={() => paint().catch((e) => toast(e.message, "error"))} busy={painting} disabled={!prompt.trim() || !!job}>
                    <Paintbrush className="size-3.5" /> {painted ? "Repaint" : "Preview frame"}
                  </Button>
                </div>
                {painted && <img src={painted} alt="Starting frame" className="w-full rounded-lg border border-line" />}
                <p className="text-[12px] text-fg-3">
                  {model.name} doesn't read text; {painter || "your image model"} paints this first frame and the world model takes it from there.
                  {minecraft && " Oasis was trained on Minecraft, so Minecraft-looking scenes work best."}
                </p>
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-line-2 p-3.5 text-[13px] text-fg-2">
                <div className="mb-1 flex items-center gap-2 font-medium text-fg">
                  <ImageIcon className="size-4" /> No image model running
                </div>
                {model.name} starts from an image, not text. Deploy an image model too (SD Turbo is 2.6 GB and quick) and you can describe
                the starting scene here.
                {admin && (
                  <Link to="/deploy/new" className="mt-2 block font-medium text-signal underline underline-offset-4">
                    Launch an image model →
                  </Link>
                )}
              </div>
            ))}
          {start === "upload" && <ImagePick file={file} onFile={setFile} label="Image" />}
          {start === "default" && <p className="text-[12.5px] text-fg-3">Starts from the model's own sample frame.</p>}
        </div>

        {model.prompt_conditioning && start !== "text" && <Field label="Scene and motion prompt"><Textarea rows={3} value={motionPrompt} onChange={e => setMotionPrompt(e.target.value)} placeholder="Describe the scene and how it should move…" /></Field>}
        {minecraft && <ActionBuilder segments={segments} onChange={setSegments} />}
        {model.controls === "pose" ? <>
          <Field label="Video chunks" hint="Each chunk produces 16 frames"><Select value={chunks} onChange={e => { const n = Number(e.target.value); setChunks(n); if (/^w-\d+$/.test(pose)) setPose(`w-${n*4}`); }}>{[1,2,4,8,16].map(n => <option key={n} value={n}>{n} · {n*16} frames</option>)}</Select></Field>
          <Field label="Camera actions" hint={`Movement: w/a/s/d. Rotation: up/down/left/right. Durations must total ${chunks*4}.`}><Input value={pose} onChange={e => setPose(e.target.value)} placeholder={`w-${chunks*4}`} /></Field>
        </> : <Slider label="Frames" value={frames} min={4} max={300} onChange={setFrames} />}
        <Slider label="Steps" value={steps} min={1} max={60} onChange={setSteps} />
        <Field label="Seed">
          <Input value={seed} onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))} placeholder="random" />
        </Field>
        <Button variant="primary" size="lg" className="w-full" busy={!!job || painting} onClick={generate} disabled={!!job || painting || blocked}>
          <Wand2 className="size-4" /> Generate
        </Button>
        {job && <JobProgress job={job} />}
        <p className="text-[12px] text-fg-3">World models run the upstream inference script per clip, so expect a minute or more.</p>
      </div>
      <ClipList videos={videos} placeholder={model.name} />
    </div>
  );
}
