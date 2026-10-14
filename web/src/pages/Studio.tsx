import { useQuery } from "@tanstack/react-query";
import { ExternalLink, Sparkles } from "lucide-react";
import { lazy, Suspense } from "react";
import { Link, useNavigate, useParams } from "react-router";

import { ProgressBar } from "../components/meters";
import { Button, Empty } from "../components/ui";
import { api } from "../lib/api";
import { ModelLogo } from "../components/ModelLogo";
import { cx, MODALITY_LABEL } from "../lib/format";
import type { StudioModel } from "../lib/types";
import ChatStudio from "./studio/Chat";
const ImageStudio = lazy(() => import("./studio/Media").then(m => ({ default: m.ImageStudio })));
const SttStudio = lazy(() => import("./studio/Media").then(m => ({ default: m.SttStudio })));
const TtsStudio = lazy(() => import("./studio/Media").then(m => ({ default: m.TtsStudio })));
const VideoStudio = lazy(() => import("./studio/Media").then(m => ({ default: m.VideoStudio })));
const WorldStudio = lazy(() => import("./studio/World"));
const WorldSceneStudio = lazy(() => import("./studio/WorldScene"));

function Body({ model, imageModels, admin }: { model: StudioModel; imageModels: StudioModel[]; admin: boolean }) {
  if (model.status !== "ready") {
    return (
      <div className="panel p-6">
        <div className="mb-3 text-[14px]">{model.status === "error" ? "This deployment failed." : model.message}</div>
        {model.status !== "error" && <ProgressBar value={model.progress} />}
      </div>
    );
  }
  if (model.runtime === "comfyui") {
    return (
      <Empty title="This one runs in ComfyUI">
        Open it from the link above. ComfyUI replaces the built-in image engine.
      </Empty>
    );
  }
  switch (model.modality) {
    case "text":
      return <ChatStudio key={model.id} model={model} />;
    case "image":
      return <ImageStudio key={model.id} model={model} />;
    case "video":
      return <VideoStudio key={model.id} model={model} />;
    case "world":
      if (model.result_kind === "scene") return <WorldSceneStudio key={model.id} model={model} />;
      if (model.engine === "diffusers_world") return <VideoStudio key={model.id} model={model} />;
      return <WorldStudio key={model.id} model={model} imageModels={imageModels} admin={admin} />;
    case "tts":
    case "audio":
      return <TtsStudio key={model.id} model={model} />;
    case "mesh":
      return <Empty title="3D checkpoint">Use model details to check this checkpoint’s available runner and downloads.</Empty>;
    case "stt":
      return <SttStudio key={model.id} model={model} />;
  }
}

export default function Studio({ admin }: { admin: boolean }) {
  const { id } = useParams();
  const nav = useNavigate();
  const q = useQuery({ queryKey: ["studio-models"], queryFn: api.studioModels, refetchInterval: 2500 });
  const models = (q.data ?? []).filter((m) => m.status !== "stopped");
  const current = models.find((m) => m.id === id) ?? models.find((m) => m.status === "ready") ?? models[0];
  const ifaceUrl = current?.interface?.status === "up" ? current.interface.url : null;
  const ifaceName = current?.interface?.kind === "openwebui" ? "Open WebUI" : current?.interface?.kind === "comfyui" ? "ComfyUI" : current?.interface?.kind === "gradio" ? "Gradio" : null;

  return (
    <div>
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="readout mb-2">Workspace / Studio</div>
          <h1 className="display text-[28px] sm:text-[34px]">Studio</h1>
        </div>
        {ifaceUrl && ifaceName && (
          <a href={ifaceUrl} target="_blank" rel="noreferrer">
            <Button>
              Open {ifaceName} <ExternalLink className="size-3.5" />
            </Button>
          </a>
        )}
      </div>

      {q.error ? <Empty title="Couldn't load the Studio">{q.error.message}<Button className="mt-4" onClick={() => q.refetch()}>Try again</Button></Empty> : q.isLoading ? <div className="panel h-48 animate-pulse" aria-label="Loading Studio" /> : models.length === 0 ? (
        <Empty icon={<Sparkles className="size-8" />} title="No models running">
          {admin ? (
            <>
              Launch one and it shows up here.
              <div className="mt-5">
                <Link to="/deploy/new">
                  <Button variant="primary">Launch a model</Button>
                </Link>
              </div>
            </>
          ) : (
            "The host hasn't started any models yet. This page updates on its own."
          )}
        </Empty>
      ) : (
        <>
          <div className="mb-5 flex gap-2 overflow-x-auto pb-1 scroll-thin">
            {models.map((m) => {
              const active = m.id === current?.id;
              return (
                <button
                  key={m.id}
                  onClick={() => nav(`/studio/${m.id}`)}
                  className={cx(
                    "flex shrink-0 items-center gap-2.5 rounded-xl border px-3.5 py-2.5 text-left transition-colors",
                    active ? "border-signal bg-signal-soft" : "border-line bg-panel hover:border-line-2",
                  )}
                >
                  <ModelLogo id={m.model_id ?? ""} name={m.name} source={m.source} className="size-8" />
                  <span>
                    <span className="block text-[14px] font-medium leading-tight">{m.name}</span>
                    <span className="block font-mono text-[10px] uppercase tracking-wider text-fg-3">
                      {MODALITY_LABEL[m.modality]} · {m.status}
                    </span>
                  </span>
                  <span className={cx("ml-1 size-2 rounded-full", m.status === "ready" ? "bg-signal" : m.status === "error" ? "bg-worst" : "bg-tight animate-blink")} />
                </button>
              );
            })}
          </div>
          {current && (
              <Suspense fallback={<div className="panel h-48 animate-pulse" aria-label="Loading workspace" />}><Body
              model={current}
              admin={admin}
              imageModels={models.filter((m) => m.modality === "image" && m.status === "ready" && m.runtime !== "comfyui")}
            /></Suspense>
          )}
        </>
      )}
    </div>
  );
}
