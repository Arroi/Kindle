import { useQuery } from "@tanstack/react-query";
import { ExternalLink, Sparkles } from "lucide-react";
import { Link, useNavigate, useParams } from "react-router";

import { ProgressBar } from "../components/meters";
import { Button, Empty } from "../components/ui";
import { api } from "../lib/api";
import { cx, MODALITY_LABEL } from "../lib/format";
import type { StudioModel } from "../lib/types";
import { MODALITY_ICON } from "./Launch";
import ChatStudio from "./studio/Chat";
import { ImageStudio, SttStudio, TtsStudio, VideoStudio } from "./studio/Media";
import WorldStudio from "./studio/World";

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
      if (model.engine === "diffusers_world") return <VideoStudio key={model.id} model={model} />;
      return <WorldStudio key={model.id} model={model} imageModels={imageModels} admin={admin} />;
    case "tts":
      return <TtsStudio key={model.id} model={model} />;
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

      {q.isLoading ? null : models.length === 0 ? (
        <Empty icon={<Sparkles className="size-8" />} title="No models running">
          {admin ? (
            <>
              Launch one and it shows up here.
              <div className="mt-5">
                <Link to="/">
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
              const Icon = MODALITY_ICON[m.modality];
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
                  <Icon className={cx("size-4", active ? "text-signal" : "text-fg-2")} />
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
            <Body
              model={current}
              admin={admin}
              imageModels={models.filter((m) => m.modality === "image" && m.status === "ready" && m.runtime !== "comfyui")}
            />
          )}
        </>
      )}
    </div>
  );
}
