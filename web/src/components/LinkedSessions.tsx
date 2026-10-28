import { Cpu, Link2, Trash2 } from "lucide-react";
import { useState } from "react";

import { api } from "../lib/api";
import { bytes } from "../lib/format";
import { useLive } from "../lib/live";
import type { LinkedSession, PoolInfo } from "../lib/types";
import { Button, Chip, Field, Input, Toggle, useToast } from "./ui";

const POOL_STATUS: Record<PoolInfo["status"], string> = {
  idle: "Waiting for a head session",
  installing: "Installing llama.cpp",
  starting: "Starting the RPC server",
  seeding: "Downloading its share of the weights",
  ready: "In the pool",
  error: "Error",
};

/** Lend this session's GPU to a head session that runs one model too big for a single GPU. */
function ShareGpu() {
  const { overview, dispatch } = useLive();
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  if (!overview) return null;
  const pool = overview.pool;
  const on = overview.settings.pool?.share_gpu ?? false;

  async function toggle(value: boolean) {
    setBusy(true);
    try {
      dispatch({ type: "settings", data: await api.updateSettings({ pool: { share_gpu: value } }) });
    } catch (e) {
      toast(e instanceof Error ? e.message : "Couldn't change GPU sharing", "error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-lg border border-line bg-bg-2 px-3 py-3">
      <Toggle checked={on} disabled={busy} onChange={toggle}
        label={<span className="flex items-center gap-2"><Cpu className="size-3.5" /> Share this GPU with a pool</span>}
        hint="For models bigger than one GPU. Link this session from the session that will run the model, then deploy it there with the llama.cpp (GPU pool) engine. Each token makes a round trip over the public link, so pooled models are slower than single-session ones." />
      {on && pool && (
        <div className="mt-3 space-y-1 border-t border-line pt-2 text-[12.5px]">
          <div className="flex flex-wrap gap-x-3 text-fg-2">
            <span>{POOL_STATUS[pool.status]}</span>
            {pool.gpus.map((g, i) => <span key={i} className="text-fg-3">{g.name} · {bytes(g.vram_free)} free</span>)}
          </div>
          {pool.message && pool.status !== "idle" && <div className="text-fg-3">{pool.message}</div>}
          {pool.progress != null && pool.status === "seeding" && (
            <div className="h-1.5 overflow-hidden rounded-full bg-panel-3"><div className="h-full bg-signal" style={{ width: `${Math.round(pool.progress * 100)}%` }} /></div>
          )}
          {pool.error && <div className="text-worst">{pool.error}</div>}
        </div>
      )}
    </div>
  );
}

const STATUS: Record<LinkedSession["status"], [string, string]> = {
  up: ["dot dot-live", "Connected"],
  connecting: ["dot dot-warn", "Connecting"],
  down: ["dot dot-warn", "Unreachable"],
  mismatch: ["dot dot-warn", "GPUs don't match"],
};

/** Pool another Kindle session's models behind this one's URL and API key. */
export function LinkedSessions() {
  const { overview, dispatch } = useLive();
  const toast = useToast();
  const [url, setUrl] = useState("");
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const links = overview?.links;
  if (!links) return null;

  async function add() {
    setBusy(true);
    try {
      dispatch({ type: "links", data: await api.addLink(url, key) });
      setUrl("");
      setKey("");
      toast("Session linked");
    } catch (e) {
      toast(e instanceof Error ? e.message : "Couldn't link that session", "error");
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    try {
      dispatch({ type: "links", data: await api.removeLink(id) });
    } catch (e) {
      toast(e instanceof Error ? e.message : "Couldn't unlink", "error");
    }
  }

  return (
    <div className="space-y-4">
      <ShareGpu />
      <p className="text-[13px] leading-snug text-fg-2">
        Link another Kindle session that has the <b className="text-fg">same GPUs</b> as this one ({links.gpu_label}). Its running models
        appear here under this URL and API key. When both sessions run the same model, requests alternate between them. Video and world
        jobs stay on the session that runs them.
      </p>
      {links.can_link ? (
        <form
          className="grid gap-3"
          onSubmit={(e) => {
            e.preventDefault();
            if (url.trim()) add();
          }}
        >
          <Field label="Other session's link">
            <Input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://….gradio.live" autoComplete="off" />
          </Field>
          <Field label="Its API key" hint="From the other session's card or its Access & API page.">
            <Input value={key} onChange={(e) => setKey(e.target.value)} placeholder="kui-…" type="password" autoComplete="off" />
          </Field>
          <div>
            <Button type="submit" variant="primary" size="sm" busy={busy} disabled={!url.trim()}>
              <Link2 className="size-3.5" /> Link session
            </Button>
          </div>
        </form>
      ) : (
        <p className="text-[13px] text-fg-3">This session has no GPU, so it can't be linked.</p>
      )}
      {links.links.length > 0 && (
        <div className="space-y-2">
          {links.links.map((l) => {
            const [dot, label] = STATUS[l.status];
            return (
              <div key={l.id} className="rounded-lg border border-line bg-bg-2 px-3 py-2.5 text-[12.5px]">
                <div className="flex flex-wrap items-center gap-2">
                  <span className={dot} />
                  <span className="min-w-0 flex-1 truncate font-mono text-fg-2">{l.url}</span>
                  <span className="text-fg-3">{label}</span>
                  <Button size="sm" variant="ghost" onClick={() => remove(l.id)} aria-label="Unlink">
                    <Trash2 className="size-3.5" />
                  </Button>
                </div>
                {l.gpu_label && <div className="mt-1 text-fg-3">{l.gpu_label}{l.version ? ` · Kindle ${l.version}` : ""}</div>}
                {l.pool?.enabled && (
                  <div className="mt-1 flex items-center gap-1.5 text-fg-2">
                    <Cpu className="size-3" /> Shares its GPU{l.pool.status && l.pool.status !== "idle" ? ` · ${POOL_STATUS[l.pool.status]}` : " · available to the pool"}
                  </div>
                )}
                {l.error && <div className="mt-1 text-worst">{l.error}</div>}
                {l.models.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {l.models.map((m) => (
                      <Chip key={`${m.modality}:${m.name}`}>{m.name}</Chip>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
