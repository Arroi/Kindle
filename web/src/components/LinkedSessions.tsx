import { Link2, Trash2 } from "lucide-react";
import { useState } from "react";

import { api } from "../lib/api";
import { useLive } from "../lib/live";
import type { LinkedSession } from "../lib/types";
import { Button, Chip, Field, Input, useToast } from "./ui";

const STATUS: Record<LinkedSession["status"], [string, string]> = {
  up: ["dot dot-live", "Connected"],
  connecting: ["dot dot-warn", "Connecting"],
  down: ["dot dot-warn", "Unreachable"],
  mismatch: ["dot dot-warn", "GPUs don't match"],
};

/** Pool another KaggleUI session's models behind this one's URL and API key. */
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
      <p className="text-[13px] leading-snug text-fg-2">
        Link another KaggleUI session that has the <b className="text-fg">same GPUs</b> as this one ({links.gpu_label}). Its running models
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
                {l.gpu_label && <div className="mt-1 text-fg-3">{l.gpu_label}{l.version ? ` · KaggleUI ${l.version}` : ""}</div>}
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
