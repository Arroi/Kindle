import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { api } from "../lib/api";
import { bytes } from "../lib/format";
import type { Modality } from "../lib/types";
import { ProgressBar } from "./meters";
import { Button, Field, Input, Select, useToast } from "./ui";

export function ModelImporter({ initialUrl = "", onBrowse }: { initialUrl?: string; onBrowse: (modality: Modality) => void }) {
  const [url, setUrl] = useState(initialUrl);
  const [kind, setKind] = useState("auto");
  const [name, setName] = useState("");
  const [revision, setRevision] = useState("");
  const [busy, setBusy] = useState(false);
  const toast = useToast();
  const cache = useQueryClient();
  const completed = useRef(new Set<string>());
  const query = useQuery({ queryKey: ["model-imports"], queryFn: api.modelImports, refetchInterval: 1500 });
  useEffect(() => {
    for (const job of query.data ?? []) {
      if (job.status === "complete" && !completed.current.has(job.id)) {
        completed.current.add(job.id);
        void cache.invalidateQueries({ queryKey: ["search"] });
      }
    }
  }, [query.data, cache]);

  return <div className="space-y-6">
    <form className="space-y-4" onSubmit={async (event) => {
      event.preventDefault();
      setBusy(true);
      try {
        await api.importModel({ url, kind, name, revision: revision || undefined });
        await cache.invalidateQueries({ queryKey: ["model-imports"] });
        setUrl("");
        toast("Model download started");
      } catch (error) { toast(error instanceof Error ? error.message : "Download failed", "error"); }
      finally { setBusy(false); }
    }}>
      <p className="text-[13px] text-fg-2">Paste a Hugging Face model, Git repository, direct file or archive URL. Recognized formats appear in Local models when the download finishes.</p>
      <Field label="Model URL"><Input required type="url" value={url} onChange={e => setUrl(e.target.value)} placeholder="https://huggingface.co/owner/model" /></Field>
      <Field label="Download type"><Select value={kind} onChange={e => setKind(e.target.value)}>
        <option value="auto">Detect automatically</option><option value="hf">Hugging Face repository</option>
        <option value="git">Git repository</option><option value="file">Direct file / archive</option>
      </Select></Field>
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Name (optional)"><Input value={name} onChange={e => setName(e.target.value)} placeholder="My model" /></Field>
        <Field label="Revision / branch (optional)"><Input value={revision} onChange={e => setRevision(e.target.value)} placeholder="Default revision" /></Field>
      </div>
      <p className="text-[12px] text-fg-3">Gated Hugging Face models use your saved HF token. Git LFS repositories need git-lfs installed. Downloading source code does not automatically create a model runner.</p>
      <Button type="submit" variant="primary" busy={busy} disabled={!url.trim()}>Download model</Button>
    </form>
    <section className="space-y-3" aria-label="Model downloads">
      <h3 className="text-[15px] font-semibold">Downloads</h3>
      {query.error && <p role="alert" className="text-[13px] text-worst">{String(query.error.message)}</p>}
      {query.isLoading && <p className="text-[13px] text-fg-3">Loading downloads…</p>}
      {query.data?.length === 0 && <p className="text-[13px] text-fg-3">Your downloads will appear here.</p>}
      {query.data?.map(job => {
        const active = ["queued", "downloading", "extracting", "scanning"].includes(job.status);
        const runnable = job.models.find(m => m.engine);
        return <div key={job.id} className="border-t border-line pt-3">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0"><h4 className="break-words text-[14px] font-medium">{job.name}</h4><p className="mt-1 text-[12px] text-fg-3">{job.kind} · {job.status} · {bytes(job.bytes)}</p></div>
            {active && <Button size="sm" variant="ghost" onClick={async () => {
              try { await api.cancelImport(job.id); await cache.invalidateQueries({ queryKey: ["model-imports"] }); }
              catch (error) { toast(error instanceof Error ? error.message : "Cancel failed", "error"); }
            }}>Cancel</Button>}
          </div>
          {active && <div className="my-2"><ProgressBar value={job.progress} /></div>}
          <p className={`mt-2 break-words text-[12px] ${job.status === "error" ? "text-worst" : "text-fg-2"}`} role="status">{job.message}</p>
          {runnable && <Button size="sm" variant="ghost" className="mt-2" onClick={() => onBrowse(runnable.modality)}>Browse local models →</Button>}
        </div>;
      })}
    </section>
  </div>;
}
