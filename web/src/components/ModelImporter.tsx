import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../lib/api";
import { bytes } from "../lib/format";
import type { CatalogItem } from "../lib/types";
import { ProgressBar } from "./meters";
import { Button, Field, Input, Segmented, Select, Toggle, useToast } from "./ui";

const ACTIVE = ["queued", "downloading", "extracting", "scanning", "abliterating"];
const KIND_LABEL: Record<string, string> = { hf: "Hugging Face", git: "Git", file: "File", abliterate: "Abliteration" };

/** Remove a text model's refusal behaviour and save the result as an import. */
export function Abliterator({ initialRepo = "" }: { initialRepo?: string }) {
  const [repo, setRepo] = useState(initialRepo);
  const [method, setMethod] = useState<"quick" | "heretic">("quick");
  const [trials, setTrials] = useState(40);
  const [quantize, setQuantize] = useState(false);
  const [busy, setBusy] = useState(false);
  const toast = useToast();
  const cache = useQueryClient();
  return <form className="panel space-y-4 p-5" onSubmit={async (event) => {
    event.preventDefault();
    setBusy(true);
    try {
      await api.abliterate({ repo, method, trials, quantize });
      await cache.invalidateQueries({ queryKey: ["model-imports"] });
      toast("Abliteration started. It appears under Downloads below.");
    } catch (error) { toast(error instanceof Error ? error.message : "Couldn't start abliteration", "error"); }
    finally { setBusy(false); }
  }}>
    <div>
      <h3 className="text-[16px] font-semibold">Abliterate a model</h3>
      <p className="mt-1 max-w-[65ch] text-[13px] text-fg-2">Removes a text model's refusal behaviour by projecting its refusal direction out of the weights, then saves the result here so you can deploy it. Works on Hugging Face chat models in safetensors format, not GGUF/Ollama, image or video models. It uses the GPUs, so stop other deployments first.</p>
    </div>
    <Field label="Hugging Face model"><Input required value={repo} onChange={e => setRepo(e.target.value)} placeholder="Qwen/Qwen3-4B-Instruct-2507" /></Field>
    {/* Not a <Field>: a <label> around buttons forwards clicks to the first one. */}
    <div role="group" aria-labelledby="abliterate-method">
      <span id="abliterate-method" className="mb-1.5 block text-[13px] font-medium">Method</span>
      <Segmented value={method} onChange={v => setMethod(v as "quick" | "heretic")} options={[
        { value: "quick", label: "Quick" },
        { value: "heretic", label: "Thorough (Heretic)" },
      ]} />
    </div>
    <p className="text-[12.5px] text-fg-3">{method === "quick"
      ? "Measures the refusal direction from 128 prompt pairs, tests a few layers and edits the weights. About 5–15 minutes for a 4–8B model."
      : "Heretic searches for the edit that removes the most refusals while changing the model least. Each trial takes a minute or more on a T4, so plan on an hour or longer; its environment installs on first use."}</p>
    <div className="grid gap-3 sm:grid-cols-2">
      {method === "heretic" && <Field label="Trials" hint="More trials find better edits but take longer (Heretic's default is 100).">
        <Input type="number" min={10} max={300} value={trials} onChange={e => setTrials(Number(e.target.value) || 40)} />
      </Field>}
      <div className="sm:col-span-2"><Toggle checked={quantize} onChange={setQuantize} label="Load in 4-bit while analysing" hint="For models too big for the GPUs at 16-bit. The saved model keeps full precision." /></div>
    </div>
    <Button type="submit" variant="primary" busy={busy} disabled={!repo.trim()}>Start abliteration</Button>
  </form>;
}

export function ModelImporter({ initialUrl = "", onCustomize }: { initialUrl?: string; onCustomize: (item: CatalogItem) => void }) {
  const [url, setUrl] = useState(initialUrl);
  const [kind, setKind] = useState("auto");
  const [name, setName] = useState("");
  const [revision, setRevision] = useState("");
  const [busy, setBusy] = useState(false);
  const toast = useToast();
  const cache = useQueryClient();
  const [customizing, setCustomizing] = useState<string | null>(null);
  const query = useQuery({ queryKey: ["model-imports"], queryFn: api.modelImports,
    refetchInterval: q => q.state.data?.some(j => ACTIVE.includes(j.status)) ? 2000 : false });

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
      <p className="max-w-[65ch] text-[14px] text-fg-2">Bring a checkpoint from Hugging Face, a Git repository, or a direct file URL. Once it downloads, choose Customize to configure and deploy it.</p>
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
        const active = ACTIVE.includes(job.status);
        return <div key={job.id} className="border-t border-line pt-3">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0"><h4 className="break-words text-[14px] font-medium">{job.name}</h4><p className="mt-1 text-[12px] text-fg-3">{KIND_LABEL[job.kind] ?? job.kind} · {job.status} · {bytes(job.bytes)}</p></div>
            {active && <Button size="sm" variant="ghost" onClick={async () => {
              try { await api.cancelImport(job.id); await cache.invalidateQueries({ queryKey: ["model-imports"] }); }
              catch (error) { toast(error instanceof Error ? error.message : "Cancel failed", "error"); }
            }}>Cancel</Button>}
          </div>
          {active && <div className="my-2"><ProgressBar value={job.progress} /></div>}
          <p className={`mt-2 break-words text-[12px] ${job.status === "error" ? "text-worst" : "text-fg-2"}`} role="status">{job.message}</p>
          {job.status === "complete" && job.models.map((model, index) => <div key={model.path} className="mt-3 flex items-center justify-between gap-3 rounded-lg bg-panel-2 px-3 py-2.5">
            <div className="min-w-0"><span className="block truncate text-[13px] font-medium">{model.name}</span><span className="text-[12px] text-fg-3">{model.modality} · {model.engine || "No compatible runner detected"}</span></div>
            {model.engine && <Button size="sm" busy={customizing === `${job.id}:${index}`} onClick={async () => {
              setCustomizing(`${job.id}:${index}`);
              try { onCustomize(await api.customizeImport(job.id, index)); }
              catch (error) { toast(error instanceof Error ? error.message : "Couldn't read checkpoint", "error"); }
              finally { setCustomizing(null); }
            }}>Customize →</Button>}
          </div>)}
        </div>;
      })}
    </section>
  </div>;
}
