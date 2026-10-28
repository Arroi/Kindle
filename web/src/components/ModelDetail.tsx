import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { AlertTriangle, ExternalLink, Loader } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";

import { api } from "../lib/api";
import { bytes, cx, ENGINE_LABEL, memoryPool, params, tokens } from "../lib/format";
import { useDebounced } from "../lib/hooks";
import { useLive } from "../lib/live";
import type { CatalogItem, DeploymentSpec, Variant } from "../lib/types";
import { FitGauge } from "./meters";
import { ModelLogo } from "./ModelLogo";
import { CapabilityTags } from "./CapabilityTags";
import { effortLabel } from "../lib/capabilities";
import { Button, Field, Input, Select, Toggle, VerdictTag } from "./ui";

export interface ModelChoice {
  item: CatalogItem;
  spec: DeploymentSpec;
}

/** Settings that change a model's size or fit, re-checked by the server as they change. */
const FIT_OPTIONS = ["ctx", "offload", "quantize", "weights"];

const K = 1024;
const CTX_MIN = 2 * K;
/** Upper limit when the catalog doesn't know the model's window. */
const CTX_UNKNOWN_MAX = 64 * K;

/** Any context size from 2K up to the model's window: drag the slider or type the size in K. */
function ContextLength({ value, max, onChange }: { value: number; max: number; onChange: (v: number) => void }) {
  const clamp = (v: number) => Math.min(max, Math.max(CTX_MIN, Math.round(v)));
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <div className="flex items-center gap-3">
      <input
        type="range"
        min={CTX_MIN}
        max={max}
        step={K}
        value={value}
        onChange={(e) => onChange(clamp(Number(e.target.value)))}
        className="min-w-0 flex-1 accent-[var(--fg)]"
        aria-label="Context length"
      />
      <div className="relative w-[92px] shrink-0">
        <Input
          type="number"
          inputMode="numeric"
          min={CTX_MIN / K}
          max={Math.floor(max / K)}
          value={draft ?? String(Math.round(value / K))}
          onChange={(e) => {
            setDraft(e.target.value);
            const k = Number(e.target.value);
            if (e.target.value && k * K >= CTX_MIN && k * K <= max) onChange(k * K);
          }}
          onBlur={() => {
            if (draft) onChange(clamp(Number(draft) * K));
            setDraft(null);
          }}
          className="pr-7 text-right font-mono"
          aria-label="Context length in K tokens"
        />
        <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-[12px] text-fg-3">K</span>
      </div>
    </div>
  );
}

/** llama.cpp: pool linked sessions' GPUs, parallel request slots, MoE experts kept in RAM. */
function LlamaCppOptions({ options, set }: { options: Record<string, any>; set: (k: string, v: unknown) => void }) {
  const { overview } = useLive();
  const sharing = (overview?.links?.links ?? []).filter((l) => l.status === "up" && l.pool?.enabled);
  const pooledVram = sharing.flatMap((l) => l.pool?.gpus ?? []).reduce((sum, g) => sum + (g.vram_free ?? 0), 0);
  return (
    <>
      <div className="sm:col-span-2">
        <Toggle checked={options.pool === "all"} disabled={!sharing.length && options.pool !== "all"}
          onChange={(v) => set("pool", v ? "all" : undefined)}
          label={`Pool GPUs from linked sessions${sharing.length ? ` (${sharing.length} sharing, ${bytes(pooledVram)} free)` : ""}`}
          hint={sharing.length
            ? "Splits the model's layers across this GPU and every linked session that shares its GPU. Each helper downloads its own share of the weights. Slower per token than one GPU: use it when the model doesn't fit."
            : "No linked session is sharing its GPU yet. On the other session, open Linked sessions and turn on 'Share this GPU', then link it here."} />
      </div>
      <Field label="Parallel requests" hint="Slots served at once. Agents with sub-tasks benefit from 4 or more; each slot shares the context window.">
        <Select value={String(options.parallel ?? 4)} onChange={(e) => set("parallel", Number(e.target.value))}>
          {[1, 2, 4, 8].map((v) => <option key={v} value={v}>{v}</option>)}
        </Select>
      </Field>
      <Field label="MoE layers with experts in RAM" hint="For mixture-of-experts models that almost fit: keeps the experts of this many layers in system RAM. Slower; 0 keeps everything on GPU.">
        <Input type="number" min={0} max={200} value={options.cpu_moe ?? ""} placeholder="0" onChange={(e) => set("cpu_moe", e.target.value ? Number(e.target.value) : "")} />
      </Field>
    </>
  );
}

/** Engine options shown per modality. Values are merged into spec.options. */
function EngineOptions({
  item,
  options,
  onChange,
}: {
  item: Pick<CatalogItem, "modality" | "engine" | "source" | "custom_code" | "context_length" | "bf16_saved">;
  options: Record<string, any>;
  onChange: (o: Record<string, any>) => void;
}) {
  const set = (k: string, v: unknown) => onChange({ ...options, [k]: v === "" ? undefined : v });
  const m = item.modality;
  if (m === "text") {
    const fallback = Math.min(32768, item.context_length ?? 32768);
    const maxCtx = Math.max(item.context_length ?? CTX_UNKNOWN_MAX, CTX_MIN);
    const ctx = Number(options.ctx ?? fallback);
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        <Field
          label={`Context length · ${ctx.toLocaleString()} tokens`}
          hint={
            item.context_length
              ? `This model supports up to ${tokens(item.context_length)} tokens. Longer context uses more VRAM for the KV cache.`
              : "Longer context uses more VRAM for the KV cache."
          }
        >
          <ContextLength value={ctx} max={maxCtx} onChange={(v) => set("ctx", v)} />
        </Field>
        {item.engine === "vllm" && (
          <Field label="GPU memory use" hint="Fraction of VRAM vLLM reserves.">
            <Select value={String(options.gpu_memory_utilization ?? 0.9)} onChange={(e) => set("gpu_memory_utilization", Number(e.target.value))}>
              {[0.8, 0.85, 0.9, 0.95].map((v) => (
                <option key={v} value={v}>
                  {Math.round(v * 100)}%
                </option>
              ))}
            </Select>
          </Field>
        )}
        {item.engine === "llamacpp" && <LlamaCppOptions options={options} set={set} />}
        {item.engine === "sglang" && (
          <Field label="GPU memory use" hint="Fraction of VRAM SGLang reserves for weights and the KV cache.">
            <Select value={String(options.mem_fraction_static ?? 0.88)} onChange={(e) => set("mem_fraction_static", Number(e.target.value))}>
              {[0.7, 0.8, 0.85, 0.88, 0.9, 0.95].map((v) => (
                <option key={v} value={v}>
                  {Math.round(v * 100)}%
                </option>
              ))}
            </Select>
          </Field>
        )}
      </div>
    );
  }
  if (m === "image" || m === "video" || (m === "world" && item.engine === "diffusers_world")) {
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Memory mode" hint="Offload keeps idle parts in RAM: slower, fits bigger models.">
          <Select value={String(options.offload ?? (m === "video" ? "model" : "none"))} onChange={(e) => set("offload", e.target.value)}>
            <option value="none">All on GPU (fastest)</option>
            <option value="model">Model offload (balanced)</option>
            <option value="sequential">Sequential offload (smallest, slowest)</option>
          </Select>
        </Field>
        <Field label="Weights" hint="4-bit shrinks big models like FLUX and Wan 14B to fit a T4.">
          <Select value={String(options.quantize ?? "")} onChange={(e) => set("quantize", e.target.value)}>
            <option value="">16-bit (full quality)</option>
            <option value="nf4">4-bit nf4 (bitsandbytes)</option>
          </Select>
        </Field>
        {item.bf16_saved ? (
          <Field label="Download as" hint="The repo is published in fp32; the runner uses 16 bits either way.">
            <Select value={String(options.weights ?? "bf16")} onChange={(e) => set("weights", e.target.value === "bf16" ? "" : e.target.value)}>
              <option value="bf16">BF16 (saves {bytes(item.bf16_saved)})</option>
              <option value="original">Original fp32 files</option>
            </Select>
          </Field>
        ) : null}
        <Field label="Default steps">
          <Input type="number" min={1} max={100} value={options.steps ?? ""} placeholder="model default" onChange={(e) => set("steps", e.target.value ? Number(e.target.value) : "")} />
        </Field>
        {m === "image" ? (
          <Field label="Default guidance">
            <Input type="number" step={0.5} min={0} max={20} value={options.guidance ?? ""} placeholder="model default" onChange={(e) => set("guidance", e.target.value ? Number(e.target.value) : "")} />
          </Field>
        ) : (
          <Field label="Default frames">
            <Input type="number" min={5} max={241} value={options.frames ?? ""} placeholder="model default" onChange={(e) => set("frames", e.target.value ? Number(e.target.value) : "")} />
          </Field>
        )}
      </div>
    );
  }
  if (m === "world") {
    if (options.recipe === "hunyuan_world_1" || options.recipe === "hy_world_2") {
      return <p className="text-[13px] text-fg-2">Scene description, panorama input and reconstruction settings are available in Studio after launch.</p>;
    }
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Default frames">
          <Input type="number" min={4} max={600} value={options.frames ?? ""} placeholder="recipe default" onChange={(e) => set("frames", e.target.value ? Number(e.target.value) : "")} />
        </Field>
        <Field label="Default steps">
          <Input type="number" min={1} max={50} value={options.steps ?? ""} placeholder="recipe default" onChange={(e) => set("steps", e.target.value ? Number(e.target.value) : "")} />
        </Field>
      </div>
    );
  }
  return <p className="text-[13px] text-fg-3">No extra settings for this engine.</p>;
}

export function OptionsForm(props: Parameters<typeof EngineOptions>[0]) {
  const { item, options, onChange } = props;
  const customCode = ["vllm", "sglang", "transformers", "transformers_tts", "transformers_audio", "transformers_asr"].includes(item.engine ?? "");
  return <div className="space-y-4">
    <EngineOptions {...props} />
    {customCode && <div className="space-y-3 border-t border-line pt-4">
      <Toggle checked={options.trust_remote_code === true}
        onChange={v => onChange({ ...options, trust_remote_code: v })}
        label="Allow model custom code"
        hint={item.custom_code ? "This checkpoint declares custom code. Enable only if you trust its source; its Python code runs in your notebook." : "For checkpoints that need their own Python code. Enable only for a source you trust."} />
      {item.source === "hf" && <Field label="Model revision" hint="Optional branch, tag or commit SHA. Use a commit SHA to pin the model and its code.">
        <Input value={options.revision ?? ""} placeholder="main" onChange={e => onChange({ ...options, revision: e.target.value || undefined })} />
      </Field>}
    </div>}
  </div>;
}

function VariantRow({ v, active, onPick }: { v: Variant; active: boolean; onPick: () => void }) {
  return (
    <button
      type="button"
      onClick={onPick}
      className={cx(
        "flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors",
        active ? "border-signal bg-signal-soft" : "border-line hover:border-line-2",
      )}
    >
      <span className={cx("size-3 shrink-0 rounded-full border-2", active ? "border-signal bg-signal" : "border-line-2")} />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-mono text-[12.5px]">{v.label}</span>
        {v.fit?.reasons?.[0] && <span className="block truncate text-[11.5px] text-fg-3">{v.fit.reasons[0]}</span>}
      </span>
      <span className="font-mono text-[11.5px] text-fg-2">{bytes(v.size)}</span>
      {v.fit && <VerdictTag verdict={v.fit.verdict} size="sm" />}
    </button>
  );
}

export function ModelDetail({
  item: base,
  actionLabel,
  onAction,
  busy,
  secondary,
  onImport,
}: {
  item: CatalogItem;
  actionLabel: string;
  onAction: (choice: ModelChoice) => void;
  busy?: boolean;
  secondary?: (choice: ModelChoice) => React.ReactNode;
  onImport?: (url: string) => void;
}) {
  const { overview } = useLive();
  const nav = useNavigate();
  const pool = overview ? memoryPool(overview.system) : null;
  const devices = pool?.devices ?? [];
  const canDetail = base.source === "hf" || base.source === "ollama";
  const detail = useQuery({
    queryKey: ["detail", base.key],
    queryFn: () => api.detail(base.key),
    enabled: canDetail,
    staleTime: 10 * 60_000,
  });
  const loaded = detail.data;
  // "Best" is relative to the list it came from; keep it when the detail view agrees the model fits well.
  const item =
    loaded && base.fit.verdict === "best" && loaded.fit.verdict === "good" ? { ...loaded, fit: { ...loaded.fit, verdict: "best" as const } } : (loaded ?? base);
  const [variant, setVariant] = useState<string | null>(base.variant);
  const [engine, setEngine] = useState<string | null>(base.engine);
  const [options, setOptions] = useState<Record<string, any>>({ ...(base.spec?.options ?? {}) });
  const [name, setName] = useState("");

  useEffect(() => {
    if (detail.data) {
      setVariant((v) => v ?? detail.data.variant);
      setOptions((o) => ({ ...(detail.data.spec?.options ?? {}), ...o }));
    }
  }, [detail.data]);

  const engines = useMemo(() => {
    if (item.modality === "text" && item.source === "hf" && item.engine !== "ollama")
      return pool?.tpu ? ["vllm", "transformers"] : ["vllm", "sglang", "transformers"];
    // GGUF repos: Ollama, or llama.cpp directly (needed to pool GPUs across sessions).
    if (item.modality === "text" && (item.source === "hf" || item.source === "attached") && item.engine === "ollama" && !pool?.tpu)
      return ["ollama", "llamacpp"];
    return item.engine ? [item.engine] : [];
  }, [item, pool?.tpu]);

  const picked = item.variants.find((v) => v.id === variant);
  // Re-check size and fit whenever a setting that changes them does (variant, engine, context, memory mode).
  const current = JSON.stringify({
    variant: variant ?? item.variant,
    engine: engine ?? item.engine,
    options: Object.fromEntries(FIT_OPTIONS.map((k) => [k, options[k]]).filter(([, v]) => v !== undefined && v !== "")),
  });
  const request = useDebounced(current, 200);
  const assessed = useQuery({
    queryKey: ["assess", base.key, request],
    queryFn: () => api.assess({ key: base.key, ...JSON.parse(request) }),
    enabled: canDetail && !!item.engine && detail.isSuccess,
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
    retry: false,
  });
  const assessedFit = assessed.data?.fit;
  const fit = assessedFit ? { ...assessedFit, verdict: assessedFit.verdict === "good" && item.fit.verdict === "best" ? ("best" as const) : assessedFit.verdict } : item.fit;
  const size = assessed.data?.size ?? picked?.size ?? item.size;
  const updating = assessed.isFetching || request !== current;
  const choice: ModelChoice | null = item.spec
    ? {
        item: { ...item, size, fit, variant: variant ?? item.variant },
        spec: {
          ...item.spec,
          model: { ...item.spec.model, variant: variant ?? item.spec.model.variant, engine: engine ?? item.spec.model.engine },
          name: name.trim() || null,
          options: Object.fromEntries(
            Object.entries<unknown>({
              ...options,
              // Placement hint for the GPU scheduler; a different variant than the default changes it.
              vram_estimate: fit.vram_needed ?? undefined,
            }).filter(([, v]) => v !== undefined && v !== ""),
          ),
        },
      }
    : null;

  return (
    <div className="flex min-h-full flex-col">
      <div className="space-y-6 p-5 sm:p-6">
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-2">
            {item.engine ? <VerdictTag verdict={fit.verdict} /> : <span className="text-[12px] text-fg-2">Runner unavailable</span>}
            {item.experimental && <span className="rounded-[5px] bg-tight-soft px-1.5 py-px text-[11px] font-medium text-tight">Experimental</span>}
            {detail.isFetching && <Loader className="size-3.5 animate-spin text-fg-3" />}
          </div>
          <div className="flex items-start gap-4"><ModelLogo id={item.id} name={item.name} source={item.source} className="size-12" /><h2 className="display min-w-0 break-words text-[20px] leading-tight">{item.name}</h2></div>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11.5px] text-fg-2">
            <span>{item.id}</span>
            {item.brand && <span>{item.brand.label}</span>}
            {item.params ? <span>{params(item.params)} params</span> : null}
            {item.context_length ? <span>{tokens(item.context_length)} context</span> : null}
            {item.engine_label && <span>{item.engine_label}</span>}
            {item.url && (
              <a href={item.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-fg-2 hover:text-signal">
                page <ExternalLink className="size-3" />
              </a>
            )}
          </div>
          {item.description && <p className="mt-3 text-[14px] text-fg-2">{item.description}</p>}
          {item.capabilities.length > 0 && <div className="mt-4"><CapabilityTags capabilities={item.capabilities} /></div>}
          {item.reasoning?.supported && <p className="mt-2 text-[12px] text-fg-3">Reasoning: {item.reasoning.efforts.length ? ["Auto", ...item.reasoning.efforts.map(effortLabel)].join(" · ") : "Model default"}. {item.reasoning.source === "family" ? "Estimated from the model family; verified by the runner when loaded." : "Detected from model metadata or its chat template."}</p>}
          {!!item.runtime_repos?.length && <div className="mt-3 text-[12px] text-fg-3"><p>The download estimate includes these base checkpoints:</p><div className="mt-1 flex flex-wrap gap-x-3 gap-y-1">{item.runtime_repos.map(repo => <a key={repo} href={`https://huggingface.co/${repo}`} target="_blank" rel="noreferrer" className="text-fg-2 hover:text-signal hover:underline">{repo}</a>)}</div></div>}
        </div>

        {!item.engine ? (
          <div className="rounded-xl border border-line bg-bg-2 p-4">
            <div className="mb-2 text-[14px] font-medium">Runner not available yet</div>
            <p className="text-[13px] text-fg-2">This model is published on Hugging Face. Kindle doesn't have a runtime for it yet, so it can't be deployed from this panel.</p>
            {item.url && <a href={item.url} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1.5 text-[13px] text-signal hover:underline">Open model documentation <ExternalLink className="size-3.5" /></a>}
            {item.url && onImport && <div className="mt-3"><Button onClick={() => onImport(item.url!)}>Download model files</Button></div>}
          </div>
        ) : <div className="rounded-xl border border-line bg-bg-2 p-4">
          <div className="mb-3 flex items-baseline justify-between gap-3">
            <span className="readout">Fit with these settings</span>
            <span className={cx("font-mono text-[11.5px] text-fg-2 transition-opacity", updating && "opacity-50")}>
              {size ? `${bytes(size)} download` : "size unknown"}
              {updating && <Loader className="ml-1.5 inline size-3 animate-spin align-[-2px]" />}
            </span>
          </div>
          <div className={cx("transition-opacity", updating && "opacity-60")}>
          <FitGauge fit={fit} devices={devices} />
          <ul className="mt-3 space-y-1 text-[13px]">
            {fit.reasons.map((r) => (
              <li key={r} className="text-fg-2">
                {r}
              </li>
            ))}
            {fit.warnings.map((w) => (
              <li key={w} className="flex items-start gap-1.5 text-tight">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                <span>
                  {w}
                  {w.startsWith("Gated") && (
                    <Link to="/access#secrets" className="ml-2 whitespace-nowrap font-medium underline underline-offset-4">
                      Add token →
                    </Link>
                  )}
                </span>
              </li>
            ))}
          </ul>
          </div>
          {assessed.error && <p className="mt-2 text-[12px] text-tight">Couldn't re-check this configuration: {(assessed.error as Error).message}</p>}
          {item.bf16_saved && options.weights !== "original" ? (
            <p className="mt-3 text-[12.5px] text-fg-2">
              Published in fp32. Kindle stores the weights as BF16 while downloading, which saves {bytes(item.bf16_saved)} of disk. Generation quality is unchanged: the runner loads 16-bit weights either way.
            </p>
          ) : null}
          {Object.keys(fit.suggest ?? {}).length > 0 && (
            <div className="mt-3 font-mono text-[11px] text-fg-3">auto-applied: {Object.entries(fit.suggest).map(([k, v]) => `${k}=${v}`).join(", ")}</div>
          )}
        </div>}

        {item.variants.length > 1 && (
          <div>
            <div className="readout mb-2">{item.source === "ollama" ? "Size / tag" : "Quantization"}</div>
            <div className="max-h-[300px] space-y-1.5 overflow-y-auto pr-1 scroll-thin">
              {item.variants.map((v) => (
                <VariantRow key={v.id} v={v} active={v.id === (variant ?? item.variant)} onPick={() => setVariant(v.id)} />
              ))}
            </div>
            {picked?.context && <div className="mt-2 font-mono text-[11px] text-fg-3">context window {picked.context}</div>}
          </div>
        )}

        {engines.length > 1 && (
          <Field label="Engine" hint="vLLM and SGLang are the fast runners (experimental on T4s); Transformers is the safe fallback.">
            <Select value={engine ?? ""} onChange={(e) => setEngine(e.target.value)}>
              {engines.map((e) => (
                <option key={e} value={e}>
                  {ENGINE_LABEL[e] ?? e}
                </option>
              ))}
            </Select>
          </Field>
        )}

        {item.engine && (
          <div>
            <div className="readout mb-2">Settings</div>
            <OptionsForm
              item={{ ...item, engine: engine ?? item.engine, context_length: picked?.context_length ?? item.context_length }}
              options={options}
              onChange={setOptions}
            />
          </div>
        )}

        {item.engine && (
          <Field label="API model name (optional)" hint="What clients send as 'model'. Defaults to a short name.">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder={item.source === "ollama" ? `${item.id}:${variant ?? "latest"}` : item.id.split("/").pop()?.toLowerCase()} />
          </Field>
        )}
      </div>

      <div className="sticky bottom-0 mt-auto flex flex-wrap items-center justify-end gap-2 border-t border-line bg-panel/95 px-5 py-4 backdrop-blur sm:px-6">
        {!item.engine ? (
          <span className="mr-auto text-[13px] text-fg-3">This architecture needs a model-specific runner.</span>
        ) : (
          <>
            {item.modality === "text" && (item.source === "hf" || item.source === "ollama") && (
              <button
                type="button"
                className="mr-auto text-[13px] text-fg-2 underline-offset-4 hover:text-fg hover:underline"
                onClick={async () => {
                  // Ollama/GGUF can't be edited: start from the original full-precision repo.
                  let repo = item.abliterate_source ?? "";
                  let guessed = Boolean(item.abliterate_guessed);
                  if (item.source === "ollama") {
                    repo = (await api.original(item.id, variant ?? item.variant ?? "").catch(() => ({ repo: null }))).repo ?? "";
                    guessed = true;
                  }
                  nav("/imports", { state: { abliterate: repo, abliterateFrom: guessed || repo !== item.id ? item.name : undefined } });
                }}
              >
                Abliterate…
              </button>
            )}
            {choice && secondary?.(choice)}
            <button
              disabled={!choice || busy}
              onClick={() => choice && onAction(choice)}
              className="inline-flex h-11 items-center gap-2 rounded-lg bg-signal px-5 text-[15px] font-semibold text-signal-ink transition hover:brightness-110 disabled:opacity-50"
            >
              {busy && <Loader className="size-4 animate-spin" />}
              {actionLabel}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
