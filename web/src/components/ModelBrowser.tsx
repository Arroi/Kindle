import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Download, HardDrive, Heart, Search, Star } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "../lib/api";
import { bytes, compact, cx, MODALITY_LABEL, params } from "../lib/format";
import { useLive } from "../lib/live";
import type { CatalogItem, Gpu, Modality } from "../lib/types";
import { FitGauge } from "./meters";
import { Chip, Empty, Input, Segmented, Toggle, VerdictTag } from "./ui";

export const MODALITY_ORDER: Modality[] = ["text", "image", "video", "tts", "stt", "world"];

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(t);
  }, [value, ms]);
  return v;
}

function SourceBadge({ item }: { item: CatalogItem }) {
  const label = item.source === "hf" ? "HF" : item.source === "ollama" ? "Ollama" : "Attached";
  return (
    <span className="flex items-center gap-1.5">
      <span
        className={cx(
          "rounded px-1.5 py-px font-mono text-[9.5px] uppercase tracking-wider",
          item.source === "attached" ? "bg-info/15 text-info" : "bg-panel-3 text-fg-2",
        )}
      >
        {label}
      </span>
      {item.curated && (
        <span className="flex items-center gap-1 font-mono text-[9.5px] uppercase tracking-wider text-signal" title="Kaggle-tested pick">
          <Star className="size-3 fill-current" />
          pick
        </span>
      )}
    </span>
  );
}

export function ModelCard({
  item,
  gpus,
  selected,
  onOpen,
}: {
  item: CatalogItem;
  gpus: Gpu[];
  selected?: boolean;
  onOpen: () => void;
}) {
  const f = item.fit;
  const unsupported = !item.engine;
  return (
    <button
      type="button"
      onClick={onOpen}
      className={cx(
        "panel group flex h-full w-full min-w-0 flex-col p-4 text-left transition-[border-color,transform] duration-150 hover:-translate-y-0.5",
        selected ? "border-signal ring-1 ring-signal" : "hover:border-line-2",
        unsupported && "opacity-60",
      )}
    >
      <div className="mb-2.5 flex flex-wrap items-start justify-between gap-2">
        <SourceBadge item={item} />
        <VerdictTag verdict={f.verdict} size="sm" />
      </div>
      <div className="mb-1 flex items-baseline gap-2">
        <h3 className="display line-clamp-2 min-w-0 break-all text-[22px] leading-[1.02] group-hover:text-signal">{item.name}</h3>
      </div>
      {item.source === "hf" && item.name !== item.id && <div className="mb-1 truncate font-mono text-[10.5px] text-fg-3">{item.id}</div>}
      <p className="mb-3 line-clamp-2 min-h-[2.6em] text-[13px] leading-snug text-fg-2">{item.description || " "}</p>
      <div className="mb-3 flex flex-wrap gap-1.5">
        {item.variant && <Chip>{item.variant}</Chip>}
        {item.params && params(item.params).toLowerCase() !== (item.variant ?? "").toLowerCase() ? <Chip>{params(item.params)}</Chip> : null}
        {item.size ? (
          <Chip>
            <HardDrive className="size-3" />
            {bytes(item.size)}
          </Chip>
        ) : null}
        {item.capabilities.slice(0, 3).map((c) => (
          <Chip key={c} tone="info">
            {c}
          </Chip>
        ))}
        {item.experimental && <Chip tone="warn">experimental</Chip>}
      </div>
      <div className="mt-auto">
        <FitGauge fit={f} gpus={gpus} />
        <div className="mt-2 flex items-start justify-between gap-2">
          <p className="line-clamp-2 text-[12px] text-fg-2">{f.reasons[0]}</p>
          {f.warnings.length > 0 && (
            <span title={f.warnings.join("\n")} className="mt-0.5 shrink-0 text-tight">
              <AlertTriangle className="size-3.5" />
            </span>
          )}
        </div>
        {(item.downloads > 0 || item.likes > 0) && (
          <div className="mt-2 flex gap-3 font-mono text-[10.5px] text-fg-3">
            {item.downloads > 0 && (
              <span className="flex items-center gap-1">
                <Download className="size-3" />
                {compact(item.downloads)}
              </span>
            )}
            {item.likes > 0 && (
              <span className="flex items-center gap-1">
                <Heart className="size-3" />
                {compact(item.likes)}
              </span>
            )}
          </div>
        )}
      </div>
    </button>
  );
}

const SORTS = [
  { value: "recommended", label: "Best fit" },
  { value: "trending", label: "Trending" },
  { value: "downloads", label: "Downloads" },
  { value: "newest", label: "Newest" },
];

export function ModelBrowser({
  modality,
  onModality,
  onOpen,
  selectedKey,
  lockModality,
}: {
  modality: Modality;
  onModality?: (m: Modality) => void;
  onOpen: (item: CatalogItem) => void;
  selectedKey?: string | null;
  lockModality?: boolean;
}) {
  const { overview } = useLive();
  const gpus = overview?.system.gpus ?? [];
  const [q, setQ] = useState("");
  const [source, setSource] = useState("all");
  const [sort, setSort] = useState("recommended");
  const [unsupported, setUnsupported] = useState(false);
  const dq = useDebounced(q, 350);

  const query = useQuery({
    queryKey: ["search", modality, dq, source, sort, unsupported],
    queryFn: () => api.search({ modality, q: dq, source, sort, show_unsupported: unsupported }),
    staleTime: 5 * 60_000,
    placeholderData: (prev) => (prev?.modality === modality ? prev : undefined),
  });

  const sources = useMemo(() => {
    const base = [
      { value: "all", label: "All" },
      { value: "curated", label: "Picks" },
      { value: "hf", label: "Hugging Face" },
    ];
    if (modality === "text") base.push({ value: "ollama", label: "Ollama" });
    base.push({ value: "attached", label: "Attached" });
    return base;
  }, [modality]);

  const errors = Object.entries(query.data?.sources ?? {}).filter(([, s]) => s.startsWith("error"));

  return (
    <div className="@container">
      {!lockModality && onModality && (
        <div className="mb-4">
          <Segmented
            value={modality}
            onChange={(m) => {
              onModality(m);
              if (m !== "text" && source === "ollama") setSource("all");
            }}
            options={MODALITY_ORDER.map((m) => ({ value: m, label: MODALITY_LABEL[m] }))}
          />
        </div>
      )}
      <div className="mb-4 flex flex-col gap-3 @4xl:flex-row @4xl:items-center">
        <div className="relative min-w-0 flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-fg-3" />
          <Input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={modality === "text" ? "Search models (qwen, gemma, coder…)" : `Search ${MODALITY_LABEL[modality].toLowerCase()} models`}
            className="pl-9"
          />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Segmented size="sm" value={source} onChange={setSource} options={sources} />
          <Segmented size="sm" value={sort} onChange={setSort} options={SORTS} />
        </div>
      </div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="text-[13px] text-fg-3">
          {query.isFetching ? "Scanning catalogs and sizing against this machine…" : query.data ? `${query.data.items.length} models · tagged for ${query.data.hardware.accelerator}` : ""}
        </div>
        <Toggle checked={unsupported} onChange={setUnsupported} label={<span className="text-[13px] font-normal text-fg-2">Show unsupported</span>} />
      </div>
      {errors.length > 0 && (
        <div className="mb-4 rounded-lg border border-tight/40 bg-tight-soft px-3 py-2 text-[13px] text-tight">
          {errors.map(([k, v]) => (
            <div key={k}>
              {k}: {v}
            </div>
          ))}
        </div>
      )}
      {query.isLoading ? (
        <div className="grid gap-3 @xl:grid-cols-2 @5xl:grid-cols-3">
          {Array.from({ length: 6 }, (_, i) => (
            <div key={i} className="panel h-[248px] animate-pulse opacity-60" />
          ))}
        </div>
      ) : query.error ? (
        <Empty title="Couldn't load models">{String((query.error as Error).message)}</Empty>
      ) : query.data && query.data.items.length === 0 ? (
        <Empty title="Nothing found">Try another search, or switch the source filter.</Empty>
      ) : (
        <div className={cx("grid gap-3 @xl:grid-cols-2 @5xl:grid-cols-3", !query.isPlaceholderData && "stagger")}>
          {query.data?.items.map((it) => (
            <ModelCard key={it.key} item={it} gpus={gpus} selected={selectedKey === it.key} onOpen={() => onOpen(it)} />
          ))}
        </div>
      )}
    </div>
  );
}
