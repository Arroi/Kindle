import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Download, HardDrive, Heart, LayoutGrid, List, Search, Star, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "../lib/api";
import { bytes, compact, cx, MODALITY_LABEL, params } from "../lib/format";
import { useLive } from "../lib/live";
import type { CatalogItem, Gpu, Modality } from "../lib/types";
import { FitGauge } from "./meters";
import { Button, Chip, Empty, Input, Segmented, Select, Toggle, VerdictTag } from "./ui";

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
  const label = item.source === "hf" ? "HF" : item.source === "ollama" ? "Ollama" : item.capabilities.includes("imported") ? "Imported" : "Attached";
  return (
    <span className="flex items-center gap-1.5">
      <span
        className={cx(
          "rounded px-1.5 py-px text-[11px] font-medium",
          item.source === "attached" ? "bg-info/15 text-info" : "bg-panel-3 text-fg-2",
        )}
      >
        {label}
      </span>
      {item.curated && (
        <span className="flex items-center gap-1 text-[11px] text-fg-2" title="Curated model">
          <Star className="size-3 fill-current" />
          Pick
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
        "panel group flex h-full w-full min-w-0 flex-col p-5 text-left transition-[border-color,background] duration-150 hover:bg-panel-2",
        selected ? "border-signal ring-1 ring-signal" : "hover:border-line-2",
      )}
      aria-pressed={!!selected}
    >
      <div className="mb-2.5 flex flex-wrap items-start justify-between gap-2">
        <SourceBadge item={item} />
        {unsupported ? <Chip>Runner unavailable</Chip> : <VerdictTag verdict={f.verdict} size="sm" />}
      </div>
      <div className="mb-1 flex items-baseline gap-2">
        <h3 className="line-clamp-2 min-w-0 break-words text-[18px] font-semibold leading-snug [overflow-wrap:anywhere]">{item.name}</h3>
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
        {unsupported ? <p className="text-[12px] text-fg-3">View the published model and its requirements.</p> : <FitGauge fit={f} gpus={gpus} />}
        <div className="mt-2 flex items-start justify-between gap-2">
          <p className="line-clamp-2 text-[12px] text-fg-2">{unsupported ? "This architecture needs a model-specific runner." : f.reasons[0]}</p>
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

function ModelRow({ item, selected, onOpen }: { item: CatalogItem; selected: boolean; onOpen: () => void }) {
  return (
    <button type="button" className="model-row model-columns" onClick={onOpen} aria-pressed={selected}>
      <div className="min-w-0">
        <div className="mb-1.5 flex flex-wrap items-center gap-x-3 gap-y-1">
          <h3 className="min-w-0 break-words font-semibold text-[15px] [overflow-wrap:anywhere]">{item.name}</h3>
          <SourceBadge item={item} />
        </div>
        <p className="line-clamp-1 text-[13px] text-fg-2">{item.description || item.id}</p>
        <div className="mt-1.5 flex flex-wrap items-center gap-x-2 text-[11px] text-fg-3">
          {item.variant && <span>{item.variant}</span>}
          {item.capabilities.slice(0, 2).map((cap) => <span key={cap}>{cap}</span>)}
          {item.experimental && <span className="text-tight">Experimental</span>}
        </div>
      </div>
      <div className="model-stats">
        <div>
          <span className="block text-[13px] font-medium tabular-nums">{params(item.params) || "Unknown"}</span>
          <span className="text-[11px] text-fg-3 @3xl:hidden">Parameters</span>
        </div>
        <div>
          <span className="block text-[13px] tabular-nums">{bytes(item.size)}</span>
          <span className="text-[11px] text-fg-3 @3xl:hidden">Download</span>
        </div>
        <div className="model-fit">
          {item.engine ? <VerdictTag verdict={item.fit.verdict} size="sm" /> : <Chip>Runner unavailable</Chip>}
          <span className="mt-1 block text-[11px] text-fg-3">{!item.engine ? "View model details" : item.fit.vram_needed ? `≈${bytes(item.fit.vram_needed)} VRAM` : "Estimate unavailable"}</span>
        </div>
      </div>
    </button>
  );
}

type FitFilter = "all" | "fits";
const FIT_KEY = "kaggleui-fit-filter";
const PAGE = 40;

function storedFit(): FitFilter {
  try {
    return localStorage.getItem(FIT_KEY) === "fits" ? "fits" : "all";
  } catch {
    return "all";
  }
}

const SORTS = [
  { value: "recommended", label: "Best fit" },
  { value: "trending", label: "Trending" },
  { value: "downloads", label: "Downloads" },
  { value: "newest", label: "Newest" },
  { value: "params_asc", label: "Lowest parameters" },
  { value: "params_desc", label: "Highest parameters" },
];

function storedSort(): string | null {
  try {
    const value = localStorage.getItem("kaggleui-catalog-sort");
    return SORTS.some((option) => option.value === value) ? value : null;
  } catch { return null; }
}

function storedView(): "list" | "grid" {
  try { return localStorage.getItem("kaggleui-catalog-view") === "grid" ? "grid" : "list"; }
  catch { return "list"; }
}

export function ModelBrowser({
  modality,
  onModality,
  onOpen,
  selectedKey,
  lockModality,
  initialSource = "all",
}: {
  modality: Modality;
  onModality?: (m: Modality) => void;
  onOpen: (item: CatalogItem) => void;
  selectedKey?: string | null;
  lockModality?: boolean;
  initialSource?: string;
}) {
  const { overview } = useLive();
  const gpus = overview?.system.gpus ?? [];
  const [q, setQ] = useState("");
  const [source, setSource] = useState(initialSource);
  const [generalFit, setGeneralFit] = useState<FitFilter>(() => initialSource === "attached" ? "all" : storedFit());
  const [worldFit, setWorldFit] = useState<FitFilter>("all");
  const fitFilter = modality === "world" ? worldFit : generalFit;
  // "All models" reads best by popularity (big models mixed in); "Fits" by how well they fit.
  const [sortChoice, setSortChoice] = useState<string | null>(storedSort);
  const [view, setView] = useState(storedView);
  const sort = sortChoice ?? (modality === "world" || fitFilter === "fits" ? "recommended" : "trending");
  const [unsupported, setUnsupported] = useState(false);
  const [launchableOnly, setLaunchableOnly] = useState(false);
  const showUnsupported = modality === "world" ? !launchableOnly : unsupported;
  const [limit, setLimit] = useState(PAGE);
  const dq = useDebounced(q, 350);

  useEffect(() => setLimit(PAGE), [modality, dq, source, sort, showUnsupported, fitFilter]);

  function changeFit(v: FitFilter) {
    if (modality === "world") {
      setWorldFit(v);
      return;
    }
    setGeneralFit(v);
    try {
      localStorage.setItem(FIT_KEY, v);
    } catch {
      /* storage unavailable */
    }
  }

  function changeSort(value: string | null) {
    setSortChoice(value);
    try {
      if (value) localStorage.setItem("kaggleui-catalog-sort", value);
      else localStorage.removeItem("kaggleui-catalog-sort");
    } catch { /* storage unavailable */ }
  }

  function changeView(value: "list" | "grid") {
    setView(value);
    try { localStorage.setItem("kaggleui-catalog-view", value); }
    catch { /* storage unavailable */ }
  }

  function resetFilters() {
    setQ("");
    setSource("all");
    setUnsupported(false);
    setLaunchableOnly(false);
    changeFit("all");
    changeSort(null);
  }

  const query = useQuery({
    queryKey: ["search", modality, dq, source, sort, showUnsupported, fitFilter, limit],
    queryFn: () => api.search({ modality, q: dq, source, sort, show_unsupported: showUnsupported, fit: fitFilter, limit }),
    staleTime: 5 * 60_000,
    placeholderData: (prev) => (prev?.modality === modality ? prev : undefined),
  });
  const loadingMore = query.isFetching && query.isPlaceholderData && (query.data?.items.length ?? 0) < limit;

  const sources = useMemo(() => {
    const base = [
      { value: "all", label: "All" },
      { value: "curated", label: "Picks" },
      { value: "hf", label: "Hugging Face" },
    ];
    if (modality === "text") base.push({ value: "ollama", label: "Ollama" });
    base.push({ value: "attached", label: "Local" });
    return base;
  }, [modality]);

  const errors = Object.entries(query.data?.sources ?? {}).filter(([, s]) => s.startsWith("error") || s.startsWith("partial"));

  return (
    <div className="@container">
      {!lockModality && onModality && (
        <nav className="catalog-tabs scroll-thin" aria-label="Model types">
          {MODALITY_ORDER.map((m) => (
            <button key={m} type="button" aria-pressed={modality === m} onClick={() => {
              onModality(m);
              if (m === "world") setSource("all");
              if (m !== "text" && source === "ollama") setSource("all");
            }}>{MODALITY_LABEL[m]}</button>
          ))}
        </nav>
      )}
      {modality === "world" && (
        <p className="mb-5 max-w-[80ch] text-[13px] text-fg-2">
          Browse published world models from Hugging Face. Models marked “Runner unavailable” can be downloaded; launch requires a runner for their architecture. GPU fit is checked separately.
        </p>
      )}
      <div className="mb-4 flex flex-col gap-3 @4xl:flex-row @4xl:items-center">
        <div className="relative min-w-0 flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-fg-3" />
          <Input
            aria-label="Search models"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={modality === "text" ? "Search models (qwen, gemma, coder…)" : `Search ${MODALITY_LABEL[modality].toLowerCase()} models`}
            className="pl-9"
          />
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Select aria-label="Sort models" value={sort} onChange={(e) => changeSort(e.target.value)} className="catalog-sort">
            {SORTS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </Select>
          <div className="flex h-10 items-center rounded-lg border border-line p-0.5" role="group" aria-label="Model layout">
            {(["list", "grid"] as const).map((value) => {
              const Icon = value === "list" ? List : LayoutGrid;
              return <button key={value} type="button" onClick={() => changeView(value)} aria-label={`${value === "list" ? "List" : "Grid"} view`} aria-pressed={view === value} className={cx("rounded-md p-2 transition-colors", view === value ? "bg-panel-3 text-fg" : "text-fg-3 hover:text-fg")}><Icon className="size-4" /></button>;
            })}
          </div>
        </div>
      </div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <Segmented size="sm" value={source} onChange={setSource} options={sources} />
        <div className="flex flex-wrap items-center gap-4">
          <Segmented<FitFilter>
            size="sm"
            value={fitFilter}
            onChange={changeFit}
            options={[
              { value: "all", label: "All models" },
              { value: "fits", label: "Fits this machine" },
            ]}
          />
          <Toggle checked={modality === "world" ? launchableOnly : unsupported} onChange={modality === "world" ? setLaunchableOnly : setUnsupported} label={<span className="text-[13px] font-normal text-fg-2">{modality === "world" ? "Launchable only" : "Show unsupported"}</span>} />
        </div>
      </div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-[12px] text-fg-3" role="status" aria-live="polite">
        <span>
          {query.isFetching && !loadingMore ? "Checking catalogs…" : query.data?.counts ? `${query.data.items.length} of ${query.data.counts.total}${query.data.has_more ? "+" : ""} models${modality === "world" ? ` · ${query.data.counts.launchable ?? 0} with launch support` : ""} · ${query.data.counts.fits} fit this machine` : query.data ? `${query.data.items.length} models` : ""}
        </span>
        {(q || source !== "all" || (modality === "world" ? launchableOnly : unsupported) || fitFilter !== "all" || sortChoice) && (
          <button type="button" onClick={resetFilters} className="flex items-center gap-1 text-fg-2 hover:text-fg"><X className="size-3" /> Reset filters</button>
        )}
      </div>
      {(sort === "params_asc" || sort === "params_desc") && (
        <p className="mb-4 text-[12px] text-fg-3">Sorted by the shown variant's parameter count. Unknown counts appear last. Load more to include more catalog results.</p>
      )}
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
        <div className={view === "grid" ? "grid gap-3 @xl:grid-cols-2 @5xl:grid-cols-3" : "panel overflow-hidden"} aria-label="Loading models" aria-busy="true">
          {Array.from({ length: 6 }, (_, i) => (
            <div key={i} className={cx("animate-pulse bg-panel-2", view === "grid" ? "panel h-[248px]" : "m-4 h-16 rounded-md")} />
          ))}
        </div>
      ) : query.error ? (
        <Empty title="Couldn't load models">{String((query.error as Error).message)}</Empty>
      ) : query.data && query.data.items.length === 0 ? (
        <Empty title="Nothing found">Try another search, or switch the source filter.</Empty>
      ) : (
        <div className={cx(view === "grid" ? "grid gap-3 @xl:grid-cols-2 @5xl:grid-cols-3" : "panel overflow-hidden", query.isPlaceholderData && "opacity-60")} aria-busy={query.isFetching}>
          {view === "list" && (
            <div className="model-header model-columns border-b border-line bg-panel-2 px-5 py-2.5 text-[12px] text-fg-3" aria-hidden="true">
              <span>Model</span><span>Parameters</span><span>Download</span><span className="text-right">{modality === "world" ? "Support / GPU fit" : "GPU fit"}</span>
            </div>
          )}
          {query.data?.items.map((it) => (
            view === "grid"
              ? <ModelCard key={it.key} item={it} gpus={gpus} selected={selectedKey === it.key} onOpen={() => onOpen(it)} />
              : <ModelRow key={it.key} item={it} selected={selectedKey === it.key} onOpen={() => onOpen(it)} />
          ))}
        </div>
      )}
      {query.data?.has_more && (query.data.items.length >= limit || modality === "world") && (
        <div className="mt-5 flex justify-center">
          <Button busy={loadingMore} onClick={() => setLimit((l) => Math.min(l + PAGE, 300))} disabled={limit >= 300}>
            {limit >= 300 ? "Search to narrow it down" : "Load more"}
          </Button>
        </div>
      )}
    </div>
  );
}
