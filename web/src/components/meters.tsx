import { bytes, cx } from "../lib/format";
import type { Fit, Gpu, Verdict } from "../lib/types";

const VERDICT_COLOR: Record<Verdict, string> = {
  best: "var(--signal)",
  good: "var(--good)",
  tight: "var(--tight)",
  worst: "var(--worst)",
};

/** Segmented level meter. Ink until it gets full, then amber, then red. */
export function LedMeter({ value, segments = 20, className }: { value: number; segments?: number; className?: string }) {
  const lit = Math.round(Math.max(0, Math.min(1, value)) * segments);
  return (
    <div className={cx("flex gap-[2px]", className)} aria-hidden>
      {Array.from({ length: segments }, (_, i) => {
        const frac = (i + 1) / segments;
        const on = i < lit;
        const color = frac > 0.9 ? "var(--worst)" : frac > 0.75 ? "var(--tight)" : "var(--fg)";
        return (
          <span
            key={i}
            className="h-full flex-1 rounded-[1px] transition-[background,opacity] duration-300"
            style={{ background: on ? color : "var(--panel-3)", opacity: on ? (frac > 0.75 ? 1 : 0.85) : 1 }}
          />
        );
      })}
    </div>
  );
}

export function GpuMeter({ gpu, compact }: { gpu: Gpu; compact?: boolean }) {
  const tpu = gpu.kind === "tpu";
  const frac = gpu.vram_total ? gpu.vram_used / gpu.vram_total : 0;
  const name = tpu ? gpu.name.replace("TPU ", "").replace(" chip", "") : gpu.name.replace("Tesla ", "").replace("NVIDIA ", "").split("-")[0];
  return (
    <div className={cx("min-w-0", compact ? "w-[132px]" : "w-full")}>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <span className="truncate text-[12.5px] text-fg-2">
          <span className="font-mono text-fg-3">{tpu ? "TPU" : "GPU"}{gpu.index}</span> {name}
        </span>
        <span className="font-mono text-[12px] text-fg">{Math.round(frac * 100)}%</span>
      </div>
      <LedMeter value={frac} segments={compact ? 16 : 32} className="h-2" />
      {!compact && (
        <div className="mt-1.5 flex justify-between font-mono text-[11.5px] text-fg-3">
          <span>
            {bytes(gpu.vram_used)} / {bytes(gpu.vram_total)}
          </span>
          {tpu ? (
            <span>HBM</span>
          ) : (
            <span>
              {gpu.util ?? 0}% util{gpu.temp != null ? ` · ${gpu.temp}°C` : ""}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * The model's estimated VRAM/HBM drawn against this machine's memory devices.
 * Each device is its own segment (so "split across 2 GPUs" is visible), overflow is hatched.
 */
export function FitGauge({ fit, devices, className }: { fit: Fit; devices: Gpu[]; className?: string }) {
  const tpu = devices.some((d) => d.kind === "tpu");
  const unit = tpu ? "HBM" : "VRAM";
  const need = fit.vram_needed ?? 0;
  const color = VERDICT_COLOR[fit.verdict];
  if (!devices.length) {
    return (
      <div className={cx("font-mono text-[11px] text-fg-3", className)}>
        {need ? `≈${bytes(need)} needed · no ${tpu ? "TPU" : "GPU"}` : `no ${tpu ? "TPU" : "GPU"}`}
      </div>
    );
  }
  const usable = fit.placement === "split" || fit.placement === "none" ? devices : devices.slice(0, 1);
  const total = devices.reduce((s, g) => s + g.vram_total, 0);
  const overflow = Math.max(0, need - total);
  const scale = total + overflow;
  let remaining = need;
  return (
    <div className={className}>
      <div className="flex h-3 w-full gap-[3px]">
        {devices.map((g) => {
          const inPlay = usable.includes(g);
          const take = inPlay ? Math.min(remaining, g.vram_total) : 0;
          remaining -= take;
          const used = Math.min(g.vram_used, g.vram_total - take);
          return (
            <div
              key={g.index}
              className="relative h-full overflow-hidden rounded-[3px] bg-panel-3"
              style={{ width: `${(g.vram_total / scale) * 100}%` }}
              title={`${tpu ? "TPU" : "GPU"}${g.index}: ${bytes(g.vram_total)}`}
            >
              {used > 0 && (
                <div
                  className="absolute inset-y-0 right-0 opacity-50"
                  style={{ width: `${(used / g.vram_total) * 100}%`, background: "var(--line-2)" }}
                />
              )}
              <div
                className="absolute inset-y-0 left-0 rounded-[3px] transition-[width] duration-500"
                style={{ width: `${(take / g.vram_total) * 100}%`, background: color }}
              />
            </div>
          );
        })}
        {overflow > 0 && (
          <div
            className="hatch h-full rounded-[3px] text-worst"
            style={{ width: `${(overflow / scale) * 100}%` }}
            title={`${bytes(overflow)} more than the ${tpu ? "chips" : "GPUs"} hold`}
          />
        )}
      </div>
      <div className="mt-1.5 flex items-baseline justify-between gap-2 font-mono text-[10.5px] text-fg-2">
        <span>{need ? `≈${bytes(need)} ${unit}` : "size unknown"}</span>
        <span className="truncate text-fg-3">
          {fit.placement === "split"
            ? tpu
              ? `sharded · ${devices.length} chips`
              : `split · ${devices.length} GPUs`
            : fit.placement === "offload"
              ? "CPU offload"
              : fit.placement === "cpu"
                ? "CPU only"
                : fit.placement === "single"
                  ? tpu
                    ? "1 chip"
                    : "1 GPU"
                  : `${bytes(total)} total`}
        </span>
      </div>
    </div>
  );
}

export function ProgressBar({ value, tone = "signal", indeterminate }: { value: number | null; tone?: "signal" | "worst"; indeterminate?: boolean }) {
  const pct = value == null ? 100 : Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-panel-3">
      <div
        className={cx("h-full rounded-full transition-[width] duration-500", (indeterminate || value == null) && "hatch-move")}
        style={{ width: `${pct}%`, background: tone === "worst" ? "var(--worst)" : "var(--signal)", opacity: value == null ? 0.55 : 1 }}
      />
    </div>
  );
}
