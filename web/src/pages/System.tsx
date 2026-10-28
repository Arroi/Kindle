import { AlertTriangle, Check, Cpu, HardDrive, KeyRound, Wifi, WifiOff, X } from "lucide-react";

import { GpuMeter, LedMeter } from "../components/meters";
import { SectionTitle } from "../components/ui";
import { UpdateNotice } from "../components/Updates";
import { bytes, cx, duration } from "../lib/format";
import { useLive } from "../lib/live";

const SECRET_HELP: Record<string, string> = {
  HF_TOKEN: "Gated Hugging Face models (FLUX, Llama, SD 3.5…)",
  NGROK_AUTHTOKEN: "ngrok tunnels with real-time streaming",
  NGROK_DOMAIN: "Your free static ngrok domain",
  CF_TUNNEL_TOKEN: "Cloudflare named tunnel",
  CF_TUNNEL_HOSTNAME: "Hostname for the named tunnel",
  BURROW_TOKEN: "burrow tunnel on your own domain",
  BURROW_HOSTNAME: "Domain for the burrow tunnel",
  KINDLE_PASSWORD: "Fixed access password",
  KINDLE_ADMIN_TOKEN: "Fixed admin token",
  KINDLE_API_KEY: "Fixed API key",
  KINDLE_NOTIFY_URL: "Push the URL to ntfy.sh or a Discord webhook",
};

export default function System() {
  const { overview } = useLive();
  if (!overview) return null;
  const s = overview.system;
  const devices = s.gpus.length ? s.gpus : s.tpus ?? [];
  return (
    <div>
      <SectionTitle title={s.accelerator_label} description={`${s.platform}. Hardware, storage, session time and the secrets Kindle can see.`} />
      <UpdateNotice full />
      {s.notes.length > 0 && (
        <div className="mb-4 space-y-2">
          {s.notes.map((n) => (
            <div key={n} className="flex items-start gap-2 rounded-lg border border-tight/40 bg-tight-soft px-3.5 py-2.5 text-[13.5px] text-tight">
              <AlertTriangle className="mt-0.5 size-4 shrink-0" /> {n}
            </div>
          ))}
        </div>
      )}
      <div className="grid gap-4 lg:grid-cols-3">
        <section className="panel min-w-0 p-5 lg:col-span-2">
          <div className="readout mb-4">Accelerators</div>
          {devices.length === 0 ? (
            <p className="text-fg-2">No GPU or TPU detected.</p>
          ) : (
            <div className="grid gap-5 sm:grid-cols-2">
              {devices.map((g) => (
                <div key={g.index} className="rounded-xl border border-line bg-bg-2 p-4">
                  <div className="mb-3 flex items-baseline justify-between">
                    <span className="display text-[18px]">{g.name.replace("NVIDIA ", "")}</span>
                    <span className="font-mono text-[11px] text-fg-3">{g.kind === "tpu" ? "HBM" : `sm_${g.sm ?? "?"}`}</span>
                  </div>
                  <GpuMeter gpu={g} />
                </div>
              ))}
            </div>
          )}
          <div className="mt-6 grid gap-5 sm:grid-cols-2">
            <div>
              <div className="mb-2 flex items-center gap-2">
                <Cpu className="size-4 text-fg-2" />
                <span className="readout">CPU & RAM</span>
              </div>
              <LedMeter value={(s.ram_total - s.ram_available) / s.ram_total} segments={28} className="h-2.5" />
              <div className="mt-1.5 flex justify-between font-mono text-[11px] text-fg-2">
                <span>
                  {bytes(s.ram_total - s.ram_available)} / {bytes(s.ram_total)}
                </span>
                <span>{s.cpu_count} cores</span>
              </div>
            </div>
            <div>
              <div className="mb-2 flex items-center gap-2">
                {s.internet ? <Wifi className="size-4 text-signal" /> : <WifiOff className="size-4 text-worst" />}
                <span className="readout">Session</span>
              </div>
              <div className="space-y-1 font-mono text-[12px] text-fg-2">
                <div className="flex justify-between"><span>elapsed</span><span>{duration(s.session_elapsed)}</span></div>
                <div className="flex justify-between"><span>≈ remaining</span><span>{duration(s.session_remaining)}</span></div>
                <div className="flex justify-between"><span>run type</span><span>{s.run_type ?? "—"}</span></div>
                <div className="flex justify-between"><span>internet</span><span>{s.internet ? "on" : "off"}</span></div>
              </div>
            </div>
          </div>
        </section>

        <section className="panel min-w-0 p-5">
          <div className="mb-4 flex items-center gap-2">
            <HardDrive className="size-4 text-fg-2" />
            <span className="readout">Storage</span>
          </div>
          <div className="space-y-5">
            {s.disks.map((d) => (
              <div key={d.key}>
                <div className="mb-1 flex items-baseline justify-between gap-2">
                  <span className="font-medium">{d.label}</span>
                  <span className="font-mono text-[11px] text-fg-2">{bytes(d.free)} free</span>
                </div>
                {d.total > 0 && <LedMeter value={d.used / d.total} segments={24} className="h-2" />}
                <div className="mt-1 flex justify-between gap-2 text-[11.5px] text-fg-3">
                  <span>{d.note}</span>
                  {d.total > 0 && <span className="shrink-0 font-mono">{bytes(d.used)} / {bytes(d.total)}</span>}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="panel min-w-0 p-5 lg:col-span-3">
          <div className="mb-1 flex items-center gap-2">
            <KeyRound className="size-4 text-fg-2" />
            <span className="readout">Kaggle secrets</span>
          </div>
          <p className="mb-4 text-[13px] text-fg-3">
            Optional. Set them in <a href="/access#secrets" className="text-fg-2 underline underline-offset-4 hover:text-fg">Access & API → Keys & tokens</a>, or add
            them as Kaggle secrets (<b>Add-ons → Secrets</b>, switched on for this notebook) and press Reload there.
          </p>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(overview.secrets).map(([name, on]) => (
              <div key={name} className={cx("flex items-start gap-2.5 rounded-lg border px-3 py-2.5", on ? "border-signal/40" : "border-line")}>
                {on ? <Check className="mt-0.5 size-4 shrink-0 text-signal" /> : <X className="mt-0.5 size-4 shrink-0 text-fg-3" />}
                <span className="min-w-0">
                  <code className="block truncate font-mono text-[11.5px]">{name}</code>
                  <span className="block text-[12px] text-fg-3">{SECRET_HELP[name]}</span>
                </span>
              </div>
            ))}
          </div>
          <div className="mt-4 font-mono text-[10.5px] text-fg-3">Kindle {overview.version}</div>
        </section>
      </div>
    </div>
  );
}
