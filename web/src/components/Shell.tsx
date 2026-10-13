import {
  Boxes,
  Gauge,
  KeyRound,
  LogOut,
  Menu,
  Moon,
  Rocket,
  Server,
  Sparkles,
  Sun,
  X,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router";

import { api } from "../lib/api";
import { bytes, cx, duration } from "../lib/format";
import { useLive } from "../lib/live";
import { useTheme } from "../lib/theme";
import { GpuMeter, LedMeter } from "./meters";
import { CopyButton } from "./ui";

const NAV = [
  { to: "/", label: "Launch", icon: Rocket, end: true },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/deployments", label: "Deployments", icon: Server },
  { to: "/studio", label: "Studio", icon: Sparkles },
  { to: "/access", label: "Access & API", icon: KeyRound },
  { to: "/system", label: "System", icon: Gauge },
];

export function Brand({ small }: { small?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <svg viewBox="0 0 32 32" className={small ? "size-7" : "size-8"} aria-hidden>
        <rect width="32" height="32" rx="7" fill="var(--panel-3)" />
        <rect x="7" y="7" width="4" height="18" rx="1" fill="var(--fg)" />
        <rect x="14" y="13" width="4" height="12" rx="1" fill="var(--fg)" />
        <rect x="21" y="7" width="4" height="4" rx="1" fill="var(--signal)" />
        <rect x="21" y="14" width="4" height="11" rx="1" fill="var(--fg)" opacity=".55" />
      </svg>
      <span className="display text-[26px] leading-none">KaggleUI</span>
    </div>
  );
}

function SessionClock() {
  const { overview } = useLive();
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, []);
  if (!overview) return null;
  const s = overview.system;
  const remaining = Math.max(0, s.session_started + s.session_limit - now / 1000);
  const frac = remaining / s.session_limit;
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="readout">Session left</span>
        <span className={cx("font-mono text-[13px]", frac < 0.1 ? "text-worst" : frac < 0.25 ? "text-tight" : "text-fg")}>
          {duration(remaining)}
        </span>
      </div>
      <div className="h-1 overflow-hidden rounded-full bg-panel-3">
        <div className="h-full rounded-full bg-fg-3" style={{ width: `${frac * 100}%` }} />
      </div>
      <div className="mt-1 text-[11px] text-fg-3">
        ≈ from a {Math.round(s.session_limit / 3600)}h limit{s.run_type ? ` · ${s.run_type}` : ""}
      </div>
    </div>
  );
}

function TunnelStatus() {
  const { overview, connected } = useLive();
  const main = overview?.tunnels.find((t) => t.name === "main");
  const up = connected && (main?.status === "up" || !main);
  const url = overview?.main_url ?? "";
  return (
    <div className="rounded-lg border border-line bg-bg-2 p-2.5">
      <div className="mb-1 flex items-center gap-2">
        <span className={cx("size-2 rounded-full", up ? "bg-signal" : "bg-tight animate-blink")} />
        <span className="readout">{connected ? (main ? main.label : "Local") : "Reconnecting"}</span>
      </div>
      <div className="flex items-center gap-1">
        <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-fg-2" title={url}>
          {url.replace(/^https?:\/\//, "")}
        </span>
        {url && <CopyButton text={url} />}
      </div>
    </div>
  );
}

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  const { overview } = useLive();
  const ready = overview?.deployments.filter((d) => d.status === "ready").length ?? 0;
  const busy = overview?.deployments.filter((d) => ["queued", "installing", "downloading", "loading"].includes(d.status)).length ?? 0;
  return (
    <nav className="flex flex-col gap-0.5">
      {NAV.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          onClick={onNavigate}
          className={({ isActive }) =>
            cx(
              "group flex items-center gap-3 rounded-lg px-3 py-2 text-[14px] transition-colors",
              isActive ? "bg-panel-3 text-fg" : "text-fg-2 hover:bg-panel-2 hover:text-fg",
            )
          }
        >
          {({ isActive }) => (
            <>
              <Icon className={cx("size-[17px]", isActive ? "text-signal" : "")} strokeWidth={1.8} />
              <span className="flex-1">{label}</span>
              {to === "/deployments" && (ready > 0 || busy > 0) && (
                <span className="flex items-center gap-1 font-mono text-[10px]">
                  {busy > 0 && <span className="rounded bg-tight-soft px-1.5 py-px text-tight">{busy}</span>}
                  {ready > 0 && <span className="rounded bg-signal-soft px-1.5 py-px text-signal">{ready}</span>}
                </span>
              )}
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );
}

function SidebarFooter() {
  const [theme, toggle] = useTheme();
  return (
    <div className="flex flex-col gap-3">
      <SessionClock />
      <TunnelStatus />
      <div className="flex items-center gap-1">
        <button
          onClick={toggle}
          className="flex flex-1 items-center gap-2 rounded-lg px-2.5 py-2 text-[13px] text-fg-2 hover:bg-panel-2 hover:text-fg"
        >
          {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
          {theme === "dark" ? "Light" : "Dark"}
        </button>
        <button
          onClick={() => api.logout().then(() => location.assign("/"))}
          className="flex items-center gap-2 rounded-lg px-2.5 py-2 text-[13px] text-fg-2 hover:bg-panel-2 hover:text-fg"
          title="Sign out"
        >
          <LogOut className="size-4" />
        </button>
      </div>
    </div>
  );
}

function StatusStrip() {
  const { overview } = useLive();
  if (!overview) return <div className="h-[58px]" />;
  const s = overview.system;
  const disk = s.disks.find((d) => d.key === "models");
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-line bg-bg/70 px-5 py-3 backdrop-blur lg:px-8">
      <div className="mr-auto flex items-center gap-2.5">
        <span className="display text-[20px]">{s.accelerator_label}</span>
        {overview.fake_hw && <span className="rounded border border-info/40 px-1.5 py-px font-mono text-[9.5px] uppercase text-info">simulated</span>}
        {overview.mock && <span className="rounded border border-tight/40 px-1.5 py-px font-mono text-[9.5px] uppercase text-tight">mock engines</span>}
      </div>
      <div className="hidden flex-wrap items-center gap-5 md:flex">
        {s.gpus.map((g) => (
          <GpuMeter key={g.index} gpu={g} compact />
        ))}
        <div className="w-[118px]">
          <div className="mb-1 flex items-baseline justify-between">
            <span className="readout">RAM</span>
            <span className="font-mono text-[10px] text-fg-2">{bytes(s.ram_total - s.ram_available)}</span>
          </div>
          <LedMeter value={(s.ram_total - s.ram_available) / s.ram_total} segments={14} className="h-2.5" />
        </div>
        {disk && (
          <div className="w-[118px]">
            <div className="mb-1 flex items-baseline justify-between">
              <span className="readout">Disk</span>
              <span className="font-mono text-[10px] text-fg-2">{bytes(disk.free)} free</span>
            </div>
            <LedMeter value={disk.used / disk.total} segments={14} className="h-2.5" />
          </div>
        )}
      </div>
    </div>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => setOpen(false), [loc.pathname]);
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-[240px] shrink-0 flex-col border-r border-line bg-bg-2/80 px-3 py-5 backdrop-blur lg:flex">
        <div className="px-2 pb-6">
          <Brand />
        </div>
        <NavItems />
        <div className="mt-auto px-1">
          <SidebarFooter />
        </div>
      </aside>

      {/* Mobile top bar + slide-over nav */}
      <div className="fixed inset-x-0 top-0 z-40 flex h-14 items-center justify-between border-b border-line bg-bg/90 px-4 backdrop-blur lg:hidden">
        <Brand small />
        <button onClick={() => setOpen(true)} className="rounded-md p-2 text-fg-2 hover:bg-panel-2" aria-label="Menu">
          <Menu className="size-5" />
        </button>
      </div>
      {open && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/60" onClick={() => setOpen(false)} />
          <div className="absolute inset-y-0 right-0 flex w-[min(300px,85vw)] flex-col border-l border-line bg-panel px-3 py-4 animate-[rise_.2s_ease_both]">
            <div className="mb-4 flex items-center justify-between px-2">
              <Brand small />
              <button onClick={() => setOpen(false)} className="rounded-md p-2 text-fg-2 hover:bg-panel-2" aria-label="Close">
                <X className="size-5" />
              </button>
            </div>
            <NavItems onNavigate={() => setOpen(false)} />
            <div className="mt-auto px-1">
              <SidebarFooter />
            </div>
          </div>
        </div>
      )}

      <div className="min-w-0 flex-1 pt-14 lg:pt-0">
        <StatusStrip />
        <main className="mx-auto w-full max-w-[1320px] px-4 py-7 sm:px-6 lg:px-8 lg:py-9">{children}</main>
      </div>
    </div>
  );
}
