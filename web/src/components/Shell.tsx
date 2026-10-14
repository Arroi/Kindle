import { Boxes, Gauge, KeyRound, LogOut, Moon, Plus, Server, MessageSquare, Sun, LayoutDashboard, Download, Search } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router";

import { api } from "../lib/api";
import { bytes, cx, duration } from "../lib/format";
import { useLive } from "../lib/live";
import { useTheme } from "../lib/theme";
import { UpdateNotice } from "./Updates";
import { CommunityLinks } from "./CommunityLinks";
import { Button, Input } from "./ui";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/imports", label: "Imports", icon: Download },
  { to: "/deployments", label: "Deployments", icon: Server },
  { to: "/studio", label: "Studio", icon: MessageSquare },
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
      <span className="text-[19px] font-semibold tracking-tight">KaggleUI</span>
    </div>
  );
}

function QuickJump({ open, close }: { open: boolean; close: () => void }) {
  const { overview } = useLive();
  const [query, setQuery] = useState("");
  const ref = useRef<HTMLDialogElement>(null);
  const nav = useNavigate();
  useEffect(() => { if (open) { setQuery(""); ref.current?.showModal(); } else ref.current?.close(); }, [open]);
  const options = [...NAV, { to: "/deploy/new", label: "New deployment", icon: Plus },
    ...(overview?.deployments.filter(d => d.status === "ready").map(d => ({ to: `/studio/${d.id}`, label: d.name, icon: MessageSquare })) ?? [])]
    .filter(option => option.label.toLowerCase().includes(query.toLowerCase()));
  return <dialog aria-label="Quick navigation" ref={ref} onClose={close} onClick={e => { if (e.target === e.currentTarget) close(); }} className="command-dialog">
    <form method="dialog" onSubmit={e => { e.preventDefault(); if (options[0]) { nav(options[0].to); close(); } }}>
      <div className="flex items-center gap-3 border-b border-line p-4"><Search className="size-4 text-fg-3" /><Input autoFocus value={query} onChange={e => setQuery(e.target.value)} placeholder="Jump to a page or running model…" aria-label="Quick navigation" className="border-0 bg-transparent" /><Button type="button" size="sm" variant="ghost" onClick={close}>Esc</Button></div>
      <nav className="max-h-80 overflow-y-auto p-2">{options.map(({ to, label, icon: Icon }) => <Link key={to} to={to} onClick={close} className="flex items-center gap-3 rounded-md px-3 py-3 hover:bg-panel-2"><Icon className="size-4 text-fg-3" />{label}</Link>)}{!options.length && <p className="p-4 text-fg-3">No matching pages or running models.</p>}</nav>
    </form>
  </dialog>;
}

export function Shell({ children }: { children: ReactNode }) {
  const [theme, toggle] = useTheme();
  const [jump, setJump] = useState(false);
  const { overview, connected, error } = useLive();
  const location = useLocation();
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); setJump(value => !value); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  const system = overview?.system;
  const ready = overview?.deployments.filter(d => d.status === "ready").length ?? 0;
  return <div className="min-h-dvh">
    <a href="#main-content" className="skip-link">Skip to content</a>
    <header className="workspace-header">
      <div className="workspace-bar">
        <Link to="/" aria-label="KaggleUI home"><Brand small /></Link>
        <span className="hidden border-l border-line pl-5 text-[12px] text-fg-3 sm:block">Your model workspace</span>
        <div className="ml-auto flex items-center gap-1.5">
          <button onClick={() => setJump(true)} className="quick-jump" aria-label="Quick navigation (Control or Command K)"><Search className="size-3.5" /><span className="hidden sm:inline">Quick jump</span><kbd className="hidden text-[10px] text-fg-3 sm:inline">⌘ K</kbd></button>
          <button onClick={toggle} className="toolbar-icon" aria-label="Toggle theme">{theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}</button>
          <button onClick={() => api.logout().then(() => window.location.assign("/login"))} className="toolbar-icon" aria-label="Sign out"><LogOut className="size-4" /></button>
          <Link to="/deploy/new" className="ml-1"><Button size="sm" variant="primary"><Plus className="size-4" /><span className="hidden sm:inline">Deploy model</span></Button></Link>
        </div>
      </div>
      <div className="workspace-nav-wrap"><nav className="workspace-nav scroll-thin" aria-label="Workspace">{NAV.map(({ to, label, icon: Icon, end }) => <NavLink key={to} to={to} end={end} className={({ isActive }) => cx("workspace-link", isActive && "is-active")}><Icon className="size-3.5" strokeWidth={1.7} />{label}{to === "/deployments" && ready > 0 && <span className="nav-count">{ready}</span>}</NavLink>)}</nav></div>
    </header>
    <div className="resource-bar"><div className="resource-inner">
      <span className="inline-flex items-center gap-2"><span className={cx("size-1.5 rounded-full", connected ? "bg-signal" : "bg-tight")} />{connected ? "Live" : error ? "Connection interrupted" : "Polling · live connection unavailable"}</span>
      {system && <><span>{system.accelerator_label}</span><span className="hidden sm:inline">{bytes(system.vram_free_all)} VRAM free</span><span className="hidden md:inline">{bytes(system.ram_available)} RAM free</span><span className="ml-auto">≈ {duration(system.session_remaining)} left</span></>}
      {overview?.mock && <span className="text-tight">Demo engines</span>}
    </div></div>
    <main id="main-content" tabIndex={-1} className="workspace-main"><UpdateNotice /><div key={location.pathname} className="page-enter">{children}</div></main>
    <footer className="workspace-footer"><span>KaggleUI {__APP_VERSION__}</span><a href="https://github.com/UltraFEmotes/KaggleUI" target="_blank" rel="noreferrer">Source & help ↗</a><CommunityLinks /></footer>
    <QuickJump open={jump} close={() => setJump(false)} />
  </div>;
}
