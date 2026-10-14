import { ArrowRight, Download, Plus, KeyRound, ChevronRight, CircleHelp } from "lucide-react";
import { Link } from "react-router";

import { GpuMeter, ProgressBar } from "../components/meters";
import { ModelLogo } from "../components/ModelLogo";
import { Button } from "../components/ui";
import { bytes, cx, MODALITY_LABEL } from "../lib/format";
import { useLive } from "../lib/live";

export default function Dashboard() {
  const { overview } = useLive();
  if (!overview) return null;
  const system = overview.system;
  const active = overview.deployments.filter(d => !["stopped", "stopping"].includes(d.status));
  const ready = active.filter(d => d.status === "ready");
  const pending = active.filter(d => ["queued", "installing", "downloading", "loading"].includes(d.status));
  const disk = system.disks.find(d => d.key === "models");
  return <div className="stagger">
    <div className="dashboard-heading">
      <div><p className="eyebrow">Workspace overview</p><h1 className="display text-[34px] sm:text-[44px]">A place for your models.</h1><p className="mt-3 max-w-xl text-[15px] text-fg-2">{ready.length ? `${ready.length} ${ready.length === 1 ? "model is" : "models are"} ready. Open Studio to start creating, or add something new.` : "Choose a model, put your GPU to work, and open it in Studio."}</p></div>
      <Link to="/deploy/new"><Button variant="primary"><Plus className="size-4" /> New deployment</Button></Link>
    </div>

    <div className="workspace-facts">
      <div><span className="eyebrow">Ready models</span><strong>{ready.length.toString().padStart(2, "0")}<small>{pending.length ? `${pending.length} starting` : "Ready to use"}</small></strong></div>
      <div><span className="eyebrow">GPU memory</span><strong>{bytes(system.vram_free_all)}<small>free of {bytes(system.vram_total_all)}</small></strong></div>
      <div><span className="eyebrow">System memory</span><strong>{bytes(system.ram_available)}<small>RAM available</small></strong></div>
      <div><span className="eyebrow">Model storage</span><strong>{bytes(disk?.free)}<small>free on this session</small></strong></div>
    </div>

    <div className="dashboard-grid">
      <section className="min-w-0">
        <div className="section-heading"><h2>Your deployments</h2><Link to="/deployments" className="inline-flex items-center gap-1 text-[12px] text-fg-2 hover:text-fg">View all <ArrowRight className="size-3.5" /></Link></div>
        {active.length ? <div className="deployment-list">{active.slice(0, 6).map(d => <Link key={d.id} to={d.status === "ready" ? `/studio/${d.id}` : `/deployments#${d.id}`} className="deployment-summary">
          <ModelLogo id={d.spec.model.id} name={d.name} source={d.spec.model.source} className="size-11" />
          <div className="min-w-0 flex-1"><h3 className="truncate font-semibold">{d.name}</h3><p className="mt-0.5 truncate text-[12px] text-fg-3">{MODALITY_LABEL[d.modality]} · {d.spec.model.engine}</p>{!["ready", "error"].includes(d.status) && <div className="mt-2 max-w-xs"><ProgressBar value={d.progress} /></div>}{d.status === "error" && <p className="mt-1 line-clamp-1 text-[12px] text-worst">{d.error || d.message}</p>}</div>
          <span className={cx("hidden text-[12px] sm:inline", d.status === "ready" ? "text-signal" : d.status === "error" ? "text-worst" : "text-fg-3")}>{d.status === "ready" ? "Open in Studio" : d.status}</span><ChevronRight className="size-4 text-fg-3" />
        </Link>)}</div> : <div className="getting-started">
          <span className="eyebrow">Your first deployment</span><h3 className="display mt-4 text-[28px]">Start with something small.</h3><p className="mt-3 max-w-md text-[14px] text-fg-2">Browse the catalog and filter by “Fits this machine” for models your current hardware can handle. Setup and downloads happen here.</p><Link to="/models" className="mt-6 inline-flex items-center gap-2 text-[14px] font-medium text-signal">Find a model <ArrowRight className="size-4" /></Link>
        </div>}
        <div className="mt-7 grid gap-3 sm:grid-cols-2">
          <Link to="/imports" className="workspace-shortcut"><Download className="size-5 text-fg-2" /><div><h3>Bring your own model</h3><p>Import a Hugging Face, Git, or file URL.</p></div><ArrowRight className="ml-auto size-4 shrink-0 text-fg-3" /></Link>
          <Link to="/access" className="workspace-shortcut"><KeyRound className="size-5 text-fg-2" /><div><h3>Connect an app</h3><p>API keys, endpoints, and sharing.</p></div><ArrowRight className="ml-auto size-4 shrink-0 text-fg-3" /></Link>
        </div>
      </section>
      <aside className="min-w-0">
        <div className="section-heading"><h2>This session</h2><Link to="/system" className="text-[12px] text-fg-3 hover:text-fg">Details ↗</Link></div>
        <div className="session-panel"><div className="mb-5 flex items-center justify-between"><span className="font-medium">{system.accelerator_label}</span><span className="eyebrow">{system.platform}</span></div>
          {system.gpus.length ? <div className="space-y-5">{system.gpus.map(gpu => <GpuMeter key={gpu.index} gpu={gpu} />)}</div> : <p className="text-[13px] text-fg-2">Enable a GPU in your notebook settings to run larger models.</p>}
          <div className="mt-6 space-y-2 border-t border-line pt-4 text-[12px] text-fg-2"><div className="flex justify-between"><span>Internet</span><span>{system.internet ? "Connected" : "Unavailable"}</span></div><div className="flex justify-between"><span>Public link</span><span>{overview.tunnels.find(t => t.name === "main")?.label ?? "Local"}</span></div></div>
        </div>
        <div className="mt-5 flex gap-2.5 px-1 text-[12px] leading-relaxed text-fg-3"><CircleHelp className="mt-0.5 size-4 shrink-0" /><p>Models and downloads live on the notebook’s temporary disk. Save a preset in Deployments to recreate your workspace next session.</p></div>
      </aside>
    </div>
  </div>;
}
