import { ArrowRight, ArrowUpRight, Boxes, Download, FileCode2, KeyRound, Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router";

import { GpuMeter, LedMeter, ProgressBar } from "../components/meters";
import { ModelLogo } from "../components/ModelLogo";
import { Button, CopyButton, SectionTitle } from "../components/ui";
import { bytes, cx, duration, MODALITY_LABEL } from "../lib/format";
import { useLive } from "../lib/live";
import type { Deployment } from "../lib/types";

const STARTING = ["queued", "installing", "downloading", "loading"];

function headline(ready: number, starting: number): string {
  if (ready && starting) return `${ready} running, ${starting} starting`;
  if (ready) return `${ready} ${ready === 1 ? "model" : "models"} running`;
  if (starting) return `${starting} ${starting === 1 ? "model" : "models"} starting`;
  return "Nothing running yet";
}

function hm(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m`;
}

function StatusText({ d }: { d: Deployment }) {
  if (d.status === "ready") {
    return (
      <span className="inline-flex items-center gap-2 text-[12.5px] text-fg-2">
        <span className="dot dot-live" /> Ready
      </span>
    );
  }
  if (d.status === "error") {
    return (
      <span className="inline-flex items-center gap-2 text-[12.5px] text-worst">
        <span className="dot dot-bad" /> Failed
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-2 text-[12.5px] capitalize text-fg-2">
      <span className="dot dot-warn" /> {d.status}
    </span>
  );
}

function RunRow({ d }: { d: Deployment }) {
  const starting = STARTING.includes(d.status);
  return (
    <Link to={d.status === "ready" ? `/studio/${d.id}` : `/deployments#${d.id}`} className="run-row">
      <ModelLogo id={d.spec.model.id} name={d.name} source={d.spec.model.source} />
      <div className="min-w-0 flex-1">
        <div className="truncate text-[14px] font-medium">{d.name}</div>
        <div className="mt-0.5 truncate text-[12.5px] text-fg-3">
          {MODALITY_LABEL[d.modality]} · {d.spec.model.engine}
          {d.gpus.length ? ` · ${d.gpus.map((g) => `GPU${g}`).join(" + ")}` : ""}
        </div>
        {starting && (
          <div className="mt-2.5 max-w-sm">
            <ProgressBar value={d.progress} />
            <div className="mt-1 truncate text-[12px] text-fg-3">{d.message}</div>
          </div>
        )}
        {d.status === "error" && <p className="mt-1 line-clamp-1 text-[12.5px] text-worst">{d.error || d.message}</p>}
      </div>
      <div className="hidden shrink-0 sm:block">
        <StatusText d={d} />
      </div>
      <span className="go flex items-center gap-1 text-[12.5px]">
        <span className="hidden md:inline">{d.status === "ready" ? "Open" : "Details"}</span>
        <ArrowRight className="size-4" />
      </span>
    </Link>
  );
}

function GettingStarted() {
  return (
    <div className="panel p-6 sm:p-7">
      <h3 className="display text-[30px]">Three steps to a running model</h3>
      <ol className="start-steps mt-5">
        <li>
          <span className="text-[14px] font-medium">Pick a model that fits</span>
          <span className="text-[13px] text-fg-2">
            Every model in the catalog is sized against this machine's GPUs, RAM and disk before you download anything.
          </span>
        </li>
        <li>
          <span className="text-[14px] font-medium">Choose how to reach it</span>
          <span className="text-[13px] text-fg-2">Built-in Studio, Gradio or Open WebUI, with an optional password and API key.</span>
        </li>
        <li>
          <span className="text-[14px] font-medium">Use it from anywhere</span>
          <span className="text-[13px] text-fg-2">Chat in Studio, or call it with OpenAI, Anthropic or Ollama clients.</span>
        </li>
      </ol>
      <div className="mt-6 flex flex-wrap gap-2">
        <Link to="/models">
          <Button variant="primary">
            <Boxes className="size-4" /> Browse models
          </Button>
        </Link>
        <Link to="/imports">
          <Button variant="ghost">Import your own</Button>
        </Link>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { overview } = useLive();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => window.clearInterval(t);
  }, []);
  if (!overview) return null;
  const system = overview.system;
  const active = overview.deployments.filter((d) => !["stopped", "stopping"].includes(d.status));
  const ready = active.filter((d) => d.status === "ready");
  const starting = active.filter((d) => STARTING.includes(d.status));
  const disk = system.disks.find((d) => d.key === "models");
  const remaining = Math.max(0, system.session_started + system.session_limit - now / 1000);
  const tunnel = overview.tunnels.find((t) => t.name === "main");
  const publicUrl = tunnel?.url ?? overview.main_url;

  return (
    <div>
      <SectionTitle
        title={headline(ready.length, starting.length)}
        description={
          <>
            {system.accelerator_label}
            {system.gpus.length > 0 && <> with {bytes(system.vram_free_all)} of {bytes(system.vram_total_all)} GPU memory free</>}. About {hm(remaining)}{" "}
            left in this session.
          </>
        }
      >
        <Link to="/models">
          <Button>Browse models</Button>
        </Link>
        <Link to="/deploy/new">
          <Button variant="primary">
            <Plus className="size-4" /> Deploy a model
          </Button>
        </Link>
      </SectionTitle>

      <div className="overview-grid">
        <section className="min-w-0 stagger" aria-labelledby="running-heading">
          <div className="section-heading">
            <h2 id="running-heading">Running</h2>
            {active.length > 0 && <Link to="/deployments">All deployments →</Link>}
          </div>
          {active.length ? (
            <div className="run-list">
              {active.slice(0, 8).map((d) => (
                <RunRow key={d.id} d={d} />
              ))}
            </div>
          ) : (
            <GettingStarted />
          )}

          <div className="mt-10">
            <div className="section-heading">
              <h2>Shortcuts</h2>
            </div>
            <Link to="/imports" className="shortcut">
              <Download className="size-4 text-fg-3" />
              <div>
                <div className="text-[13.5px] font-medium">Import a checkpoint</div>
                <div className="text-[12.5px] text-fg-3">From a Hugging Face repo, Git URL or direct file link.</div>
              </div>
              <ArrowRight className="go size-4" />
            </Link>
            <Link to="/access" className="shortcut">
              <KeyRound className="size-4 text-fg-3" />
              <div>
                <div className="text-[13.5px] font-medium">Connect an app</div>
                <div className="text-[12.5px] text-fg-3">API keys, endpoints and the password for shared links.</div>
              </div>
              <ArrowRight className="go size-4" />
            </Link>
            <Link to="/deployments" className="shortcut">
              <FileCode2 className="size-4 text-fg-3" />
              <div>
                <div className="text-[13.5px] font-medium">Save this setup as a launch cell</div>
                <div className="text-[12.5px] text-fg-3">Recreate it next session, or run it unattended with Save & Run All.</div>
              </div>
              <ArrowRight className="go size-4" />
            </Link>
          </div>
        </section>

        <aside className="min-w-0" aria-labelledby="machine-heading">
          <div className="section-heading">
            <h2 id="machine-heading">This machine</h2>
            <Link to="/system">Details →</Link>
          </div>
          <div className="machine-card">
            <section>
              <div className="mb-4 flex items-baseline justify-between gap-3">
                <span className="text-[14px] font-medium">{system.accelerator_label}</span>
                <span className="text-[12px] text-fg-3">{system.platform}</span>
              </div>
              {system.gpus.length ? (
                <div className="space-y-4">
                  {system.gpus.map((gpu) => (
                    <GpuMeter key={gpu.index} gpu={gpu} />
                  ))}
                </div>
              ) : (
                <p className="text-[13px] text-fg-2">No GPU. Turn on an accelerator in the notebook settings (GPU T4 ×2 is a good default).</p>
              )}
            </section>
            <section className="space-y-3">
              <div>
                <div className="stat-line">
                  <span>Memory</span>
                  <b>{bytes(system.ram_available)} free</b>
                </div>
                <LedMeter value={(system.ram_total - system.ram_available) / system.ram_total} segments={30} className="mt-2 h-1.5" />
              </div>
              {disk && (
                <div>
                  <div className="stat-line">
                    <span>Model disk</span>
                    <b>{bytes(disk.free)} free</b>
                  </div>
                  <LedMeter value={disk.total ? disk.used / disk.total : 0} segments={30} className="mt-2 h-1.5" />
                </div>
              )}
              <div className="stat-line pt-1">
                <span>Session left</span>
                <b className={cx(remaining < system.session_limit * 0.1 && "text-worst")}>{duration(remaining)}</b>
              </div>
              <div className="stat-line">
                <span>Internet</span>
                <b className={cx(!system.internet && "text-worst")}>{system.internet ? "on" : "off"}</b>
              </div>
            </section>
            <section>
              <div className="stat-line mb-2">
                <span>Public link</span>
                <span className="text-[12px] text-fg-3">{tunnel?.label ?? "Local only"}</span>
              </div>
              <div className="flex h-9 items-center gap-1 rounded-lg border border-line bg-bg-2 pl-3 pr-1">
                <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-fg-2">{publicUrl.replace(/^https?:\/\//, "")}</span>
                <CopyButton text={publicUrl} />
                <a href={publicUrl} target="_blank" rel="noreferrer" className="toolbar-icon !size-7" aria-label="Open public link">
                  <ArrowUpRight className="size-3.5" />
                </a>
              </div>
            </section>
          </div>
          <p className="mt-4 px-1 text-[12px] leading-relaxed text-fg-3">
            Models live on the notebook's temporary disk and disappear when the session ends. Save a launch cell to bring them back.
          </p>
        </aside>
      </div>
    </div>
  );
}
