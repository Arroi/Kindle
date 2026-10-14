import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowDownToLine, ArrowUpRight, RefreshCw } from "lucide-react";
import { useState } from "react";

import { api } from "../lib/api";
import { Button, useToast } from "./ui";

export function UpdateNotice({ full = false }: { full?: boolean }) {
  const cache = useQueryClient();
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const [logs, setLogs] = useState<string[] | null>(null);
  const query = useQuery({ queryKey: ["updates"], queryFn: () => api.updates(), staleTime: 60_000,
    refetchInterval: q => ["installing", "restarting"].includes(q.state.data?.progress?.status ?? "") ? 1500 : 600_000 });
  const update = query.data;
  const running = ["installing", "restarting"].includes(update?.progress?.status ?? "");
  if (!full && !update?.available && !update?.blocked && !running) return null;
  return <section className={full ? "panel mb-6 p-5" : "update-notice mb-6"} aria-label="KaggleUI updates">
    <div className="flex flex-wrap items-center gap-4">
      <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-signal-soft text-signal"><ArrowDownToLine className="size-4" /></span>
      <div className="min-w-0 flex-1">
        <h2 className="text-[14px] font-semibold">{update?.blocked ? "Update required" : update?.available ? `KaggleUI ${update.latest} is available` : `KaggleUI ${update?.current ?? __APP_VERSION__}`}</h2>
        <p className="mt-0.5 text-[13px] text-fg-2">{running ? update?.progress?.message : update?.blocked ? update.message : update?.available ? "Install the latest release and restart your workspace." : "Your workspace is up to date."}</p>
      </div>
      {update?.available && !running && <Button size="sm" variant="primary" busy={busy} onClick={async () => {
        setBusy(true);
        try { const data = await api.installUpdate(); cache.setQueryData(["updates"], data); }
        catch (error) { toast(error instanceof Error ? error.message : "Update failed", "error"); }
        finally { setBusy(false); }
      }}>Update to {update.latest}</Button>}
      {full && <Button size="sm" variant="ghost" busy={query.isFetching} onClick={async () => {
        try { cache.setQueryData(["updates"], await api.updates(true)); }
        catch (error) { toast(error instanceof Error ? error.message : "Couldn't check releases", "error"); }
      }}><RefreshCw className="size-3.5" /> Check</Button>}
      {update?.release_url && <a href={update.release_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[12px] text-fg-2 hover:text-fg">Release notes <ArrowUpRight className="size-3.5" /></a>}
    </div>
    {(full || running || update?.available) && <p className="mt-3 text-[12px] text-fg-3">Updating interrupts active requests and relaunches your deployments. With a temporary tunnel, re-run <code>kaggleui.launch()</code> in the notebook to get the new link. Credentials and downloaded models are kept.</p>}
    {update?.progress?.status === "error" && <p role="alert" className="mt-2 text-[13px] text-worst">{update.progress.message}</p>}
    {full && query.error && <p role="alert" className="mt-2 text-[12px] text-tight">{query.error.message}</p>}
    {full && update?.error && <p className="mt-2 text-[12px] text-tight">{update.error}</p>}
    {full && <div className="mt-3"><Button variant="ghost" size="sm" onClick={async () => {
      try { setLogs(logs ? null : (await api.updateLogs()).lines); }
      catch (error) { toast(error instanceof Error ? error.message : "Couldn't read update log", "error"); }
    }}>{logs ? "Hide update log" : "Show update log"}</Button>{logs && <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-bg-2 p-3 font-mono text-[10px] text-fg-2">{logs.join("\n") || "No update activity yet."}</pre>}</div>}
    {full && <p className="mt-3 text-[12px] text-fg-3">Minimum supported version: {update?.minimum_supported ?? "checking…"}. Releases that support compatibility checks stop serving models when revoked. Releases without this check cannot be remotely disabled.</p>}
  </section>;
}
