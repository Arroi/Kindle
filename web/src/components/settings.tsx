import { AlertTriangle, Check, Cloud, Globe2, Laptop, RefreshCw, Zap } from "lucide-react";
import type { ReactNode } from "react";

import { api } from "../lib/api";
import { cx } from "../lib/format";
import { useLive } from "../lib/live";
import type { Settings, TunnelProvider } from "../lib/types";
import { Button, CopyField, Toggle, useToast } from "./ui";

export function OptionCard({
  active,
  disabled,
  onClick,
  icon,
  title,
  children,
  badge,
}: {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  icon?: ReactNode;
  title: ReactNode;
  children?: ReactNode;
  badge?: ReactNode;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={cx(
        "relative flex w-full items-start gap-3 rounded-xl border p-3.5 text-left transition-colors",
        active ? "border-signal bg-signal-soft" : "border-line bg-bg-2 hover:border-line-2",
        disabled && "cursor-not-allowed opacity-45 hover:border-line",
      )}
    >
      {icon && <span className={cx("mt-0.5", active ? "text-signal" : "text-fg-2")}>{icon}</span>}
      <span className="min-w-0 flex-1">
        <span className="flex flex-wrap items-center gap-2 font-medium">
          {title}
          {badge}
        </span>
        {children && <span className="mt-0.5 block text-[13px] leading-snug text-fg-2">{children}</span>}
      </span>
      <span
        className={cx(
          "mt-1 grid size-4 shrink-0 place-items-center rounded-full border",
          active ? "border-signal bg-signal text-signal-ink" : "border-line-2",
        )}
      >
        {active && <Check className="size-3" strokeWidth={3} />}
      </span>
    </button>
  );
}

export function ExposurePicker({ value, onChange }: { value: TunnelProvider; onChange: (v: TunnelProvider) => void }) {
  const { overview } = useLive();
  const sec = overview?.secrets ?? {};
  const named = Boolean(sec.CF_TUNNEL_TOKEN && sec.CF_TUNNEL_HOSTNAME);
  return (
    <div className="grid gap-2 sm:grid-cols-2">
      <OptionCard active={value === "cloudflare"} onClick={() => onChange("cloudflare")} icon={<Cloud className="size-4" />} title="Cloudflare quick tunnel" badge={<Tag>default</Tag>}>
        No account needed. App updates keep your link. A fresh notebook session or tunnel reconnect changes it. API streaming arrives in one chunk.
      </OptionCard>
      <OptionCard
        active={value === "cloudflare_named"}
        disabled={!named}
        onClick={() => onChange("cloudflare_named")}
        icon={<Globe2 className="size-4" />}
        title="Cloudflare named tunnel"
        badge={<Tag tone="signal">stable URL</Tag>}
      >
        {named ? "Your own hostname, with real-time streaming." : "Add CF_TUNNEL_TOKEN and CF_TUNNEL_HOSTNAME secrets to enable."}
      </OptionCard>
      <OptionCard
        active={value === "ngrok"}
        disabled={!sec.NGROK_AUTHTOKEN}
        onClick={() => onChange("ngrok")}
        icon={<Zap className="size-4" />}
        title="ngrok"
        badge={<Tag tone="signal">streams</Tag>}
      >
        {sec.NGROK_AUTHTOKEN ? "Real-time streaming; free static domain via NGROK_DOMAIN." : "Add an NGROK_AUTHTOKEN secret to enable."}
      </OptionCard>
      <OptionCard active={value === "none"} onClick={() => onChange("none")} icon={<Laptop className="size-4" />} title="Local only">
        No public URL. Only reachable from inside the notebook.
      </OptionCard>
    </div>
  );
}

function Tag({ children, tone = "plain" }: { children: ReactNode; tone?: "plain" | "signal" }) {
  return (
    <span className={cx("rounded-[5px] px-1.5 py-px text-[11px] font-medium", tone === "signal" ? "bg-good-soft text-good" : "bg-panel-3 text-fg-3")}>
      {children}
    </span>
  );
}

export function AccessForm({ enabled, onChange, showSecret = true }: { enabled: boolean; onChange: (v: boolean) => void; showSecret?: boolean }) {
  const { overview, dispatch } = useLive();
  const toast = useToast();
  return (
    <div className="space-y-4">
      <Toggle
        checked={enabled}
        onChange={onChange}
        label="Password-lock the interfaces"
        hint="Studio, Gradio, Open WebUI and ComfyUI ask for this password. The control panel always needs your admin token."
      />
      {enabled && showSecret && overview && (
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
          <div className="min-w-0 flex-1">
            <CopyField label="Access password" value={overview.credentials.password} secret />
          </div>
          <Button
            size="md"
            onClick={async () => {
              const creds = await api.regenerate("password");
              dispatch({ type: "credentials", data: creds });
              toast("New password generated. Anyone signed in with the old one is signed out.");
            }}
          >
            <RefreshCw className="size-3.5" /> New password
          </Button>
        </div>
      )}
    </div>
  );
}

export type ApiState = Settings["api"];

export function ApiForm({ value, onChange, showSecret = true }: { value: ApiState; onChange: (v: ApiState) => void; showSecret?: boolean }) {
  const { overview, dispatch } = useLive();
  const toast = useToast();
  const set = (patch: Partial<ApiState>) => onChange({ ...value, ...patch });
  const surfaces: { key: "openai" | "anthropic" | "ollama"; title: string; desc: string }[] = [
    { key: "openai", title: "OpenAI-compatible", desc: "/v1/chat/completions, embeddings, images, audio, videos" },
    { key: "anthropic", title: "Anthropic-compatible", desc: "/v1/messages, with tools and streaming (works with Claude-style SDKs)" },
    { key: "ollama", title: "Ollama-compatible", desc: "/api/chat, /api/generate, /api/tags for Ollama apps" },
  ];
  return (
    <div className="space-y-4">
      <Toggle checked={value.enabled} onChange={(v) => set({ enabled: v })} label="Expose an API" hint="Call your models from code or other apps over the public URL." />
      {value.enabled && (
        <>
          <div className="grid gap-2 md:grid-cols-3">
            {surfaces.map((s) => (
              <OptionCard key={s.key} active={value[s.key]} onClick={() => set({ [s.key]: !value[s.key] } as Partial<ApiState>)} title={s.title}>
                <span className="font-mono text-[11px]">{s.desc}</span>
              </OptionCard>
            ))}
          </div>
          <Toggle checked={value.require_key} onChange={(v) => set({ require_key: v })} label="Require an API key" hint="Strongly recommended: tunnel URLs are public." />
          {value.require_key && showSecret && overview && (
            <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
              <div className="min-w-0 flex-1">
                <CopyField label="API key" value={overview.credentials.api_key} secret />
              </div>
              <Button
                onClick={async () => {
                  const creds = await api.regenerate("api_key");
                  dispatch({ type: "credentials", data: creds });
                  toast("New API key generated. The old key stops working now.");
                }}
              >
                <RefreshCw className="size-3.5" /> New key
              </Button>
            </div>
          )}
          {!value.require_key && (
            <div className="flex items-start gap-2 rounded-lg border border-worst/40 bg-worst-soft px-3 py-2 text-[13px] text-worst">
              <AlertTriangle className="mt-0.5 size-4 shrink-0" />
              Anyone who finds the URL can use your GPU through the API.
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function StreamingNote() {
  const { overview } = useLive();
  if (!overview || overview.streaming_ok || overview.settings.exposure.provider === "none") return null;
  return (
    <div className="flex items-start gap-2 rounded-lg border border-tight/40 bg-tight-soft px-3 py-2.5 text-[13px] text-tight">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      <span>
        Cloudflare quick tunnels buffer streaming HTTP. With <code className="font-mono">stream: true</code> the reply still arrives, but all at
        once at the end. The Studio streams token by token over WebSockets. For live API streaming, switch to ngrok or a named tunnel.
      </span>
    </div>
  );
}
