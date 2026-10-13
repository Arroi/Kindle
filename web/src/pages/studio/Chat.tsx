import { Brain, ChevronDown, ImagePlus, Send, SlidersHorizontal, Square, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button, Field, Segmented, Select, Slider, Textarea } from "../../components/ui";
import { cx } from "../../lib/format";
import { renderMarkdown } from "../../lib/markdown";
import type { StudioModel } from "../../lib/types";

interface Msg {
  role: "user" | "assistant";
  content: string;
  images?: string[];
  reasoning?: string;
  stats?: { tps: number | null; ttft: number; tokens?: number };
  error?: string;
  pending?: boolean;
}

type Handler = (msg: any) => void;

type Effort = "default" | "none" | "low" | "medium" | "high";
const EFFORTS: { value: Effort; label: string }[] = [
  { value: "default", label: "Auto" },
  { value: "none", label: "Off" },
  { value: "low", label: "Low" },
  { value: "medium", label: "Med" },
  { value: "high", label: "High" },
];

/** One WebSocket per Studio, multiplexed by request id. */
function useStudioSocket() {
  const ws = useRef<WebSocket | null>(null);
  const handlers = useRef(new Map<string, Handler>());
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let closed = false;
    let timer: number | undefined;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const sock = new WebSocket(`${proto}://${location.host}/_kui/studio/ws`);
      ws.current = sock;
      sock.onopen = () => setReady(true);
      sock.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        handlers.current.get(msg.id)?.(msg);
      };
      sock.onclose = () => {
        setReady(false);
        for (const h of handlers.current.values()) h({ op: "error", message: "Connection lost" });
        handlers.current.clear();
        if (!closed) timer = window.setTimeout(connect, 1500);
      };
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      ws.current?.close();
    };
  }, []);

  const send = useCallback((payload: object, onMsg: Handler) => {
    const id = Math.random().toString(36).slice(2);
    handlers.current.set(id, onMsg);
    ws.current?.send(JSON.stringify({ ...payload, id }));
    return id;
  }, []);
  const cancel = useCallback((id: string) => ws.current?.send(JSON.stringify({ op: "cancel", id })), []);
  const done = useCallback((id: string) => handlers.current.delete(id), []);
  return { ready, send, cancel, done };
}

function toDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

function Reasoning({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mb-2 rounded-lg border border-line bg-bg-2">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 px-3 py-2 text-[12.5px] text-fg-2 hover:text-fg">
        <Brain className={cx("size-3.5", live && "animate-pulse text-signal")} />
        {live ? "Thinking…" : "Thought process"}
        <ChevronDown className={cx("ml-auto size-3.5 transition-transform", open && "rotate-180")} />
      </button>
      {open && <div className="whitespace-pre-wrap border-t border-line px-3 py-2 font-mono text-[11.5px] leading-relaxed text-fg-3">{text}</div>}
    </div>
  );
}

export default function ChatStudio({ model }: { model: StudioModel }) {
  const { ready, send, cancel, done } = useStudioSocket();
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [current, setCurrent] = useState<string | null>(null);
  const [showParams, setShowParams] = useState(false);
  const [system, setSystem] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [effort, setEffort] = useState<Effort>("default");
  const scroller = useRef<HTMLDivElement>(null);
  const vision = model.capabilities.includes("vision");
  const thinking = model.capabilities.includes("thinking");

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [messages]);

  function submit() {
    const text = input.trim();
    if ((!text && !images.length) || current || !ready) return;
    const user: Msg = { role: "user", content: text, images };
    const history = [...messages.filter((m) => !m.error), user];
    setMessages([...history, { role: "assistant", content: "", pending: true }]);
    setInput("");
    setImages([]);
    const apiMessages = [
      ...(system.trim() ? [{ role: "system", content: system.trim() }] : []),
      ...history.map((m) =>
        m.images?.length
          ? { role: m.role, content: [{ type: "text", text: m.content }, ...m.images.map((url) => ({ type: "image_url", image_url: { url } }))] }
          : { role: m.role, content: m.content },
      ),
    ];
    const id = send(
      {
        op: "chat",
        model: model.name,
        messages: apiMessages,
        params: { temperature, max_tokens: maxTokens, ...(effort !== "default" ? { reasoning_effort: effort } : {}) },
      },
      (ev) => {
        setMessages((prev) => {
          const next = [...prev];
          const last = { ...next[next.length - 1] };
          if (ev.op === "delta") {
            if (ev.content) last.content += ev.content;
            if (ev.reasoning) last.reasoning = (last.reasoning ?? "") + ev.reasoning;
          } else if (ev.op === "done") {
            last.pending = false;
            last.stats = { tps: ev.tps, ttft: ev.ttft, tokens: ev.usage?.completion_tokens };
          } else if (ev.op === "error") {
            last.pending = false;
            last.error = ev.message;
          }
          next[next.length - 1] = last;
          return next;
        });
        if (ev.op === "done" || ev.op === "error") {
          setCurrent(null);
          done(id);
        }
      },
    );
    setCurrent(id);
  }

  return (
    <div className="panel flex h-[calc(100vh-260px)] min-h-[520px] flex-col overflow-hidden">
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <div className="flex items-center gap-2 text-[12.5px] text-fg-2">
          <span className={cx("size-2 rounded-full", ready ? "bg-signal" : "bg-tight animate-blink")} />
          {ready ? "Streaming over WebSocket" : "Connecting…"}
        </div>
        <div className="flex flex-wrap items-center justify-end gap-1">
          {thinking && (
            <div className="mr-1 flex items-center gap-2" title="How much the model reasons before answering">
              <span className="readout hidden sm:inline">Thinking</span>
              <Segmented<Effort> size="sm" value={effort} onChange={setEffort} options={EFFORTS} />
            </div>
          )}
          <Button size="sm" variant="ghost" onClick={() => setShowParams(!showParams)}>
            <SlidersHorizontal className="size-3.5" /> Settings
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setMessages([])} disabled={!messages.length || !!current}>
            <Trash2 className="size-3.5" /> Clear
          </Button>
        </div>
      </div>
      {showParams && (
        <div className="grid gap-4 border-b border-line bg-bg-2 px-4 py-4 md:grid-cols-[2fr_1fr_1fr_1fr]">
          <Field label="System prompt">
            <Textarea rows={2} value={system} onChange={(e) => setSystem(e.target.value)} placeholder="You are a helpful assistant." />
          </Field>
          <Slider label="Temperature" value={temperature} min={0} max={2} step={0.05} onChange={setTemperature} format={(v) => v.toFixed(2)} />
          <Slider label="Max tokens" value={maxTokens} min={64} max={16384} step={64} onChange={setMaxTokens} />
          <Field label="Reasoning effort" hint="For thinking models (Qwen3, gpt-oss, DeepSeek-R1…). Off = answer directly.">
            <Select value={effort} onChange={(e) => setEffort(e.target.value as Effort)}>
              <option value="default">Model default</option>
              <option value="none">Off (no thinking)</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </Select>
          </Field>
        </div>
      )}

      <div ref={scroller} className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-5 scroll-thin sm:px-6">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <div className="display text-[44px] text-fg-3">{model.name}</div>
            <p className="mt-2 max-w-sm text-[13.5px] text-fg-3">Say something. Replies stream token by token, even through a quick tunnel.</p>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-[85%] rounded-2xl rounded-br-md border border-line bg-panel-3 px-4 py-2.5 text-[14.5px] whitespace-pre-wrap">
                {m.images?.map((src) => <img key={src.slice(-24)} src={src} alt="" className="mb-2 max-h-48 rounded-lg" />)}
                {m.content}
              </div>
            </div>
          ) : (
            <div key={i} className="max-w-[92%]">
              {m.reasoning && <Reasoning text={m.reasoning} live={!!m.pending && !m.content} />}
              {m.content ? (
                <div className="prose-chat text-[14.5px] leading-relaxed" dangerouslySetInnerHTML={{ __html: renderMarkdown(m.content) }} />
              ) : m.pending && !m.reasoning ? (
                <span className="inline-block h-4 w-2 bg-signal animate-blink" />
              ) : null}
              {m.error && <div className="mt-2 rounded-lg border border-worst/40 bg-worst-soft px-3 py-2 text-[13px] text-worst">{m.error}</div>}
              {m.stats && (
                <div className="mt-2 flex gap-3 font-mono text-[10.5px] text-fg-3">
                  {m.stats.tokens ? <span>{m.stats.tokens} tok</span> : null}
                  {m.stats.tps ? <span>{m.stats.tps} tok/s</span> : null}
                  <span>first token {m.stats.ttft}s</span>
                </div>
              )}
            </div>
          ),
        )}
      </div>

      <div className="border-t border-line p-3">
        {images.length > 0 && (
          <div className="mb-2 flex gap-2">
            {images.map((src, i) => (
              <div key={i} className="relative">
                <img src={src} alt="" className="size-14 rounded-md object-cover" />
                <button onClick={() => setImages(images.filter((_, j) => j !== i))} className="absolute -right-1.5 -top-1.5 rounded-full bg-panel-3 p-0.5">
                  <X className="size-3" />
                </button>
              </div>
            ))}
          </div>
        )}
        <div className="flex items-end gap-2">
          {vision && (
            <label className="grid size-11 shrink-0 cursor-pointer place-items-center rounded-lg border border-line text-fg-2 hover:border-line-2 hover:text-fg" title="Attach image">
              <ImagePlus className="size-4" />
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={async (e) => {
                  const f = e.target.files?.[0];
                  if (f) setImages([...images, await toDataUrl(f)]);
                  e.target.value = "";
                }}
              />
            </label>
          )}
          <Textarea
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={`Message ${model.name}…`}
            className="max-h-48 min-h-11 resize-none"
          />
          {current ? (
            <Button variant="outline" className="h-11 shrink-0" onClick={() => cancel(current)}>
              <Square className="size-4" />
            </Button>
          ) : (
            <Button variant="primary" className="h-11 shrink-0" onClick={submit} disabled={!ready || (!input.trim() && !images.length)}>
              <Send className="size-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
