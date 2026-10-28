import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowDown, Brain, Check, ChevronDown, Copy, Download, ImagePlus, PanelLeft, Pencil, Plus, RotateCcw, Send,
  SlidersHorizontal, Square, Trash2, Upload, Wrench, X,
} from "lucide-react";
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { Button, Field, Select, Slider, Textarea, useToast } from "../../components/ui";
import { api } from "../../lib/api";
import { effortLabel } from "../../lib/capabilities";
import { cx, tokens } from "../../lib/format";
import { markdownBlocks } from "../../lib/markdown";
import type { ChatSummary, StudioModel } from "../../lib/types";

// ---------------------------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------------------------

interface ToolCall { id: string; type: "function"; function: { name: string; arguments: string } }

interface Msg {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  images?: string[];
  reasoning?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
  stats?: { tps: number | null; ttft: number; tokens?: number; prompt?: number };
  finish?: string;
  error?: string;
  pending?: boolean;
}

interface Conversation {
  id: string;
  title: string;
  model: string;
  system: string;
  params: { temperature: number; maxTokens: number; effort: string };
  tools: string;
  messages: Msg[];
}

const NEW_TITLE = "New chat";
const ACTIVE_KEY = "kui.chat.active";
const PRESETS_KEY = "kui.chat.presets";

const newId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 10);

function fresh(model: string, from?: Conversation): Conversation {
  return {
    id: newId(), title: NEW_TITLE, model, system: from?.system ?? "",
    params: from?.params ?? { temperature: 0.7, maxTokens: 2048, effort: "default" }, tools: from?.tools ?? "", messages: [],
  };
}

function normalize(data: any, model: string): Conversation {
  const base = fresh(model);
  return {
    ...base, ...data,
    params: { ...base.params, ...(data?.params ?? {}) },
    messages: (Array.isArray(data?.messages) ? data.messages : []).map((m: Msg) => ({ ...m, id: m.id || newId(), pending: false })),
  };
}

function titleFrom(messages: Msg[]): string {
  const first = messages.find((m) => m.role === "user" && m.content.trim());
  if (!first) return NEW_TITLE;
  const text = first.content.trim().replace(/\s+/g, " ");
  return text.length > 60 ? text.slice(0, 57) + "…" : text;
}

/** Rough token count for context budgeting; replaced by the runner's numbers once a reply reports usage. */
function estimate(m: Pick<Msg, "content" | "tool_calls" | "images">): number {
  const chars = m.content.length + (m.tool_calls ? JSON.stringify(m.tool_calls).length : 0);
  return Math.ceil(chars / 3.5) + 4 + (m.images?.length ?? 0) * 800;
}

/** Drop the oldest turns until the history fits the window. Never starts mid tool exchange. */
function fitContext(history: Msg[], budget: number): { sent: Msg[]; dropped: number } {
  let used = 0;
  let start = history.length;
  for (let i = history.length - 1; i >= 0; i--) {
    used += estimate(history[i]);
    if (used > budget && start < history.length) break;
    start = i;
  }
  while (start < history.length - 1 && history[start].role !== "user") start++;
  return { sent: history.slice(start), dropped: start };
}

function toApi(messages: Msg[]): any[] {
  const out: any[] = [];
  for (const m of messages) {
    if (m.pending) continue;
    if (m.role === "assistant") {
      if (!m.content && !m.tool_calls?.length) continue; // failed or empty replies aren't history
      out.push({ role: "assistant", content: m.content || null, ...(m.tool_calls?.length ? { tool_calls: m.tool_calls } : {}) });
    } else if (m.role === "tool") {
      out.push({ role: "tool", tool_call_id: m.tool_call_id, content: m.content, ...(m.name ? { name: m.name } : {}) });
    } else {
      const content = m.images?.length
        ? [{ type: "text", text: m.content }, ...m.images.map((url) => ({ type: "image_url", image_url: { url } }))]
        : m.content;
      const last = out[out.length - 1];
      // Two user turns in a row (the reply between them failed): many templates insist on alternation.
      if (last?.role === "user" && typeof last.content === "string" && typeof content === "string") last.content += "\n\n" + content;
      else out.push({ role: "user", content });
    }
  }
  return out;
}

function mergeCalls(existing: ToolCall[] | undefined, deltas: any[]): ToolCall[] {
  const calls = [...(existing ?? [])];
  for (const d of deltas) {
    const i = typeof d.index === "number" ? d.index : calls.length;
    const prev = calls[i] ?? { id: "", type: "function" as const, function: { name: "", arguments: "" } };
    calls[i] = {
      id: d.id || prev.id || `call_${newId()}`, type: "function",
      function: { name: prev.function.name || d.function?.name || "", arguments: prev.function.arguments + (d.function?.arguments ?? "") },
    };
  }
  return calls;
}

function parseTools(text: string): { tools?: unknown[]; error?: string } {
  if (!text.trim()) return {};
  try {
    const value = JSON.parse(text);
    const list = Array.isArray(value) ? value : [value];
    const tools = list.map((t) => (t?.type === "function" ? t : { type: "function", function: t }));
    if (tools.some((t) => !t.function?.name)) return { error: "Every tool needs a name." };
    return { tools };
  } catch (e) {
    return { error: e instanceof Error ? e.message : "Invalid JSON" };
  }
}

async function copyText(text: string): Promise<void> {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  document.execCommand("copy");
  area.remove();
}

function download(name: string, body: string, type: string) {
  const url = URL.createObjectURL(new Blob([body], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function asMarkdown(c: Conversation): string {
  const parts = [`# ${c.title}`, "", `Model: ${c.model}`, ""];
  if (c.system) parts.push("## System", "", c.system, "");
  for (const m of c.messages) {
    if (m.role === "tool") parts.push(`### Tool result (${m.name ?? m.tool_call_id})`, "", "```", m.content, "```", "");
    else {
      parts.push(m.role === "user" ? "## You" : "## Assistant", "");
      if (m.content) parts.push(m.content, "");
      for (const call of m.tool_calls ?? []) parts.push(`Tool call \`${call.function.name}\`:`, "", "```json", call.function.arguments, "```", "");
    }
  }
  return parts.join("\n");
}

function ago(ts: number): string {
  const s = Date.now() / 1000 - ts;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

function readPresets(): { name: string; text: string }[] {
  try { return JSON.parse(localStorage.getItem(PRESETS_KEY) ?? "[]"); } catch { return []; }
}

// ---------------------------------------------------------------------------------------------
// Connection: one WebSocket per Studio, multiplexed by request id; plain HTTP when it's down
// ---------------------------------------------------------------------------------------------

type Handler = (msg: any) => void;

function useStudioSocket() {
  const ws = useRef<WebSocket | null>(null);
  const handlers = useRef(new Map<string, Handler>());
  const [ready, setReady] = useState(false);
  const requests = useRef(new Map<string, AbortController>());

  useEffect(() => {
    let closed = false;
    let timer: number | undefined;
    // Deltas are batched per animation-ish frame so a fast model doesn't re-render per token.
    const deltas = new Map<string, any>();
    const flush = (id: string) => {
      const delta = deltas.get(id);
      if (delta) { handlers.current.get(id)?.(delta); deltas.delete(id); }
    };
    const frame = window.setInterval(() => { for (const id of deltas.keys()) flush(id); }, 50);
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const sock = new WebSocket(`${proto}://${location.host}/_kui/studio/ws`);
      ws.current = sock;
      sock.onopen = () => setReady(true);
      sock.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.op === "delta") {
          const pending = deltas.get(msg.id) ?? { op: "delta", id: msg.id, content: "", reasoning: "", tool_calls: [] };
          pending.content += msg.content ?? "";
          pending.reasoning += msg.reasoning ?? "";
          if (msg.tool_calls) pending.tool_calls.push(...msg.tool_calls);
          deltas.set(msg.id, pending);
        } else { flush(msg.id); handlers.current.get(msg.id)?.(msg); }
      };
      sock.onclose = (event) => {
        setReady(false);
        for (const [id, h] of handlers.current) {
          if (!requests.current.has(id)) { h({ op: "error", message: "Live connection lost. Send again to use HTTP mode." }); handlers.current.delete(id); }
        }
        if (!closed && event.code !== 4401) timer = window.setTimeout(connect, 4000);
      };
    };
    connect();
    const ping = window.setInterval(() => { if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ op: "ping" })); }, 20_000);
    return () => {
      closed = true;
      window.clearTimeout(timer);
      window.clearInterval(ping);
      window.clearInterval(frame);
      for (const controller of requests.current.values()) controller.abort();
      requests.current.clear();
      ws.current?.close();
    };
  }, []);

  const send = useCallback((payload: { model: string; messages: unknown[]; params: object }, onMsg: Handler) => {
    const id = newId();
    handlers.current.set(id, onMsg);
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ op: "chat", ...payload, id }));
    else {
      const controller = new AbortController();
      requests.current.set(id, controller);
      const started = performance.now();
      void (async () => {
        try {
          const response = await fetch("/v1/chat/completions", {
            method: "POST", credentials: "same-origin", signal: controller.signal,
            headers: { "content-type": "application/json", "x-kui-studio": "1" },
            body: JSON.stringify({ model: payload.model, messages: payload.messages, ...payload.params, stream: false }),
          });
          const data = await response.json();
          if (!response.ok || data.error) throw new Error(data.error?.message || data.detail || "Model request failed");
          const choice = data.choices?.[0];
          const message = choice?.message;
          onMsg({ op: "delta", content: message?.content ?? "", reasoning: message?.reasoning_content ?? message?.reasoning ?? "",
                  tool_calls: (message?.tool_calls ?? []).map((c: ToolCall, index: number) => ({ ...c, index })) });
          onMsg({ op: "done", usage: data.usage, finish_reason: choice?.finish_reason,
                  ttft: Number(((performance.now() - started) / 1000).toFixed(2)), tps: null });
        } catch (error) {
          onMsg(controller.signal.aborted ? { op: "done", finish_reason: "cancelled" } : { op: "error", message: error instanceof Error ? error.message : "Request failed" });
        } finally { requests.current.delete(id); }
      })();
    }
    return id;
  }, []);
  const cancel = useCallback((id: string) => {
    if (requests.current.has(id)) requests.current.get(id)?.abort();
    else if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ op: "cancel", id }));
  }, []);
  const done = useCallback((id: string) => handlers.current.delete(id), []);
  return { ready, send, cancel, done };
}

// ---------------------------------------------------------------------------------------------
// Pieces
// ---------------------------------------------------------------------------------------------

/** One markdown block. Memoised, so a streaming reply only rewrites its last block. */
const Block = memo(function Block({ html }: { html: string }) {
  return <div className="md-block" dangerouslySetInnerHTML={{ __html: html }} />;
});

function Markdown({ text }: { text: string }) {
  const blocks = useMemo(() => markdownBlocks(text), [text]);
  return <div className="prose-chat">{blocks.map((html, i) => <Block key={i} html={html} />)}</div>;
}

function Reasoning({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="chat-reasoning">
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}>
        <Brain className={cx("size-3.5", live && "animate-pulse text-signal")} />
        {live ? "Thinking…" : "Thought process"}
        <span className="text-fg-3">{tokens(Math.ceil(text.length / 4))} tokens</span>
        <ChevronDown className={cx("ml-auto size-3.5 transition-transform", open && "rotate-180")} />
      </button>
      {open && <div className="chat-reasoning-body scroll-thin">{text}</div>}
    </div>
  );
}

function prettyArgs(raw: string): string {
  try { return JSON.stringify(JSON.parse(raw || "{}"), null, 2); } catch { return raw; }
}

function ToolCallCard({ call, result, canAnswer, onAnswer }: {
  call: ToolCall; result?: Msg; canAnswer: boolean; onAnswer: (text: string) => void;
}) {
  const [text, setText] = useState("");
  return (
    <div className="chat-tool">
      <div className="chat-tool-head"><Wrench className="size-3.5" /> <span className="font-mono">{call.function.name || "tool"}</span></div>
      <pre className="scroll-thin">{prettyArgs(call.function.arguments)}</pre>
      {result ? (
        <div className="chat-tool-result"><span className="text-fg-3">Result</span><pre className="scroll-thin">{result.content}</pre></div>
      ) : canAnswer ? (
        <div className="chat-tool-result space-y-2">
          <Textarea rows={2} value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste or type what the tool returned…" />
          <div className="flex gap-2">
            <Button size="sm" variant="primary" onClick={() => onAnswer(text)} disabled={!text.trim()}>Send result</Button>
            <Button size="sm" variant="ghost" onClick={() => onAnswer("The user skipped this tool call.")}>Skip</Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function IconButton({ label, onClick, children, disabled }: { label: string; onClick: () => void; children: React.ReactNode; disabled?: boolean }) {
  return (
    <button type="button" className="chat-icon" aria-label={label} title={label} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

function CopyMessage({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <IconButton label={copied ? "Copied" : "Copy"} onClick={() => copyText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1400); })}>
      {copied ? <Check className="size-3.5 text-good" /> : <Copy className="size-3.5" />}
    </IconButton>
  );
}

// ---------------------------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------------------------

export default function ChatStudio({ model }: { model: StudioModel }) {
  const { ready, send, cancel, done } = useStudioSocket();
  const toast = useToast();
  const qc = useQueryClient();
  const chats = useQuery({ queryKey: ["chats"], queryFn: api.chats, staleTime: 30_000 });
  const [conv, setConv] = useState<Conversation>(() => fresh(model.name));
  const [current, setCurrent] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const [showList, setShowList] = useState(false);
  const [editing, setEditing] = useState<{ id: string; text: string } | null>(null);
  const [dropped, setDropped] = useState(0);
  const [presets, setPresets] = useState(readPresets);
  const [renaming, setRenaming] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const stick = useRef(true);
  const [showJump, setShowJump] = useState(false);

  // The reply cap can reach the context the model was launched with; prompt and history share that window.
  const ctxCap = Math.max(256, Math.round(Number(model.options?.ctx) || 8192));
  const tokenStep = ctxCap <= 32768 ? 64 : ctxCap <= 131072 ? 256 : 1024;
  const vision = model.capabilities.includes("vision");
  const thinking = model.reasoning?.supported || model.capabilities.some((cap) => ["reasoning", "thinking"].includes(cap));
  const effortOptions = [{ value: "default", label: "Auto" }, ...(model.reasoning?.supported === false ? [] : model.reasoning?.efforts ?? []).map((value) => ({ value, label: effortLabel(value) }))];
  const { temperature, maxTokens } = conv.params;
  const effort = effortOptions.some((o) => o.value === conv.params.effort) ? conv.params.effort : "default";
  const toolSpec = useMemo(() => parseTools(conv.tools), [conv.tools]);
  const messages = conv.messages;
  const setParams = (patch: Partial<Conversation["params"]>) => setConv((c) => ({ ...c, params: { ...c.params, ...patch } }));

  // Reopen the conversation that was open (also across model switches, which remount this view).
  useEffect(() => {
    const id = sessionStorage.getItem(ACTIVE_KEY);
    if (id) api.chat<any>(id).then((data) => setConv(normalize(data, model.name))).catch(() => sessionStorage.removeItem(ACTIVE_KEY));
  }, [model.name]);
  useEffect(() => { if (messages.length) sessionStorage.setItem(ACTIVE_KEY, conv.id); }, [conv.id, messages.length]);
  useEffect(() => { setParams({ maxTokens: Math.min(maxTokens, ctxCap) }); }, [ctxCap]); // eslint-disable-line react-hooks/exhaustive-deps

  // Save after each finished turn or edit.
  useEffect(() => {
    if (current || !messages.length) return;
    const timer = window.setTimeout(() => {
      api.saveChat(conv.id, conv)
        .then(({ updated }) => qc.setQueryData<ChatSummary[]>(["chats"], (list = []) => [
          { id: conv.id, title: conv.title, model: conv.model, updated, messages: messages.length },
          ...list.filter((c) => c.id !== conv.id),
        ]))
        .catch((e) => toast(e instanceof Error ? e.message : "Couldn't save this conversation", "error"));
    }, 400);
    return () => window.clearTimeout(timer);
  }, [conv, current]); // eslint-disable-line react-hooks/exhaustive-deps

  // Follow new output only while the reader is at the bottom. Scrolling up stays put.
  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < 72;
    stick.current = near;
    setShowJump(!near);
  };
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [messages]);
  const jump = () => {
    stick.current = true;
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  };

  // Copy buttons inside rendered code blocks.
  const onMessagesClick = (e: React.MouseEvent) => {
    const button = (e.target as HTMLElement).closest<HTMLButtonElement>("[data-copy-code]");
    if (!button) return;
    const code = button.closest(".code-block")?.querySelector("code")?.textContent ?? "";
    void copyText(code).then(() => {
      button.textContent = "Copied";
      setTimeout(() => { button.textContent = "Copy"; }, 1400);
    });
  };

  const systemTokens = estimate({ content: conv.system }) + (toolSpec.tools ? Math.ceil(JSON.stringify(toolSpec.tools).length / 3.5) : 0);
  const lastUsage = [...messages].reverse().find((m) => m.stats?.prompt);
  const used = lastUsage?.stats?.prompt
    ? lastUsage.stats.prompt + (lastUsage.stats.tokens ?? 0)
    : systemTokens + messages.reduce((sum, m) => sum + estimate(m), 0);
  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
  const unanswered = !current && lastAssistant?.tool_calls?.length
    ? lastAssistant.tool_calls.filter((call) => !messages.some((m) => m.role === "tool" && m.tool_call_id === call.id))
    : [];

  function run(history: Msg[]) {
    if (toolSpec.error) { toast(`Tools: ${toolSpec.error}`, "error"); return; }
    const assistant: Msg = { id: newId(), role: "assistant", content: "", pending: true };
    const title = conv.title === NEW_TITLE ? titleFrom(history) : conv.title;
    setConv((c) => ({ ...c, title, model: model.name, messages: [...history, assistant] }));
    stick.current = true;
    setShowJump(false);
    const fitted = fitContext(history, Math.max(ctxCap - maxTokens - systemTokens, 512));
    setDropped(fitted.dropped);
    const apiMessages = [...(conv.system.trim() ? [{ role: "system", content: conv.system.trim() }] : []), ...toApi(fitted.sent)];
    const params = {
      temperature, max_tokens: maxTokens,
      ...(effort !== "default" ? { reasoning_effort: effort } : {}),
      ...(toolSpec.tools?.length ? { tools: toolSpec.tools } : {}),
    };
    let reqId = "";
    reqId = send({ model: model.name, messages: apiMessages, params }, (ev) => {
      setConv((c) => ({
        ...c,
        messages: c.messages.map((m) => {
          if (m.id !== assistant.id) return m;
          if (ev.op === "delta") {
            return { ...m, content: m.content + (ev.content ?? ""), reasoning: ev.reasoning ? (m.reasoning ?? "") + ev.reasoning : m.reasoning,
                     tool_calls: ev.tool_calls?.length ? mergeCalls(m.tool_calls, ev.tool_calls) : m.tool_calls };
          }
          if (ev.op === "done") {
            return { ...m, pending: false, finish: ev.finish_reason,
                     stats: { tps: ev.tps ?? null, ttft: ev.ttft ?? 0, tokens: ev.usage?.completion_tokens, prompt: ev.usage?.prompt_tokens } };
          }
          if (ev.op === "error") return { ...m, pending: false, error: ev.message };
          return m;
        }),
      }));
      if (ev.op === "done" || ev.op === "error") {
        setCurrent(null);
        done(reqId);
      }
    });
    setCurrent(reqId);
  }

  function submit() {
    const text = input.trim();
    if ((!text && !images.length) || current || unanswered.length) return;
    run([...messages, { id: newId(), role: "user", content: text, images: images.length ? images : undefined }]);
    setInput("");
    setImages([]);
    if (composer.current) composer.current.style.height = "";
  }

  function answer(call: ToolCall, text: string) {
    const next = [...messages, { id: newId(), role: "tool" as const, tool_call_id: call.id, name: call.function.name, content: text }];
    const remaining = unanswered.filter((c) => c.id !== call.id);
    if (remaining.length) setConv((c) => ({ ...c, messages: next }));
    else run(next); // every call answered: the model continues
  }

  function regenerate() {
    const idx = messages.map((m) => m.role).lastIndexOf("user");
    if (idx >= 0) run(messages.slice(0, idx + 1));
  }

  function saveEdit() {
    if (!editing) return;
    const idx = messages.findIndex((m) => m.id === editing.id);
    if (idx < 0 || !editing.text.trim()) return;
    const edited = { ...messages[idx], content: editing.text.trim() };
    setEditing(null);
    run([...messages.slice(0, idx), edited]);
  }

  function open(id: string) {
    if (current) return;
    setShowList(false);
    api.chat<any>(id).then((data) => { setConv(normalize(data, model.name)); setDropped(0); stick.current = true; })
      .catch((e) => toast(e instanceof Error ? e.message : "Couldn't open that conversation", "error"));
  }

  function startNew() {
    if (current) return;
    setConv(fresh(model.name, conv));
    setDropped(0);
    setShowList(false);
    sessionStorage.removeItem(ACTIVE_KEY);
    composer.current?.focus();
  }

  async function remove(id: string) {
    if (!window.confirm("Delete this conversation?")) return;
    await api.deleteChat(id).catch(() => undefined);
    qc.setQueryData<ChatSummary[]>(["chats"], (list = []) => list.filter((c) => c.id !== id));
    if (id === conv.id) startNew();
  }

  async function importFile(file: File) {
    try {
      const data = JSON.parse(await file.text());
      if (!Array.isArray(data?.messages)) throw new Error("That file isn't an exported conversation.");
      setConv(normalize({ ...data, id: newId() }, model.name));
      setShowList(false);
    } catch (e) {
      toast(e instanceof Error ? e.message : "Couldn't import that file", "error");
    }
  }

  function savePreset() {
    const name = window.prompt("Name this system prompt");
    if (!name?.trim() || !conv.system.trim()) return;
    const next = [...presets.filter((p) => p.name !== name.trim()), { name: name.trim(), text: conv.system }];
    setPresets(next);
    try { localStorage.setItem(PRESETS_KEY, JSON.stringify(next)); } catch { /* private mode */ }
  }

  const usedPct = Math.min(used / ctxCap, 1);
  const list = chats.data ?? [];
  const resultFor = (call: ToolCall) => messages.find((m) => m.role === "tool" && m.tool_call_id === call.id);

  return (
    <div className="chat-shell">
      {/* Conversations */}
      <aside className={cx("chat-list", showList && "is-open")} aria-label="Conversations">
        <div className="chat-list-head">
          <Button size="sm" variant="primary" className="flex-1" onClick={startNew} disabled={!!current}><Plus className="size-3.5" /> New chat</Button>
          <label className="chat-icon cursor-pointer" title="Import a conversation (.json)" aria-label="Import a conversation">
            <Upload className="size-3.5" />
            <input type="file" accept="application/json,.json" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) void importFile(f); e.target.value = ""; }} />
          </label>
          <span className="lg:hidden"><IconButton label="Close" onClick={() => setShowList(false)}><X className="size-3.5" /></IconButton></span>
        </div>
        <div className="chat-list-items scroll-thin">
          {list.length === 0 && <p className="px-3 py-4 text-[12.5px] text-fg-3">Conversations you have here are saved on this session's disk.</p>}
          {list.map((c) => (
            <div key={c.id} className={cx("chat-list-item", c.id === conv.id && "is-active")}>
              <button type="button" onClick={() => open(c.id)} disabled={!!current} className="min-w-0 flex-1 text-left">
                <span className="block truncate text-[13px]">{c.title}</span>
                <span className="block truncate text-[11.5px] text-fg-3">{c.model ?? "—"} · {ago(c.updated)}</span>
              </button>
              <IconButton label="Delete conversation" onClick={() => void remove(c.id)}><Trash2 className="size-3.5" /></IconButton>
            </div>
          ))}
        </div>
      </aside>
      {showList && <div className="chat-list-backdrop lg:hidden" onClick={() => setShowList(false)} />}

      {/* Conversation */}
      <section className="chat-main">
        <header className="chat-head">
          <span className="lg:hidden"><IconButton label="Conversations" onClick={() => setShowList(true)}><PanelLeft className="size-4" /></IconButton></span>
          <div className="min-w-0 flex-1">
            {renaming ? (
              <input autoFocus className="chat-title-input" defaultValue={conv.title}
                onBlur={(e) => { setConv((c) => ({ ...c, title: e.target.value.trim() || c.title })); setRenaming(false); }}
                onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); if (e.key === "Escape") setRenaming(false); }} />
            ) : (
              <button type="button" className="chat-title" onClick={() => messages.length && setRenaming(true)} title={messages.length ? "Rename" : undefined}>{conv.title}</button>
            )}
            <div className="flex items-center gap-2 text-[12px] text-fg-3">
              <span className={cx("dot", ready ? "dot-live" : "dot-warn")} />
              <span>{ready ? "Live" : "HTTP mode"}</span>
              <span aria-hidden>·</span>
              <span className="chat-meter" title="Context used by this conversation">
                <span className="chat-meter-bar"><span style={{ width: `${usedPct * 100}%` }} className={cx(usedPct > 0.85 ? "bg-tight" : "bg-signal")} /></span>
                {lastUsage ? "" : "≈"}{tokens(used)} / {tokens(ctxCap)}
              </span>
            </div>
          </div>
          {thinking && (
            <Select aria-label="Reasoning effort" value={effort} onChange={(e) => setParams({ effort: e.target.value })} className="hidden h-8 w-auto text-[12.5px] sm:block" disabled={effortOptions.length < 2}>
              {effortOptions.map((o) => <option key={o.value} value={o.value}>Reasoning: {o.label}</option>)}
            </Select>
          )}
          <IconButton label="Export as Markdown" onClick={() => download(`${conv.title.replace(/[^\w-]+/g, "_")}.md`, asMarkdown(conv), "text/markdown")} disabled={!messages.length}><Download className="size-4" /></IconButton>
          <IconButton label="Export as JSON" onClick={() => download(`${conv.title.replace(/[^\w-]+/g, "_")}.json`, JSON.stringify(conv, null, 2), "application/json")} disabled={!messages.length}><span className="text-[10.5px] font-semibold">JSON</span></IconButton>
          <Button size="sm" variant={showSettings ? "outline" : "ghost"} onClick={() => setShowSettings(!showSettings)} aria-expanded={showSettings}>
            <SlidersHorizontal className="size-3.5" /> <span className="hidden sm:inline">Settings</span>
          </Button>
        </header>

        <div className="chat-body">
          <div className="chat-scroll-wrap">
            <div ref={scroller} onScroll={onScroll} onClick={onMessagesClick} className="chat-scroll scroll-thin">
              <div className="chat-thread">
                {messages.length === 0 && (
                  <div className="chat-empty">
                    <div className="text-[20px] font-semibold tracking-tight">{model.name}</div>
                    <p className="mt-1 text-[13.5px] text-fg-3">
                      {[vision && "Reads images", thinking && "Reasons", (model.capabilities.includes("tools") || toolSpec.tools) && "Calls tools"].filter(Boolean).join(" · ") || "Text chat"}
                      {" · "}{tokens(ctxCap)} context
                    </p>
                    <div className="mt-5 flex flex-wrap justify-center gap-2">
                      {["Explain a concept step by step", "Review this code for bugs", "Draft a plan for a project"].map((p) => (
                        <button key={p} type="button" className="chat-suggestion" onClick={() => { setInput(p); composer.current?.focus(); }}>{p}</button>
                      ))}
                    </div>
                  </div>
                )}
                {dropped > 0 && (
                  <div className="chat-notice">{dropped} earlier message{dropped === 1 ? "" : "s"} weren't sent so the conversation fits the {tokens(ctxCap)} context window.</div>
                )}
                {messages.map((m) => {
                  if (m.role === "tool") return null; // shown inside the call it answers
                  if (m.role === "user") {
                    const isEditing = editing?.id === m.id;
                    return (
                      <div key={m.id} className="chat-row is-user group">
                        <div className={cx("chat-bubble", isEditing && "w-full")}>
                          {m.images?.map((src, k) => <img key={k} src={src} alt="" className="mb-2 max-h-56 rounded-lg" />)}
                          {isEditing ? (
                            <div className="space-y-2">
                              <Textarea autoFocus rows={3} value={editing.text} onChange={(e) => setEditing({ id: m.id, text: e.target.value })}
                                onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) saveEdit(); if (e.key === "Escape") setEditing(null); }} />
                              <div className="flex justify-end gap-2">
                                <Button size="sm" variant="ghost" onClick={() => setEditing(null)}>Cancel</Button>
                                <Button size="sm" variant="primary" onClick={saveEdit} disabled={!editing.text.trim()}>Save and send</Button>
                              </div>
                            </div>
                          ) : <div className="whitespace-pre-wrap break-words">{m.content}</div>}
                        </div>
                        {!isEditing && (
                          <div className="chat-actions">
                            <CopyMessage text={m.content} />
                            <IconButton label="Edit and resend" onClick={() => setEditing({ id: m.id, text: m.content })} disabled={!!current}><Pencil className="size-3.5" /></IconButton>
                          </div>
                        )}
                      </div>
                    );
                  }
                  const isLast = m.id === lastAssistant?.id;
                  return (
                    <div key={m.id} className="chat-row is-assistant group">
                      {m.reasoning && <Reasoning text={m.reasoning} live={!!m.pending && !m.content} />}
                      {m.content ? <Markdown text={m.content} /> : m.pending && !m.reasoning && !m.tool_calls?.length ? <span className="chat-caret" aria-label="Waiting for the model" /> : null}
                      {m.tool_calls?.map((call) => (
                        <ToolCallCard key={call.id} call={call} result={resultFor(call)} canAnswer={isLast && !current}
                          onAnswer={(text) => answer(call, text)} />
                      ))}
                      {m.error && <div className="chat-error">{m.error}</div>}
                      {m.finish === "length" && !m.pending && <div className="chat-notice">Stopped at the {tokens(maxTokens)}-token reply limit. Raise Max tokens in Settings, or ask it to continue.</div>}
                      {!m.pending && (
                        <div className="chat-meta">
                          <CopyMessage text={m.content} />
                          {isLast && <IconButton label="Regenerate" onClick={regenerate} disabled={!!current}><RotateCcw className="size-3.5" /></IconButton>}
                          {m.stats && (
                            <span className="ml-1 font-mono text-[11px] text-fg-3">
                              {m.stats.tokens ? `${m.stats.tokens} tok` : ""}{m.stats.tps ? ` · ${m.stats.tps} tok/s` : ""} · first token {m.stats.ttft}s
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
            {showJump && (
              <button type="button" className="chat-jump" onClick={jump}><ArrowDown className="size-3.5" /> Jump to latest</button>
            )}
          </div>

          {showSettings && (
            <aside className="chat-settings scroll-thin" aria-label="Chat settings">
              <Field label="System prompt">
                <Textarea rows={5} value={conv.system} onChange={(e) => setConv((c) => ({ ...c, system: e.target.value }))} placeholder="You are a helpful assistant." />
              </Field>
              <div className="-mt-2 flex flex-wrap items-center gap-2">
                {presets.length > 0 && (
                  <Select aria-label="System prompt presets" value="" onChange={(e) => { const p = presets.find((x) => x.name === e.target.value); if (p) setConv((c) => ({ ...c, system: p.text })); }} className="h-8 w-auto flex-1 text-[12.5px]">
                    <option value="">Use a preset…</option>
                    {presets.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
                  </Select>
                )}
                <Button size="sm" variant="ghost" onClick={savePreset} disabled={!conv.system.trim()}>Save as preset</Button>
              </div>
              <Slider label="Temperature" value={temperature} min={0} max={2} step={0.05} onChange={(v) => setParams({ temperature: v })} format={(v) => v.toFixed(2)} />
              <div>
                <Slider label="Max tokens" value={maxTokens} min={64} max={ctxCap} step={tokenStep} onChange={(v) => setParams({ maxTokens: v })} format={tokens} />
                <p className="mt-1 text-[11.5px] text-fg-3">Reply cap. The conversation and the reply share the {tokens(ctxCap)} window; older turns are left out when it fills up.</p>
              </div>
              {thinking && (
                <Field label="Reasoning effort">
                  <Select value={effort} onChange={(e) => setParams({ effort: e.target.value })} disabled={effortOptions.length < 2}>
                    {effortOptions.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                  </Select>
                </Field>
              )}
              <Field label="Tools (JSON)" hint={toolSpec.error ? <span className="text-worst">{toolSpec.error}</span> : "OpenAI-style function definitions. When the model calls one, you answer it in the chat, which is handy for testing agent prompts."}>
                <Textarea rows={6} className="font-mono text-[12px]" value={conv.tools} onChange={(e) => setConv((c) => ({ ...c, tools: e.target.value }))}
                  placeholder={'[{"name": "get_weather", "description": "Current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]'} />
              </Field>
            </aside>
          )}
        </div>

        <footer className="chat-composer">
          {images.length > 0 && (
            <div className="mb-2 flex gap-2">
              {images.map((src, i) => (
                <div key={i} className="relative">
                  <img src={src} alt="" className="size-14 rounded-md object-cover" />
                  <button type="button" aria-label="Remove image" onClick={() => setImages(images.filter((_, j) => j !== i))} className="absolute -right-1.5 -top-1.5 rounded-full border border-line bg-panel p-0.5"><X className="size-3" /></button>
                </div>
              ))}
            </div>
          )}
          <div className="chat-input">
            {vision && (
              <label className="chat-icon cursor-pointer" title="Attach image" aria-label="Attach image">
                <ImagePlus className="size-4" />
                <input type="file" accept="image/*" multiple className="hidden" onChange={async (e) => {
                  const files = [...(e.target.files ?? [])];
                  e.target.value = "";
                  const urls = await Promise.all(files.map((f) => new Promise<string>((res, rej) => { const r = new FileReader(); r.onload = () => res(String(r.result)); r.onerror = rej; r.readAsDataURL(f); })));
                  setImages((prev) => [...prev, ...urls]);
                }} />
              </label>
            )}
            <textarea
              ref={composer}
              rows={1}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 220)}px`;
              }}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); } }}
              placeholder={unanswered.length ? "Answer the tool call above to continue" : `Message ${model.name}`}
              disabled={unanswered.length > 0}
              aria-label="Message"
            />
            {current ? (
              <Button variant="outline" aria-label="Stop generating" className="h-9 w-9 shrink-0 px-0" onClick={() => cancel(current)}><Square className="size-3.5" /></Button>
            ) : (
              <Button variant="primary" aria-label="Send message" className="h-9 w-9 shrink-0 px-0" onClick={submit} disabled={(!input.trim() && !images.length) || unanswered.length > 0}><Send className="size-3.5" /></Button>
            )}
          </div>
          <p className="mt-1.5 text-[11.5px] text-fg-3">Enter to send · Shift+Enter for a new line</p>
        </footer>
      </section>
    </div>
  );
}
