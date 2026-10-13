import { createContext, useContext, useEffect, useMemo, useReducer, useRef, type ReactNode } from "react";

import type { Deployment, Overview, Settings, System, Tunnel } from "./types";

interface LiveState {
  overview: Overview | null;
  logs: Record<string, string[]>;
  connected: boolean;
}

type Action =
  | { type: "overview"; data: Overview }
  | { type: "metrics"; data: System }
  | { type: "deployment"; data: Deployment }
  | { type: "deployment_removed"; data: { id: string } }
  | { type: "log"; data: { id: string; line: string } }
  | { type: "logs"; id: string; lines: string[] }
  | { type: "tunnel"; data: Tunnel }
  | { type: "settings"; data: Settings }
  | { type: "credentials"; data: Overview["credentials"] }
  | { type: "secrets"; data: Record<string, boolean> }
  | { type: "connected"; value: boolean };

const MAX_LOG = 2000;

function reducer(state: LiveState, a: Action): LiveState {
  const ov = state.overview;
  switch (a.type) {
    case "overview":
      return { ...state, overview: a.data };
    case "connected":
      return { ...state, connected: a.value };
    case "logs":
      return { ...state, logs: { ...state.logs, [a.id]: a.lines.slice(-MAX_LOG) } };
    case "log": {
      const prev = state.logs[a.data.id] ?? [];
      const next = prev.length >= MAX_LOG ? [...prev.slice(-MAX_LOG + 1), a.data.line] : [...prev, a.data.line];
      return { ...state, logs: { ...state.logs, [a.data.id]: next } };
    }
  }
  if (!ov) return state;
  switch (a.type) {
    case "metrics":
      return { ...state, overview: { ...ov, system: a.data } };
    case "deployment": {
      const exists = ov.deployments.some((d) => d.id === a.data.id);
      const deployments = exists
        ? ov.deployments.map((d) => (d.id === a.data.id ? a.data : d))
        : [...ov.deployments, a.data];
      return { ...state, overview: { ...ov, deployments } };
    }
    case "deployment_removed":
      return { ...state, overview: { ...ov, deployments: ov.deployments.filter((d) => d.id !== a.data.id) } };
    case "tunnel": {
      const others = ov.tunnels.filter((t) => t.name !== a.data.name);
      const tunnels = a.data.status === "stopped" ? others : [...others, a.data];
      const main = tunnels.find((t) => t.name === "main");
      return {
        ...state,
        overview: {
          ...ov,
          tunnels,
          main_url: main?.url ?? ov.main_url,
          streaming_ok: main ? main.supports_sse : ov.streaming_ok,
        },
      };
    }
    case "settings":
      return { ...state, overview: { ...ov, settings: a.data } };
    case "credentials":
      return { ...state, overview: { ...ov, credentials: a.data } };
    case "secrets":
      return { ...state, overview: { ...ov, secrets: a.data } };
  }
  return state;
}

interface LiveCtx extends LiveState {
  dispatch: (a: Action) => void;
}

const Ctx = createContext<LiveCtx | null>(null);

export function LiveProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, { overview: null, logs: {}, connected: false });
  const retry = useRef(0);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let closed = false;

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/_kui/ws`);
      ws.onopen = () => {
        retry.current = 0;
        dispatch({ type: "connected", value: true });
      };
      ws.onmessage = (ev) => {
        try {
          dispatch(JSON.parse(ev.data));
        } catch {
          /* ignore */
        }
      };
      ws.onclose = (ev) => {
        dispatch({ type: "connected", value: false });
        if (closed) return;
        if (ev.code === 4401) {
          location.reload();
          return;
        }
        const delay = Math.min(1000 * 2 ** retry.current, 15000);
        retry.current++;
        timer = window.setTimeout(connect, delay);
      };
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      ws?.close();
    };
  }, []);

  const value = useMemo(() => ({ ...state, dispatch }), [state]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useLive(): LiveCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useLive outside LiveProvider");
  return ctx;
}

export function useOverview(): Overview | null {
  return useLive().overview;
}
