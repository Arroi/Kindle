export type Modality = "text" | "image" | "video" | "tts" | "audio" | "stt" | "world" | "mesh";
export type Verdict = "best" | "good" | "tight" | "worst";

export interface Gpu {
  index: number;
  name: string;
  vram_total: number;
  vram_used: number;
  vram_free: number;
  util: number | null;
  temp: number | null;
  compute_capability: string | null;
  sm: number | null;
  kind?: "gpu" | "tpu";
}

export interface Disk {
  key: string;
  label: string;
  path: string;
  total: number;
  used: number;
  free: number;
  note: string;
}

export interface System {
  platform: string;
  accelerator: string;
  accelerator_label: string;
  gpus: Gpu[];
  tpus: Gpu[];
  cpu_count: number;
  ram_total: number;
  ram_available: number;
  disks: Disk[];
  run_type: string | null;
  session_started: number;
  session_limit: number;
  session_elapsed: number;
  session_remaining: number;
  internet: boolean | null;
  fake: string | null;
  notes: string[];
  vram_total_all: number;
  vram_free_all: number;
  hbm_total_all: number;
  hbm_free_all: number;
}

export interface Settings {
  access: { password_enabled: boolean };
  api: { enabled: boolean; openai: boolean; anthropic: boolean; ollama: boolean; require_key: boolean };
  exposure: { provider: TunnelProvider };
}

export type TunnelProvider = "cloudflare" | "cloudflare_named" | "ngrok" | "none";

export interface Tunnel {
  name: string;
  provider: TunnelProvider;
  label: string;
  port: number;
  url: string | null;
  status: string;
  error: string | null;
  restarts: number;
  supports_sse: boolean;
}

export interface ModelRef {
  source: "hf" | "ollama" | "attached" | "curated";
  id: string;
  variant: string | null;
  engine: string;
}

export interface DeploymentSpec {
  modality: Modality;
  model: ModelRef;
  name?: string | null;
  interface?: "studio" | "gradio" | "openwebui" | "comfyui" | "none";
  options?: Record<string, unknown>;
  gpus?: number[] | null;
}

export interface Deployment {
  id: string;
  spec: DeploymentSpec;
  name: string;
  modality: Modality;
  status: "queued" | "installing" | "downloading" | "loading" | "ready" | "error" | "stopping" | "stopped";
  progress: number | null;
  message: string;
  error: string | null;
  gpus: number[];
  vram_estimate: number;
  created_at: number;
  ready_at: number | null;
  info: Record<string, any>;
  interface: { kind?: string; status?: string; url?: string | null; error?: string | null; share_url?: string; message?: string };
}

export interface Overview {
  version: string;
  mock: boolean;
  fake_hw: string | null;
  system: System;
  settings: Settings;
  credentials: { admin_token: string; password: string; api_key: string };
  secrets: Record<string, boolean>;
  tunnels: Tunnel[];
  main_url: string;
  streaming_ok: boolean;
  deployments: Deployment[];
  started_at: number;
}

export interface Fit {
  verdict: Verdict;
  vram_needed: number | null;
  capacity: number | null;
  placement: "single" | "split" | "offload" | "cpu" | "none";
  reasons: string[];
  warnings: string[];
  disk_needed: number | null;
  score: number;
  suggest: Record<string, unknown>;
}

export interface Variant {
  id: string;
  ref: string;
  label: string;
  size: number | null;
  params: number | null;
  context?: string | null;
  context_length?: number | null;
  input?: string | null;
  fit?: { verdict: Verdict; reasons: string[]; warnings?: string[] };
}

export interface CatalogBrand {
  id: string;
  label: string;
  logo: string | null;
  count?: number;
}

export interface CatalogItem {
  brand?: CatalogBrand;
  custom_code?: boolean;
  reasoning?: import("./capabilities").ReasoningSupport;
  runtime_repos?: string[];
  extra_size?: number;
  key: string;
  source: "hf" | "ollama" | "attached";
  id: string;
  name: string;
  modality: Modality;
  engine: string | null;
  engine_label?: string;
  description: string;
  params: number | null;
  size: number | null;
  variant: string | null;
  variants: Variant[];
  downloads: number;
  likes: number;
  updated: number | null;
  gated: boolean;
  capabilities: string[];
  experimental: boolean;
  curated: boolean;
  selftest?: boolean;
  options: Record<string, unknown>;
  url: string | null;
  fit: Fit;
  spec: DeploymentSpec | null;
  library?: string | null;
  files?: number;
  context_length?: number | null;
}

export interface SearchResult {
  filters_active?: boolean;
  facets?: { brands: CatalogBrand[]; engines: { id: string; label: string; count: number }[]; capabilities: Record<string, number> };
  modality: Modality;
  items: CatalogItem[];
  sources: Record<string, string>;
  hardware: { accelerator: string; gpus: number; tpus: number; vram_free: number; hbm_free: number; ram_available: number; disk_free: number | null; fake: string | null };
  counts?: { total: number; fits: number; shown: number; launchable?: number };
  has_more?: boolean;
}

export interface CatalogMeta {
  modalities: Modality[];
  engines: Record<Modality, string[]>;
  engine_labels: Record<string, string>;
  recipes: { name: string; title: string; repo: string; controls: string | null; unverified: boolean }[];
  tunnel_providers: Record<string, string>;
}

export interface Session {
  compatibility?: { blocked: boolean; message: string };
  admin: boolean;
  access: boolean;
  password_enabled: boolean;
  version: string;
  mock: boolean;
  fake_hw: string | null;
}

export interface UpdateStatus {
  current: string;
  latest: string | null;
  available: boolean;
  blocked: boolean;
  minimum_supported: string | null;
  message: string;
  release_url: string;
  checked_at: number | null;
  error: string | null;
  progress: { status: string; message: string } | null;
}

export interface StudioModel {
  adapter?: string;
  reasoning?: import("./capabilities").ReasoningSupport;
  model_id?: string;
  source?: string;
  text_input?: boolean;
  result_kind?: "video" | "scene";
  needs_vlm?: boolean;
  prompt_conditioning?: boolean;
  id: string;
  name: string;
  modality: Modality;
  status: Deployment["status"];
  message: string;
  progress: number | null;
  capabilities: string[];
  controls: string | null;
  start_frame?: { size: string; prompt_prefix?: string } | null;
  options: Record<string, any>;
  interface: Deployment["interface"];
  engine: string;
  runtime?: string;
}

export interface SecretItem {
  name: string;
  set: boolean;
  source: "kaggle" | "env" | "kaggleui" | null;
  editable: boolean;
}

export interface SecretDetails {
  items: SecretItem[];
  kaggle: { state: string; error: string | null };
  hf?: { ok: boolean; user: string | null; error: string | null };
  added?: string[];
}
export interface ModelImport {
  id: string;
  name: string;
  url: string;
  kind: string;
  created: number;
  status: string;
  progress: number | null;
  bytes: number;
  message: string;
  models: { path: string; name: string; modality: Modality; engine: string | null }[];
}
