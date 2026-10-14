import { Brain, Eye, Wrench, Layers } from "lucide-react";

const TAGS = [
  { id: "reasoning", label: "Reasoning", icon: Brain },
  { id: "vision", label: "Vision", icon: Eye },
  { id: "tools", label: "Tools", icon: Wrench },
  { id: "embedding", label: "Embeddings", icon: Layers },
];

export function CapabilityTags({ capabilities }: { capabilities: string[] }) {
  const caps = new Set(capabilities.map(value => value === "thinking" ? "reasoning" : value));
  return <span className="inline-flex flex-wrap items-center gap-1.5" aria-label="Model capabilities">
    {TAGS.filter(tag => caps.has(tag.id)).map(({ id, label, icon: Icon }) => <span key={id} className="capability-tag"><Icon className="size-3" strokeWidth={1.6} />{label}</span>)}
  </span>;
}
