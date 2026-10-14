import { Box } from "lucide-react";

import { cx } from "../lib/format";
import { inferBrand } from "../lib/catalogFilters";

export function ModelLogo({ id, name = "", source, className }: { id: string; name?: string; source?: string; className?: string }) {
  const brand = inferBrand(id, name, source);
  const fallback = source === "ollama" ? { logo: "ollama", label: "Ollama" } : source === "hf" ? { logo: "huggingface-color", label: "Hugging Face" } : null;
  const match = brand?.logo ? brand : fallback;
  const logo = brand?.id === "google" && /gemma/i.test(`${id} ${name}`) ? "gemma-color" : match?.logo;
  return <span className={cx("model-logo", className)}>
    {match && logo ? <img width="32" height="32" src={`/brands/${logo}.svg`} alt={`${match.label} logo`} loading="lazy" decoding="async" className={cx("size-full object-contain", !logo.endsWith("color") && "brand-mono")} /> : <Box className="size-5 text-fg-3" strokeWidth={1.5} />}
  </span>;
}
