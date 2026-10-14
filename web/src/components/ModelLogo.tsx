import { Box } from "lucide-react";

import { cx } from "../lib/format";

const BRANDS: [RegExp, string, string][] = [
  [/qwen|qwq|qvq/i, "qwen-color", "Qwen"],
  [/deepseek/i, "deepseek-color", "DeepSeek"],
  [/mistral|mixtral|codestral|ministral|devstral|voxtral/i, "mistral-color", "Mistral"],
  [/gemma/i, "gemma-color", "Google Gemma"],
  [/llama|meta-llama/i, "meta-color", "Meta"],
  [/hunyuan|hy-world|hy_world|worldplay|tencent/i, "hunyuan-color", "Tencent Hunyuan"],
  [/flux|black.forest/i, "flux", "Black Forest Labs"],
  [/stable.diffusion|stability|sdxl|stablelm/i, "stability-color", "Stability AI"],
  [/wan[\d_. -]|wan-ai|cosyvoice|alibaba/i, "qwen-color", "Alibaba"],
  [/nvidia|nemotron|cosmos/i, "nvidia-color", "NVIDIA"],
  [/microsoft|phi[\d_.-]/i, "microsoft-color", "Microsoft"],
  [/minimax/i, "minimax-color", "MiniMax"],
  [/moonshot|kimi/i, "moonshot", "Moonshot AI"],
  [/openai|gpt-oss|whisper/i, "openai", "OpenAI"],
  [/bytedance|seedance|seedream|bagel/i, "bytedance-color", "ByteDance"],
  [/zhipu|zai-org|glm[\d_. -]/i, "zhipu-color", "Z.ai"],
  [/bark|suno/i, "suno", "Suno"],
  [/google/i, "google-color", "Google"],
];

export function ModelLogo({ id, name = "", source, className }: { id: string; name?: string; source?: string; className?: string }) {
  const brand = BRANDS.find(([pattern]) => pattern.test(`${id} ${name}`));
  const fallback = source === "ollama" ? [null, "ollama", "Ollama"] : source === "hf" ? [null, "huggingface-color", "Hugging Face"] : null;
  const match = brand ?? fallback;
  return <span className={cx("model-logo", className)}>
    {match ? <img width="32" height="32" src={`/brands/${match[1]}.svg`} alt={`${match[2]} logo`} loading="lazy" decoding="async" className={cx("size-full object-contain", !String(match[1]).endsWith("color") && "brand-mono")} /> : <Box className="size-5 text-fg-3" strokeWidth={1.5} />}
  </span>;
}
