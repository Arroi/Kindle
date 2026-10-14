export const EFFORT_LABELS: Record<string, string> = {
  default: "Auto", none: "Off", on: "On", minimal: "Minimal", low: "Low", medium: "Medium",
  high: "High", xhigh: "xHigh", max: "Max",
};

export function effortLabel(value: string): string {
  return EFFORT_LABELS[value] ?? value.replace(/[_-]/g, " ").replace(/^./, c => c.toUpperCase());
}

export interface ReasoningSupport {
  supported: boolean | null;
  efforts: string[];
  source: "metadata" | "template" | "family" | "unknown";
  default?: string | boolean | null;
}
