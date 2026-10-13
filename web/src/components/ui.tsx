import { Check, Copy, Eye, EyeOff, Loader, X } from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";
import { createPortal } from "react-dom";

import { cx, VERDICT_LABEL } from "../lib/format";
import type { Verdict } from "../lib/types";

type Variant = "primary" | "ghost" | "outline" | "danger";

export function Button({
  variant = "outline",
  size = "md",
  busy,
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: "sm" | "md" | "lg"; busy?: boolean }) {
  return (
    <button
      {...rest}
      disabled={rest.disabled || busy}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-[background,border-color,color,transform] duration-150",
        "disabled:cursor-not-allowed disabled:opacity-45 active:translate-y-px",
        size === "sm" && "h-8 px-3 text-[13px]",
        size === "md" && "h-10 px-4 text-sm",
        size === "lg" && "h-12 px-6 text-[15px]",
        variant === "primary" && "bg-signal text-signal-ink hover:brightness-110 font-semibold",
        variant === "outline" && "border border-line-2 bg-panel-2 text-fg hover:border-fg-3",
        variant === "ghost" && "text-fg-2 hover:bg-panel-2 hover:text-fg",
        variant === "danger" && "border border-worst/40 text-worst hover:bg-worst-soft",
        className,
      )}
    >
      {busy ? <Loader className="size-4 animate-spin" /> : null}
      {children}
    </button>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label?: ReactNode;
  hint?: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className={cx("flex items-start gap-3 select-none", disabled ? "opacity-50" : "cursor-pointer")}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cx(
          "relative mt-0.5 h-6 w-11 shrink-0 rounded-full border transition-colors",
          checked ? "border-signal bg-signal" : "border-line-2 bg-panel-3",
        )}
      >
        <span
          className={cx(
            "absolute top-0.5 size-[18px] rounded-full transition-all",
            checked ? "left-[22px] bg-signal-ink" : "left-0.5 bg-fg-3",
          )}
        />
      </button>
      {(label || hint) && (
        <span className="min-w-0">
          {label && <span className="block font-medium">{label}</span>}
          {hint && <span className="block text-[13px] text-fg-2">{hint}</span>}
        </span>
      )}
    </label>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  size = "md",
}: {
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (v: T) => void;
  size?: "sm" | "md";
}) {
  return (
    <div className="inline-flex max-w-full overflow-x-auto rounded-lg border border-line bg-bg-2 p-0.5 scroll-thin">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          aria-pressed={value === o.value}
          className={cx(
            "whitespace-nowrap rounded-md font-medium transition-colors",
            size === "sm" ? "px-2.5 py-1 text-[12.5px]" : "px-3.5 py-1.5 text-[13px]",
            value === o.value ? "bg-panel-3 text-fg shadow-sm" : "text-fg-2 hover:text-fg",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function useCopy(): [boolean, (text: string) => void] {
  const [copied, setCopied] = useState(false);
  const copy = useCallback((text: string) => {
    const done = () => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    };
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text) && done());
    } else if (fallbackCopy(text)) {
      done();
    }
  }, []);
  return [copied, copy];
}

function fallbackCopy(text: string): boolean {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  const ok = document.execCommand("copy");
  ta.remove();
  return ok;
}

export function CopyButton({ text, className, label }: { text: string; className?: string; label?: string }) {
  const [copied, copy] = useCopy();
  return (
    <button
      type="button"
      onClick={() => copy(text)}
      title="Copy"
      className={cx(
        "inline-flex shrink-0 items-center gap-1.5 rounded-md px-2 py-1 text-[12px] text-fg-2 transition-colors hover:bg-panel-3 hover:text-fg",
        className,
      )}
    >
      {copied ? <Check className="size-3.5 text-signal" /> : <Copy className="size-3.5" />}
      {label && <span>{copied ? "Copied" : label}</span>}
    </button>
  );
}

export function CopyField({ value, secret, label }: { value: string; secret?: boolean; label?: string }) {
  const [shown, setShown] = useState(!secret);
  return (
    <div>
      {label && <div className="readout mb-1.5">{label}</div>}
      <div className="flex items-center gap-1 rounded-lg border border-line bg-bg-2 py-1 pl-3 pr-1">
        <code className="min-w-0 flex-1 truncate font-mono text-[12.5px]">
          {shown ? value : "•".repeat(Math.min(value.length, 28))}
        </code>
        {secret && (
          <button
            type="button"
            onClick={() => setShown(!shown)}
            className="rounded-md p-1.5 text-fg-2 hover:bg-panel-3 hover:text-fg"
            title={shown ? "Hide" : "Show"}
          >
            {shown ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
          </button>
        )}
        <CopyButton text={value} />
      </div>
    </div>
  );
}

const SHORT_VERDICT: Record<Verdict, string> = { best: "Best fit", good: "Fits", tight: "Tight", worst: "Won't fit" };

export function VerdictTag({ verdict, size = "md" }: { verdict: Verdict; size?: "sm" | "md" }) {
  const styles: Record<Verdict, string> = {
    best: "bg-signal text-signal-ink border-signal",
    good: "border-good/45 text-good bg-good-soft",
    tight: "border-tight/45 text-tight bg-tight-soft",
    worst: "border-worst/45 text-worst bg-worst-soft",
  };
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded border font-medium",
        size === "sm" ? "px-1.5 py-px text-[11px]" : "px-2 py-0.5 text-[12px]",
        styles[verdict],
      )}
    >
      {verdict === "best" && <span className="size-1.5 rounded-full bg-signal-ink" />}
      {size === "sm" ? SHORT_VERDICT[verdict] : VERDICT_LABEL[verdict]}
    </span>
  );
}

export function Chip({ children, tone = "plain", className }: { children: ReactNode; tone?: "plain" | "info" | "warn" | "signal"; className?: string }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-1.5 py-px text-[11.5px]",
        tone === "plain" && "border-line bg-panel-2 text-fg-2",
        tone === "info" && "border-info/35 text-info",
        tone === "warn" && "border-tight/40 text-tight",
        tone === "signal" && "border-signal/40 text-signal",
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...rest}
      className={cx(
        "h-10 w-full rounded-lg border border-line bg-bg-2 px-3 text-sm text-fg placeholder:text-fg-3",
        "transition-colors focus:border-signal/60 focus:outline-none",
        className,
      )}
    />
  );
}

export function Textarea({ className, ...rest }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      {...rest}
      className={cx(
        "w-full rounded-lg border border-line bg-bg-2 px-3 py-2.5 text-sm text-fg placeholder:text-fg-3",
        "resize-y transition-colors focus:border-signal/60 focus:outline-none",
        className,
      )}
    />
  );
}

export function Select({ className, children, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...rest}
      className={cx(
        "h-10 w-full rounded-lg border border-line bg-bg-2 px-3 text-sm text-fg focus:border-signal/60 focus:outline-none",
        className,
      )}
    >
      {children}
    </select>
  );
}

export function Field({ label, hint, children }: { label: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="block">
      <span className="readout mb-1.5 block">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[12.5px] text-fg-3">{hint}</span>}
    </label>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 flex items-baseline justify-between">
        <span className="readout">{label}</span>
        <span className="font-mono text-[12px] text-fg">{format ? format(value) : value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[var(--signal)]"
      />
    </label>
  );
}

export function SectionTitle({ kicker, title, children }: { kicker?: string; title: ReactNode; children?: ReactNode }) {
  return (
    <div className="mb-7 flex flex-wrap items-end justify-between gap-4">
      <div>
        {kicker && <div className="readout mb-2">{kicker}</div>}
        <h1 className="display text-[28px] sm:text-[34px]">{title}</h1>
      </div>
      {children}
    </div>
  );
}

export function Empty({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="panel flex flex-col items-center px-6 py-12 text-center">
      {icon && <div className="mb-3 text-fg-3">{icon}</div>}
      <div className="display text-2xl">{title}</div>
      {children && <div className="mt-2 max-w-md text-fg-2">{children}</div>}
    </div>
  );
}

export function Drawer({
  open,
  onClose,
  children,
  title,
  width = 560,
}: {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  title?: ReactNode;
  width?: number;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);
  if (!open) return null;
  return createPortal(
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/55 backdrop-blur-[2px] animate-[rise_.2s_ease_both]" onClick={onClose} />
      <aside
        className="relative flex h-full w-full flex-col border-l border-line bg-panel shadow-2xl animate-[rise_.25s_cubic-bezier(.2,.7,.2,1)_both]"
        style={{ maxWidth: width }}
      >
        <div className="flex items-center justify-between gap-3 border-b border-line px-5 py-3.5">
          <div className="min-w-0 truncate font-medium">{title}</div>
          <button onClick={onClose} className="rounded-md p-1.5 text-fg-2 hover:bg-panel-3 hover:text-fg" title="Close">
            <X className="size-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto scroll-thin">{children}</div>
      </aside>
    </div>,
    document.body,
  );
}

/* ---------------------------------------------------------------------------------------- */
/* Toasts                                                                                     */
/* ---------------------------------------------------------------------------------------- */

interface Toast {
  id: number;
  text: string;
  tone: "ok" | "error";
}
const ToastCtx = createContext<(text: string, tone?: Toast["tone"]) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const push = useCallback((text: string, tone: Toast["tone"] = "ok") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, text, tone }]);
    window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), tone === "error" ? 7000 : 3200);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed bottom-4 left-1/2 z-[60] flex w-[min(440px,calc(100vw-32px))] -translate-x-1/2 flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={cx(
              "pointer-events-auto rounded-xl border px-4 py-3 text-sm shadow-xl animate-[rise_.25s_ease_both]",
              t.tone === "ok" ? "border-line-2 bg-panel-3 text-fg" : "border-worst/50 bg-panel-3 text-worst",
            )}
          >
            {t.text}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

export const useToast = () => useContext(ToastCtx);
