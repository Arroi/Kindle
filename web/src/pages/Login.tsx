import { KeyRound, Lock } from "lucide-react";
import { useState, type FormEvent } from "react";

import { Brand } from "../components/Shell";
import { Button, Input } from "../components/ui";
import { api } from "../lib/api";

function Frame({ children, footer }: { children: React.ReactNode; footer?: React.ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-10">
      <div className="w-full max-w-[400px] stagger">
        <div className="mb-8 flex justify-center">
          <Brand />
        </div>
        <div className="panel p-6 sm:p-7">{children}</div>
        {footer && <div className="mt-5 text-center text-[13px] text-fg-3">{footer}</div>}
      </div>
    </div>
  );
}

export function AdminLogin() {
  const bad = new URLSearchParams(location.search).has("bad_token");
  const [token, setToken] = useState("");
  const [error, setError] = useState(bad ? "That link's token didn't match. Paste the admin token from the notebook output." : "");
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.login(token.trim());
      location.assign("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Frame
      footer={
        <>
          Here to use a model someone shared? <a className="text-fg-2 underline underline-offset-4 hover:text-fg" href="/login?next=/studio">Open the Studio</a>
        </>
      }
    >
      <div className="mb-1 flex items-center gap-2 text-signal">
        <KeyRound className="size-4" />
        <span className="readout text-signal">Control panel</span>
      </div>
      <h1 className="display mb-2 text-4xl">Admin token</h1>
      <p className="mb-5 text-[14px] text-fg-2">
        Paste the token printed in your Kaggle notebook, or open the link from that cell. It signs you in directly.
      </p>
      <form onSubmit={submit} className="space-y-3">
        <Input
          autoFocus
          value={token}
          onChange={(e) => setToken(e.target.value)}
          placeholder="Admin token"
          autoComplete="current-password"
          type="password"
          className="font-mono"
        />
        {error && <div className="text-[13px] text-worst">{error}</div>}
        <Button variant="primary" size="lg" className="w-full" busy={busy} disabled={!token.trim()}>
          Unlock panel
        </Button>
      </form>
    </Frame>
  );
}

export function AccessLogin() {
  const next = new URLSearchParams(location.search).get("next") || "/studio";
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.accessLogin(password.trim());
      location.assign(next.startsWith("/") ? next : "/studio");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Wrong password");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Frame footer={<a className="underline underline-offset-4 hover:text-fg" href="/">Host? Sign in to the control panel</a>}>
      <div className="mb-1 flex items-center gap-2 text-signal">
        <Lock className="size-4" />
        <span className="readout text-signal">Password locked</span>
      </div>
      <h1 className="display mb-2 text-4xl">Studio access</h1>
      <p className="mb-5 text-[14px] text-fg-2">The host locked this KaggleUI with a password. Ask them for it.</p>
      <form onSubmit={submit} className="space-y-3">
        <Input autoFocus value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" type="password" />
        {error && <div className="text-[13px] text-worst">{error}</div>}
        <Button variant="primary" size="lg" className="w-full" busy={busy} disabled={!password.trim()}>
          Enter
        </Button>
      </form>
    </Frame>
  );
}
