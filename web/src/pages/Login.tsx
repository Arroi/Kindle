import { Eye, EyeOff, KeyRound, Lock } from "lucide-react";
import { useState, type FormEvent } from "react";

import { Brand } from "../components/Shell";
import { Button, Input } from "../components/ui";
import { api } from "../lib/api";
import { CommunityLinks } from "../components/CommunityLinks";

function Frame({ children, footer }: { children: React.ReactNode; footer?: React.ReactNode }) {
  return (
    <div className="login-page">
      <header className="login-header"><Brand /><span className="text-[12px] text-fg-3">{__APP_VERSION__}</span></header>
      <main className="login-layout page-enter">
        <section className="login-intro"><span className="eyebrow">A notebook. A GPU. Your workspace.</span><h2 className="display mt-5 text-[22px] sm:text-[26px]">Open models,<br />within reach.</h2><p className="mt-5 max-w-sm text-[15px] leading-relaxed text-fg-2">Chat, images, video, speech, and 3D worlds. Run a model on your notebook and use it from any browser.</p><div className="mt-8 flex gap-3" aria-label="Supported model families">{["qwen-color", "deepseek-color", "mistral-color", "meta-color", "hunyuan-color"].map(brand => <span key={brand} className="model-logo size-11"><img src={`/brands/${brand}.svg`} width="28" height="28" alt={`${brand.replace("-color", "")} logo`} /></span>)}</div><p className="mt-5 text-[12px] text-fg-3">Open source · Powered by your notebook session</p></section>
        <section className="login-form"><div>{children}</div>{footer && <div className="mt-6 border-t border-line pt-5 text-[13px] text-fg-3">{footer}</div>}</section>
      </main>
      <footer className="login-footer"><CommunityLinks /></footer>
    </div>
  );
}

export function AdminLogin() {
  const bad = new URLSearchParams(location.search).has("bad_token");
  const [token, setToken] = useState("");
  const [error, setError] = useState(bad ? "That link's token didn't match. Paste the admin token from the notebook output." : "");
  const [busy, setBusy] = useState(false);
  const [shown, setShown] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.login(token.trim());
      if (!(await api.session()).admin) throw new Error("Your browser did not save the sign-in cookie. Open this link directly in a regular browser tab and try again.");
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
      <h1 className="display mb-2 text-2xl">Admin token</h1>
      <p className="mb-5 text-[14px] text-fg-2">
        Paste the token printed in your Kaggle notebook, or open the link from that cell. It signs you in directly.
      </p>
      <form onSubmit={submit} className="space-y-3">
        <label className="block text-[12px] font-medium text-fg-2" htmlFor="admin-token">Host token</label>
        <div className="relative"><Input
          id="admin-token"
          autoFocus
          value={token}
          onChange={(e) => setToken(e.target.value)}
          placeholder="Admin token"
          autoComplete="current-password"
          type={shown ? "text" : "password"}
          className="pr-11 font-mono"
        /><button type="button" onClick={() => setShown(!shown)} className="absolute right-3 top-3 text-fg-3" aria-label={shown ? "Hide token" : "Show token"}>{shown ? <EyeOff className="size-4" /> : <Eye className="size-4" />}</button></div>
        {error && <div role="alert" className="text-[13px] text-worst">{error}</div>}
        <Button variant="primary" size="lg" className="w-full" busy={busy} disabled={!token.trim()}>
          Sign in as host
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
  const [shown, setShown] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.accessLogin(password);
      if (!(await api.session()).access) throw new Error("Your browser did not save the sign-in cookie. Open the link directly in a regular browser tab and try again.");
      location.assign(next.startsWith("/") && !next.startsWith("//") && !next.includes("\\") ? next : "/studio");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Wrong password");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Frame footer={<a className="underline underline-offset-4 hover:text-fg" href="/host">Host? Sign in to the control panel</a>}>
      <div className="mb-1 flex items-center gap-2 text-signal">
        <Lock className="size-4" />
        <span className="readout text-signal">Password locked</span>
      </div>
      <h1 className="display mb-2 text-2xl">Studio access</h1>
      <p className="mb-5 text-[14px] text-fg-2">The host locked this Kindle with a password. Ask them for it.</p>
      <form onSubmit={submit} className="space-y-3">
        <label className="block text-[12px] font-medium text-fg-2" htmlFor="studio-password">Studio password</label>
        <div className="relative"><Input id="studio-password" autoFocus autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Enter the host’s password" type={shown ? "text" : "password"} className="pr-11" /><button type="button" onClick={() => setShown(!shown)} className="absolute right-3 top-3 text-fg-3" aria-label={shown ? "Hide password" : "Show password"}>{shown ? <EyeOff className="size-4" /> : <Eye className="size-4" />}</button></div>
        {error && <div role="alert" className="text-[13px] text-worst">{error}</div>}
        <Button variant="primary" size="lg" className="w-full" busy={busy} disabled={!password.trim()}>
          Open Studio
        </Button>
      </form>
    </Frame>
  );
}
