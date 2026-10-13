import { useQuery } from "@tanstack/react-query";
import { Moon, Sun } from "lucide-react";
import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router";

import { ErrorBoundary } from "./components/ErrorBoundary";

import { Brand, Shell } from "./components/Shell";
import { api } from "./lib/api";
import { LiveProvider, useLive } from "./lib/live";
import { useTheme } from "./lib/theme";
import Launch from "./pages/Launch";
import { AccessLogin, AdminLogin } from "./pages/Login";

const Models = lazy(() => import("./pages/Models"));
const Deployments = lazy(() => import("./pages/Deployments"));
const Studio = lazy(() => import("./pages/Studio"));
const Access = lazy(() => import("./pages/Access"));
const System = lazy(() => import("./pages/System"));

function Loading() {
  return (
    <div className="flex min-h-[40vh] items-center justify-center">
      <span className="inline-block h-5 w-2.5 bg-signal animate-blink" />
    </div>
  );
}

function VersionBanner({ server }: { server: string }) {
  if (server === __APP_VERSION__) return null;
  return (
    <div className="mb-5 rounded-lg border border-tight/40 bg-tight-soft px-4 py-3 text-[13.5px] text-tight">
      The page is KaggleUI {__APP_VERSION__} but the server is still {server}. In the notebook, re-run the KaggleUI cell. It
      restarts the server.
    </div>
  );
}

function AdminApp() {
  const { overview } = useLive();
  const loc = useLocation();
  if (!overview) return <Loading />;
  return (
    <Shell>
      <VersionBanner server={overview.version} />
      <ErrorBoundary resetKey={loc.pathname}>
      <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/" element={<Launch />} />
          <Route path="/models" element={<Models />} />
          <Route path="/deployments" element={<Deployments />} />
          <Route path="/studio" element={<Studio admin />} />
          <Route path="/studio/:id" element={<Studio admin />} />
          <Route path="/access" element={<Access />} />
          <Route path="/system" element={<System />} />
          <Route path="/login" element={<Navigate to="/studio" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
      </ErrorBoundary>
    </Shell>
  );
}

function GuestStudio() {
  const [theme, toggle] = useTheme();
  return (
    <div className="min-h-screen">
      <header className="flex items-center justify-between border-b border-line px-4 py-3 sm:px-6">
        <Brand small />
        <button onClick={toggle} className="rounded-md p-2 text-fg-2 hover:bg-panel-2" aria-label="Toggle theme">
          {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </button>
      </header>
      <main className="mx-auto w-full max-w-[1320px] px-4 py-7 sm:px-6 lg:px-8">
        <ErrorBoundary>
          <Suspense fallback={<Loading />}>
            <Studio admin={false} />
          </Suspense>
        </ErrorBoundary>
      </main>
    </div>
  );
}

export default function App() {
  const session = useQuery({ queryKey: ["session"], queryFn: api.session, retry: 1 });
  if (session.isLoading) return <Loading />;
  const s = session.data;
  const path = location.pathname;

  if (s?.admin) {
    return (
      <BrowserRouter>
        <LiveProvider>
          <AdminApp />
        </LiveProvider>
      </BrowserRouter>
    );
  }
  if (path.startsWith("/studio") || path.startsWith("/login")) {
    return (
      <BrowserRouter>
        <Suspense fallback={<Loading />}>
          <Routes>
            <Route path="/studio/*" element={s?.access ? <GuestStudio /> : <Navigate to={`/login?next=${encodeURIComponent(path)}`} replace />} />
            <Route path="/login" element={s?.access ? <Navigate to="/studio" replace /> : <AccessLogin />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    );
  }
  return <AdminLogin />;
}
