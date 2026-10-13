import { RotateCcw, TriangleAlert } from "lucide-react";
import { Component, type ErrorInfo, type ReactNode } from "react";

/** Shows what broke instead of a blank screen. Resets when ``resetKey`` changes (e.g. the route). */
export class ErrorBoundary extends Component<{ children: ReactNode; resetKey?: string }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[kaggleui] render error", error, info.componentStack);
  }

  componentDidUpdate(prev: { resetKey?: string }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="panel mx-auto mt-6 max-w-xl p-6">
        <div className="mb-2 flex items-center gap-2 text-worst">
          <TriangleAlert className="size-4" />
          <span className="readout text-worst">This page hit an error</span>
        </div>
        <p className="mb-3 text-[14px] text-fg-2">
          If you just updated KaggleUI, the old server may still be running. In the notebook, re-run the KaggleUI cell. It
          restarts the server.
        </p>
        <pre className="mb-4 overflow-x-auto rounded-lg border border-line bg-bg p-3 font-mono text-[11.5px] text-fg-3">{String(this.state.error.message || this.state.error)}</pre>
        <button
          onClick={() => location.reload()}
          className="inline-flex h-10 items-center gap-2 rounded-lg border border-line-2 bg-panel-2 px-4 text-sm hover:border-fg-3"
        >
          <RotateCcw className="size-4" /> Reload
        </button>
      </div>
    );
  }
}
