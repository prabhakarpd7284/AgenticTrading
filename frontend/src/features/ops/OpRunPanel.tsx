/**
 * OpRunPanel — the run/log half of the ops console, extracted so any page
 * can embed it (Monthly's "refresh signal outcomes", Swing Scanner's
 * "Re-scan now", Pyramid's "Run via CLI", Setup's "Plan a trade for X",
 * etc.).
 *
 * Self-contained: opens the WS on Run, streams log lines in, fires
 * `onSuccess` when the subprocess exits 0 so the parent can refetch its
 * data. Stop kills the subprocess. Argument input is a single text box
 * pre-filled with `defaultArgs`; the user can edit before firing.
 */
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, ChevronDown, ChevronRight, Play, Square } from "lucide-react";

import { api } from "@/lib/api";
import { connect } from "@/lib/ws";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";

export type RunStatus = "idle" | "running" | "done" | "error";

export interface OpRunPanelProps {
  /** Django management command name, e.g. "run_pyramid". */
  command: string;
  /** Optional one-line description shown above the args input. */
  description?: string;
  /** Initial args, either a shell-style string ("--strike 24200 --dry-run")
   *  or a list. Editable in the input field before Run. */
  defaultArgs?: string | string[];
  /** Flag the command visually as destructive (matches the badge in OpsPage). */
  dangerous?: boolean;
  /** Fired on subprocess exit 0 so the caller can refetch its data. */
  onSuccess?: () => void;
  /** Fired on any non-zero exit / spawn error. */
  onError?: (detail: string) => void;
  /** Hide the --help collapsible (defaults to visible). */
  hideHelp?: boolean;
}

interface LogLine {
  kind: "log" | "info" | "done" | "error";
  text: string;
}

export function OpRunPanel(props: OpRunPanelProps) {
  const {
    command, description, defaultArgs = "", dangerous,
    onSuccess, onError, hideHelp,
  } = props;

  const [args, setArgs] = React.useState(() =>
    Array.isArray(defaultArgs) ? defaultArgs.join(" ") : defaultArgs,
  );
  React.useEffect(() => {
    // When the caller passes new defaults (page state changed), reset.
    setArgs(Array.isArray(defaultArgs) ? defaultArgs.join(" ") : defaultArgs);
  }, [defaultArgs]);

  const [status, setStatus] = React.useState<RunStatus>("idle");
  const [lines, setLines] = React.useState<LogLine[]>([]);
  const [helpOpen, setHelpOpen] = React.useState(false);
  const wsRef = React.useRef<ReturnType<typeof connect> | null>(null);
  const logEnd = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    // Cleanup on unmount — close the WS, which triggers server-side SIGTERM.
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  React.useEffect(() => {
    logEnd.current?.scrollIntoView({ behavior: "auto", block: "end" });
  }, [lines]);

  const helpQ = useQuery<{ help: string }>({
    enabled: !hideHelp && helpOpen,
    queryKey: ["ops-command-help", command],
    queryFn: () => api.get(`/ops/commands/${command}/help/`).then((r) => r.data),
  });

  const start = () => {
    if (status === "running") return;
    setLines([]);
    setStatus("running");
    const ws = connect("/ws/ops/", (msg) => {
      const m = msg as {
        type: string; line?: string; detail?: string;
        exit_code?: number; argv?: string[]; pid?: number;
      };
      if (m.type === "log" && typeof m.line === "string") {
        setLines((prev) => [...prev, { kind: "log", text: m.line! }]);
      } else if (m.type === "started") {
        setLines((prev) => [
          ...prev,
          { kind: "info", text: `▶ pid=${m.pid}  argv=${(m.argv ?? []).join(" ")}` },
        ]);
      } else if (m.type === "done") {
        const ok = m.exit_code === 0;
        setStatus(ok ? "done" : "error");
        setLines((prev) => [...prev, { kind: "done", text: `── exit ${m.exit_code} ──` }]);
        wsRef.current?.close();
        wsRef.current = null;
        if (ok) onSuccess?.();
        else onError?.(`exit ${m.exit_code}`);
      } else if (m.type === "error") {
        setStatus("error");
        setLines((prev) => [...prev, { kind: "error", text: `✗ ${m.detail}` }]);
        onError?.(m.detail ?? "unknown error");
      } else if (m.type === "stopped") {
        setStatus("done");
        setLines((prev) => [...prev, { kind: "info", text: "── stopped ──" }]);
      }
    }, {
      onOpen: () => {
        wsRef.current?.send({ type: "start", command, args });
      },
      onError: () => {
        setStatus("error");
        setLines((prev) => [...prev, { kind: "error", text: "✗ websocket error" }]);
        onError?.("websocket error");
      },
    });
    wsRef.current = ws;
  };

  const stop = () => wsRef.current?.send({ type: "stop" });

  return (
    <div className="flex h-full flex-col gap-3">
      {/* ── header row: name · badges · run/stop ── */}
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate font-mono text-sm font-medium">{command}</span>
            {dangerous && (
              <Badge tone="warning">
                <AlertTriangle className="mr-1 size-3" /> dangerous
              </Badge>
            )}
          </div>
          {description && (
            <div className="mt-0.5 text-xs text-fg-muted">{description}</div>
          )}
        </div>
        {status === "running" ? (
          <Button onClick={stop} variant="destructive" size="sm">
            <Square className="mr-1 size-4" /> Stop
          </Button>
        ) : (
          <Button onClick={start} size="sm">
            <Play className="mr-1 size-4" /> {status === "done" || status === "error" ? "Run again" : "Run"}
          </Button>
        )}
      </div>

      {/* ── args input ── */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-medium uppercase tracking-wide text-fg-muted">
          Arguments
        </label>
        <input
          value={args}
          onChange={(e) => setArgs(e.target.value)}
          disabled={status === "running"}
          spellCheck={false}
          placeholder="(no args)"
          className="rounded-md border border-border bg-bg-subtle px-3 py-2 font-mono text-sm outline-none focus:border-fg-muted disabled:opacity-50"
          onKeyDown={(e) => {
            if (e.key === "Enter" && status !== "running") start();
          }}
        />
        {!hideHelp && (
          <>
            <button
              type="button"
              onClick={() => setHelpOpen((v) => !v)}
              className="flex items-center gap-1 self-start text-xs text-fg-muted hover:text-fg"
            >
              {helpOpen ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
              --help
            </button>
            {helpOpen && helpQ.data && (
              <pre className="max-h-48 overflow-auto rounded-md bg-bg-subtle p-3 font-mono text-xs leading-relaxed text-fg-muted">
                {helpQ.data.help}
              </pre>
            )}
          </>
        )}
      </div>

      {/* ── log panel ── */}
      <div
        className={cn(
          "flex-1 overflow-y-auto rounded-md border border-border bg-bg-subtle font-mono text-xs leading-relaxed",
          lines.length === 0 && "flex items-center justify-center",
        )}
      >
        {lines.length === 0 ? (
          <div className="text-fg-subtle">Run output will appear here.</div>
        ) : (
          <div className="p-3">
            {lines.map((l, i) => (
              <div
                key={i}
                className={cn(
                  "whitespace-pre-wrap break-all",
                  l.kind === "error" && "text-red-400",
                  l.kind === "info" && "text-fg-muted",
                  l.kind === "done" && (status === "error" ? "text-red-400" : "text-emerald-500"),
                )}
              >
                {l.text}
              </div>
            ))}
            <div ref={logEnd} />
          </div>
        )}
      </div>
    </div>
  );
}
