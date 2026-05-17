/**
 * Developer Ops Console — pick a `manage.py` command, type args, click Run,
 * watch stdout/stderr stream in over WebSocket.
 *
 * Backend pair: apps.system.api (REST: list + help) + apps.system.consumers
 * (WebSocket: stream). Owner-role gated server-side.
 */
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, ChevronDown, ChevronRight, Play, Search, Square, Terminal } from "lucide-react";

import { api } from "@/lib/api";
import { connect } from "@/lib/ws";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";

interface CommandInfo {
  name: string;
  app: string;
  help: string;
  dangerous: boolean;
}

interface CommandList {
  commands: CommandInfo[];
  trading_mode: string;
}

type RunStatus = "idle" | "running" | "done" | "error";

interface LogLine {
  kind: "log" | "started" | "done" | "error" | "stopped" | "info";
  text: string;
}

export function OpsPage() {
  const { data, isLoading } = useQuery<CommandList>({
    queryKey: ["ops-commands"],
    queryFn: () => api.get<CommandList>("/ops/commands/").then((r) => r.data),
  });

  const [filter, setFilter] = React.useState("");
  const [selected, setSelected] = React.useState<string | null>(null);

  const filtered = React.useMemo(() => {
    const f = filter.trim().toLowerCase();
    const all = data?.commands ?? [];
    if (!f) return all;
    return all.filter(
      (c) => c.name.includes(f) || c.help.toLowerCase().includes(f) || c.app.includes(f),
    );
  }, [filter, data]);

  // Group by app so the long list is navigable.
  const grouped = React.useMemo(() => {
    const m = new Map<string, CommandInfo[]>();
    for (const c of filtered) {
      const arr = m.get(c.app) ?? [];
      arr.push(c);
      m.set(c.app, arr);
    }
    return [...m.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [filtered]);

  const selectedCmd = data?.commands.find((c) => c.name === selected) ?? null;

  return (
    <div className="flex h-full flex-col gap-3 p-4">
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Terminal className="size-5 text-fg-muted" />
          <h1 className="text-lg font-semibold">Ops Console</h1>
          <span className="text-sm text-fg-muted">
            Stream any <code className="rounded bg-bg-subtle px-1 py-0.5 text-xs">manage.py</code> command
          </span>
        </div>
        {data && (
          <Badge variant={data.trading_mode === "live" ? "destructive" : "outline"}>
            TRADING_MODE = {data.trading_mode}
          </Badge>
        )}
      </header>

      <div className="grid h-[calc(100vh-160px)] grid-cols-[320px_1fr] gap-4">
        {/* ── command list ── */}
        <Card className="flex flex-col overflow-hidden">
          <CardHeader className="border-b border-border pb-3">
            <div className="flex items-center gap-2">
              <Search className="size-4 text-fg-muted" />
              <input
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Filter commands…"
                className="flex-1 bg-transparent text-sm outline-none placeholder:text-fg-subtle"
              />
            </div>
          </CardHeader>
          <CardContent className="flex-1 overflow-y-auto p-0">
            {isLoading && <div className="p-4 text-sm text-fg-muted">Loading…</div>}
            {!isLoading && grouped.length === 0 && (
              <div className="p-4 text-sm text-fg-muted">No commands match.</div>
            )}
            {grouped.map(([app, cmds]) => (
              <div key={app}>
                <div className="sticky top-0 z-10 border-b border-border bg-surface px-3 py-1 text-xs font-medium uppercase tracking-wide text-fg-subtle">
                  {app.replace(/^apps\./, "")}
                </div>
                {cmds.map((c) => (
                  <button
                    key={c.name}
                    type="button"
                    onClick={() => setSelected(c.name)}
                    className={cn(
                      "flex w-full items-center justify-between gap-2 border-b border-border px-3 py-2 text-left text-sm hover:bg-bg-subtle",
                      selected === c.name && "bg-bg-subtle",
                    )}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="truncate font-mono text-xs">{c.name}</span>
                        {c.dangerous && (
                          <AlertTriangle className="size-3 shrink-0 text-amber-500" />
                        )}
                      </div>
                      {c.help && (
                        <div className="truncate text-xs text-fg-muted">{c.help}</div>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            ))}
          </CardContent>
        </Card>

        {/* ── run pane ── */}
        {selectedCmd ? (
          <RunPanel command={selectedCmd} />
        ) : (
          <Card className="flex items-center justify-center">
            <CardContent className="text-center text-sm text-fg-muted">
              Pick a command from the list to run it.
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

// ─── run panel ──────────────────────────────────────────────────────────

interface RunPanelProps {
  command: CommandInfo;
}

function RunPanel({ command }: RunPanelProps) {
  const [args, setArgs] = React.useState("");
  const [helpOpen, setHelpOpen] = React.useState(false);
  const [status, setStatus] = React.useState<RunStatus>("idle");
  const [lines, setLines] = React.useState<LogLine[]>([]);
  const wsRef = React.useRef<ReturnType<typeof connect> | null>(null);
  const logEnd = React.useRef<HTMLDivElement | null>(null);

  // Reset state when the selected command changes.
  React.useEffect(() => {
    setArgs("");
    setLines([]);
    setStatus("idle");
    setHelpOpen(false);
    wsRef.current?.close();
    wsRef.current = null;
  }, [command.name]);

  // Auto-scroll the log panel on each new line.
  React.useEffect(() => {
    logEnd.current?.scrollIntoView({ behavior: "auto", block: "end" });
  }, [lines]);

  const { data: helpData } = useQuery<{ help: string }>({
    queryKey: ["ops-command-help", command.name],
    queryFn: () => api.get(`/ops/commands/${command.name}/help/`).then((r) => r.data),
  });

  const start = () => {
    if (status === "running") return;
    setLines([]);
    setStatus("running");
    const ws = connect("/ws/ops/", (msg) => {
      const m = msg as { type: string; line?: string; detail?: string; exit_code?: number; argv?: string[]; pid?: number };
      if (m.type === "log" && typeof m.line === "string") {
        setLines((prev) => [...prev, { kind: "log", text: m.line! }]);
      } else if (m.type === "started") {
        setLines((prev) => [
          ...prev,
          { kind: "info", text: `▶ pid=${m.pid}  argv=${(m.argv ?? []).join(" ")}` },
        ]);
      } else if (m.type === "done") {
        setStatus(m.exit_code === 0 ? "done" : "error");
        setLines((prev) => [
          ...prev,
          { kind: "done", text: `── exit ${m.exit_code} ──` },
        ]);
        wsRef.current?.close();
        wsRef.current = null;
      } else if (m.type === "error") {
        setStatus("error");
        setLines((prev) => [...prev, { kind: "error", text: `✗ ${m.detail}` }]);
      } else if (m.type === "stopped") {
        setStatus("done");
        setLines((prev) => [...prev, { kind: "info", text: "── stopped ──" }]);
      }
    }, {
      onOpen: () => {
        wsRef.current?.send({ type: "start", command: command.name, args });
      },
      onError: () => {
        setStatus("error");
        setLines((prev) => [...prev, { kind: "error", text: "✗ websocket error" }]);
      },
      onClose: () => {
        if (status === "running") setStatus("done");
      },
    });
    wsRef.current = ws;
  };

  const stop = () => {
    wsRef.current?.send({ type: "stop" });
  };

  return (
    <Card className="flex flex-col overflow-hidden">
      <CardHeader className="border-b border-border pb-3">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0 flex-1">
            <CardTitle className="flex items-center gap-2 font-mono text-base">
              {command.name}
              {command.dangerous && (
                <Badge variant="outline" className="border-amber-500 text-amber-500">
                  <AlertTriangle className="mr-1 size-3" /> dangerous
                </Badge>
              )}
            </CardTitle>
            {command.help && (
              <div className="mt-1 text-xs text-fg-muted">{command.help}</div>
            )}
          </div>
          {status === "running" ? (
            <Button onClick={stop} variant="destructive" size="sm">
              <Square className="mr-1 size-4" /> Stop
            </Button>
          ) : (
            <Button onClick={start} size="sm">
              <Play className="mr-1 size-4" /> Run
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-3 border-b border-border p-3">
        <label className="text-xs font-medium uppercase tracking-wide text-fg-muted">
          Arguments
        </label>
        <input
          value={args}
          onChange={(e) => setArgs(e.target.value)}
          placeholder={`e.g. --strike 24200 --type CE --underlying NIFTY --dry-run`}
          disabled={status === "running"}
          spellCheck={false}
          className="rounded-md border border-border bg-bg-subtle px-3 py-2 font-mono text-sm outline-none focus:border-fg-muted disabled:opacity-50"
          onKeyDown={(e) => {
            if (e.key === "Enter" && status !== "running") start();
          }}
        />
        <button
          type="button"
          onClick={() => setHelpOpen((v) => !v)}
          className="flex items-center gap-1 text-xs text-fg-muted hover:text-fg"
        >
          {helpOpen ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
          --help
        </button>
        {helpOpen && helpData && (
          <pre className="max-h-60 overflow-auto rounded-md bg-bg-subtle p-3 font-mono text-xs leading-relaxed text-fg-muted">
            {helpData.help}
          </pre>
        )}
      </CardContent>

      {/* ── log panel ── */}
      <div
        className={cn(
          "flex-1 overflow-y-auto bg-bg-subtle font-mono text-xs leading-relaxed",
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
    </Card>
  );
}
