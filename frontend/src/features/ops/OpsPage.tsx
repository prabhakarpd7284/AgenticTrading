/**
 * Developer Ops Console — full catalog view.
 *
 * Left pane: searchable list of every visible `manage.py` command, grouped
 * by app, with dangerous-command badges. Right pane: the same OpRunPanel
 * that every other page embeds — pick a command, type args, hit Run, watch
 * stdout/stderr stream in over WebSocket.
 *
 * For *contextual* runs (Monthly's "refresh signal outcomes", Pyramid's
 * "Run via CLI", etc.) use <OpButton> directly — this page is the
 * everything-at-once catalog, not the daily workflow.
 */
import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Search, Terminal } from "lucide-react";

import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardHeader } from "@/components/ui/Card";

import { OpRunPanel } from "./OpRunPanel";

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
          <Badge tone={data.trading_mode === "live" ? "danger" : "neutral"}>
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
          <Card className="overflow-hidden p-4">
            {/* `key` resets the panel state when the selected command changes. */}
            <OpRunPanel
              key={selectedCmd.name}
              command={selectedCmd.name}
              description={selectedCmd.help || undefined}
              dangerous={selectedCmd.dangerous}
            />
          </Card>
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
