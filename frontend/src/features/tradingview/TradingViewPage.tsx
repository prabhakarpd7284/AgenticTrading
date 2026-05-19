/**
 * TradingView Manager — dedicated page for managing the TradingView
 * integration: webhook links, watchlists, and a faceted view of incoming
 * signals.
 *
 *   /tradingview
 *
 *   ┌─ KPI strip ─────────────────────────────────────────────────────────┐
 *   │ Links · Watchlists · Signals (7d) · Last alert                       │
 *   └──────────────────────────────────────────────────────────────────────┘
 *   ┌─ Webhook links (reuses TradingViewSection from broker page) ────────┐
 *   ...
 *   ┌─ Watchlists ─────────────────────────────────────────────────────────┐
 *   ...
 *   ┌─ Signals · group by [symbol|strategy|source|day] ───────────────────┐
 *   ...
 */
import * as React from "react";
import { toast } from "sonner";
import {
  Plus, RefreshCw, Sparkles, Tag, Trash2, X, Zap,
} from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import {
  Dialog, DialogContent, DialogDescription, DialogTitle,
} from "@/components/ui/Dialog";
import { EmptyState } from "@/components/ui/EmptyState";
import { FreshnessIndicator } from "@/components/ui/FreshnessIndicator";
import { Input } from "@/components/ui/Input";
import { Skeleton } from "@/components/ui/Skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/Tabs";

import {
  type GroupedSignalRow, type SignalGroupBy,
  type TradingViewWatchlist, type WatchlistKind,
  WATCHLIST_KIND_META,
  useAddSymbolsToWatchlist, useCreateTradingViewWatchlist,
  useDeleteTradingViewWatchlist, useGroupedSignals,
  useRefreshTradingViewWatchlist,
  useRemoveSymbolsFromWatchlist, useTradingViewLinks,
  useTradingViewWatchlists, useUpdateTradingViewWatchlist,
} from "@/lib/v2";
import { cn, fmtRel } from "@/lib/utils";

import { TradingViewSection } from "@/features/broker/TradingViewSection";


export function TradingViewPage() {
  const linksQ = useTradingViewLinks();
  const watchlistsQ = useTradingViewWatchlists();
  const groupedQ = useGroupedSignals({ by: "symbol", days: 7 });

  const totalSignals = (groupedQ.data?.rows || []).reduce((s, r) => s + r.count, 0);
  const lastAlert = (linksQ.data || [])
    .map((l) => l.last_received_at)
    .filter(Boolean)
    .sort()
    .at(-1);

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      <header>
        <p className="text-caption uppercase tracking-wider text-fg-subtle">Integration</p>
        <h1 className="text-h1 text-fg">TradingView Manager</h1>
        <p className="text-body-sm text-fg-muted mt-1 max-w-2xl">
          Webhook URLs, named watchlists, and grouped incoming signals — all
          in one place. Configure auto-fire per link to turn alerts into
          AgentRuns (RiskGuard still gates execution).
        </p>
      </header>

      {/* ── KPI strip ───────────────────────────────────────────────── */}
      <dl className="grid grid-cols-2 md:grid-cols-4 gap-2 rounded-md border border-border bg-surface-2/40 p-2">
        <KpiCell
          label="Webhook links"
          value={String((linksQ.data || []).length)}
        />
        <KpiCell
          label="Watchlists"
          value={String((watchlistsQ.data || []).length)}
        />
        <KpiCell
          label="Signals · 7d"
          value={String(totalSignals)}
        />
        <KpiCell
          label="Last alert"
          value={lastAlert ? `${fmtRel(lastAlert)} ago` : "—"}
        />
      </dl>

      {/* ── Webhook links (reused section) ──────────────────────────── */}
      <TradingViewSection />

      {/* ── Watchlists ──────────────────────────────────────────────── */}
      <WatchlistsCard />

      {/* ── Grouped signals ─────────────────────────────────────────── */}
      <GroupedSignalsCard />
    </div>
  );
}


/* =================================================================== */
/* Watchlists                                                           */
/* =================================================================== */

function WatchlistsCard() {
  const { data: watchlists = [], isLoading } = useTradingViewWatchlists();
  const [newOpen, setNewOpen] = React.useState(false);

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-3">
        <div>
          <CardTitle className="flex items-center gap-2">
            Watchlists
            <Badge tone="neutral">{watchlists.length}</Badge>
          </CardTitle>
          <CardDescription>
            Named symbol lists. Use as a soft filter on the signals view below,
            or as a reference when configuring auto-fire allowlists.
          </CardDescription>
        </div>
        <Button size="sm" onClick={() => setNewOpen(true)} leading={<Plus className="h-3.5 w-3.5" />}>
          New watchlist
        </Button>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-24 w-full" />
        ) : watchlists.length === 0 ? (
          <EmptyState
            icon={<Tag />}
            title="No watchlists yet"
            description="Create one to group symbols you want to track from TradingView alerts."
          />
        ) : (
          <ul className="space-y-3" role="list">
            {watchlists.map((w) => <WatchlistRow key={w.id} watchlist={w} />)}
          </ul>
        )}
      </CardContent>
      <NewWatchlistDialog open={newOpen} onClose={() => setNewOpen(false)} />
    </Card>
  );
}

function WatchlistRow({ watchlist }: { watchlist: TradingViewWatchlist }) {
  const update = useUpdateTradingViewWatchlist();
  const remove = useDeleteTradingViewWatchlist();
  const add = useAddSymbolsToWatchlist();
  const rm = useRemoveSymbolsFromWatchlist();
  const refresh = useRefreshTradingViewWatchlist();

  const meta = WATCHLIST_KIND_META[watchlist.kind];
  const [editingName, setEditingName] = React.useState(false);
  const [draftName, setDraftName] = React.useState(watchlist.name);
  const [newSymbol, setNewSymbol] = React.useState("");

  React.useEffect(() => setDraftName(watchlist.name), [watchlist.name]);

  const onAddSymbol = async (e: React.FormEvent) => {
    e.preventDefault();
    const sym = newSymbol.trim();
    if (!sym) return;
    await add.mutateAsync({ id: watchlist.id, symbols: [sym] });
    setNewSymbol("");
  };

  const onRenameCommit = async () => {
    if (draftName.trim() && draftName.trim() !== watchlist.name) {
      await update.mutateAsync({ id: watchlist.id, name: draftName.trim() });
    }
    setEditingName(false);
  };

  const onRemoveOne = (sym: string) =>
    rm.mutateAsync({ id: watchlist.id, symbols: [sym] });

  const onDelete = async () => {
    if (!window.confirm(`Delete watchlist "${watchlist.name}"?`)) return;
    await remove.mutateAsync(watchlist.id);
    toast.success("Watchlist deleted");
  };

  const onRefresh = async () => {
    await refresh.mutateAsync(watchlist.id);
    toast.success("Watchlist refreshed");
  };

  return (
    <li className="rounded-md border border-border bg-surface-2 p-4 space-y-3">
      <div className="flex items-center gap-3 flex-wrap">
        {editingName ? (
          <input
            value={draftName}
            onChange={(e) => setDraftName(e.target.value)}
            onBlur={onRenameCommit}
            onKeyDown={(e) => e.key === "Enter" && onRenameCommit()}
            autoFocus
            className="text-body-sm font-semibold text-fg bg-surface border border-accent/60 rounded-xs px-2 py-0.5"
          />
        ) : (
          <button
            type="button"
            onClick={() => setEditingName(true)}
            className="text-body-sm font-semibold text-fg hover:text-accent"
          >
            {watchlist.name}
          </button>
        )}
        <Badge tone={watchlist.is_auto ? "brand" : "neutral"} title={meta.blurb}>
          {watchlist.is_auto && <Sparkles className="h-3 w-3 mr-1" aria-hidden />}
          {meta.label}
        </Badge>
        <Badge tone="neutral">{watchlist.symbol_count} symbol{watchlist.symbol_count === 1 ? "" : "s"}</Badge>
        {watchlist.is_auto && watchlist.symbols_refreshed_at && (
          <span className="text-caption text-fg-subtle" title={watchlist.symbols_refreshed_at}>
            Refreshed {fmtRel(watchlist.symbols_refreshed_at)} ago
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          {watchlist.is_auto && (
            <Button
              size="sm" variant="secondary"
              onClick={onRefresh}
              loading={refresh.isPending}
              leading={<RefreshCw className="h-3.5 w-3.5" />}
            >
              Refresh
            </Button>
          )}
          <Button
            size="sm" variant="secondary"
            onClick={onDelete}
            loading={remove.isPending}
            leading={<Trash2 className="h-3.5 w-3.5" />}
          >
            Delete
          </Button>
        </div>
      </div>

      {/* Auto kinds: config summary line + non-editable chip readout */}
      {watchlist.is_auto && (
        <div className="text-caption text-fg-subtle font-mono">
          config {JSON.stringify(watchlist.config)}
        </div>
      )}

      {/* Symbol chips — manual kinds let you edit, auto kinds are read-only */}
      <div className="flex flex-wrap gap-1.5">
        {watchlist.symbols.map((sym) =>
          watchlist.is_auto ? (
            <span
              key={sym}
              className="inline-flex items-center rounded-xs bg-surface border border-border px-2 py-0.5 text-caption font-mono text-fg"
            >
              {sym}
            </span>
          ) : (
            <SymbolChip key={sym} symbol={sym} onRemove={() => onRemoveOne(sym)} />
          ),
        )}
        {!watchlist.is_auto && (
          <form onSubmit={onAddSymbol} className="inline-flex items-center gap-1">
            <input
              type="text"
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
              placeholder="+ symbol"
              className={cn(
                "h-7 px-2 rounded-xs text-caption font-mono",
                "bg-surface border border-border focus:border-accent focus:outline-none",
                "w-24 placeholder:text-fg-subtle",
              )}
            />
            {newSymbol && (
              <Button type="submit" size="sm" loading={add.isPending}>Add</Button>
            )}
          </form>
        )}
      </div>
    </li>
  );
}

function SymbolChip({ symbol, onRemove }: { symbol: string; onRemove: () => void }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-xs bg-surface border border-border px-2 py-0.5 text-caption font-mono text-fg group">
      {symbol}
      <button
        type="button"
        onClick={onRemove}
        aria-label={`Remove ${symbol}`}
        className="opacity-50 hover:opacity-100 hover:text-danger"
      >
        <X className="h-3 w-3" />
      </button>
    </span>
  );
}

function NewWatchlistDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [name, setName] = React.useState("");
  const [kind, setKind] = React.useState<WatchlistKind>("MANUAL");
  const [symbolsRaw, setSymbolsRaw] = React.useState("");
  const [config, setConfig] = React.useState<Record<string, unknown>>({});
  const create = useCreateTradingViewWatchlist();

  // Reset config to the picked kind's defaults whenever kind changes.
  React.useEffect(() => {
    setConfig({ ...WATCHLIST_KIND_META[kind].defaultConfig });
  }, [kind]);

  React.useEffect(() => {
    if (!open) {
      const t = setTimeout(() => {
        setName(""); setSymbolsRaw(""); setKind("MANUAL"); setConfig({});
      }, 250);
      return () => clearTimeout(t);
    }
  }, [open]);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const payload: Parameters<typeof create.mutateAsync>[0] = {
      name: name.trim(),
      kind,
      config,
    };
    if (kind === "MANUAL") {
      payload.symbols = symbolsRaw.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean);
    }
    try {
      await create.mutateAsync(payload);
      toast.success("Watchlist created");
      onClose();
    } catch (err) {
      const detail = (err as any)?.response?.data;
      const msg = detail?.name || detail?.config || detail?.detail || "Failed to create watchlist";
      toast.error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="w-[min(92vw,640px)] max-h-[85vh] overflow-auto">
        <DialogTitle>New watchlist</DialogTitle>
        <DialogDescription>
          Manual list, or one that auto-populates from an AlphaDesk source. Pick a kind below.
        </DialogDescription>

        <form onSubmit={onSubmit} className="mt-4 space-y-4">
          <Input
            label="Name"
            placeholder="e.g. NIFTY 50 — top picks"
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoFocus
          />

          <KindPicker value={kind} onChange={setKind} />

          {/* Kind-specific config inputs */}
          {kind === "MANUAL" && (
            <Input
              label="Symbols"
              hint="Comma or whitespace separated. Server uppercases + dedupes."
              placeholder="RELIANCE, TCS, HDFCBANK"
              value={symbolsRaw}
              onChange={(e) => setSymbolsRaw(e.target.value)}
            />
          )}

          <ConfigInputs kind={kind} config={config} onChange={setConfig} />

          <div className="flex items-center justify-end gap-2 pt-2">
            <Button type="button" variant="secondary" onClick={onClose}>Cancel</Button>
            <Button type="submit" loading={create.isPending} disabled={!name.trim()}>
              Create
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function KindPicker({
  value, onChange,
}: { value: WatchlistKind; onChange: (k: WatchlistKind) => void }) {
  const kinds = Object.entries(WATCHLIST_KIND_META) as [WatchlistKind, typeof WATCHLIST_KIND_META[WatchlistKind]][];
  return (
    <div>
      <div className="text-body-sm text-fg mb-1.5">Kind</div>
      <div className="grid grid-cols-1 gap-1.5">
        {kinds.map(([k, meta]) => (
          <label
            key={k}
            className={cn(
              "flex items-start gap-3 rounded-sm border p-3 cursor-pointer",
              value === k
                ? "border-accent/60 bg-accent/5"
                : "border-border hover:border-border-strong hover:bg-surface-2",
            )}
          >
            <input
              type="radio"
              name="kind"
              checked={value === k}
              onChange={() => onChange(k)}
              className="sr-only"
            />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-body-sm text-fg">{meta.label}</span>
                {meta.isAuto && <Badge tone="brand">Auto</Badge>}
              </div>
              <div className="text-caption text-fg-subtle mt-0.5">{meta.blurb}</div>
            </div>
          </label>
        ))}
      </div>
    </div>
  );
}

/** Per-kind config form fields. Keeps the dialog focused — each kind only
 *  asks for what its resolver actually reads. */
function ConfigInputs({
  kind, config, onChange,
}: {
  kind: WatchlistKind;
  config: Record<string, unknown>;
  onChange: (c: Record<string, unknown>) => void;
}) {
  const set = (k: string, v: unknown) => onChange({ ...config, [k]: v });

  if (kind === "MANUAL") return null;

  return (
    <div className="rounded-sm border border-border bg-surface-2 p-3 space-y-3">
      {kind === "SOURCE_HOT" && (
        <div>
          <div className="text-body-sm text-fg mb-1">Source</div>
          <select
            value={String(config.source ?? "TRADINGVIEW")}
            onChange={(e) => set("source", e.target.value)}
            className="h-9 w-full rounded-xs bg-surface border border-border px-2 text-body-sm"
          >
            <option value="TRADINGVIEW">TradingView</option>
            <option value="SCREENER">Screener</option>
            <option value="OK_SCANNER">OK Scanner</option>
            <option value="PREMARKET">Premarket basket</option>
          </select>
        </div>
      )}

      {(kind === "SIGNAL_RANK" || kind === "SOURCE_HOT" || kind === "TRADED_RECENTLY") && (
        <Input
          label="Window (days)"
          type="number"
          min={1}
          max={90}
          value={String(config.window_days ?? 7)}
          onChange={(e) => set("window_days", Number(e.target.value) || 7)}
        />
      )}

      {(kind === "SIGNAL_RANK" || kind === "SOURCE_HOT") && (
        <Input
          label="Top N"
          type="number"
          min={1}
          max={200}
          value={String(config.top_n ?? 20)}
          onChange={(e) => set("top_n", Number(e.target.value) || 20)}
        />
      )}

      {kind === "RECENT_ACTIVE" && (
        <Input
          label="Window (hours)"
          type="number"
          min={1}
          max={24 * 30}
          value={String(config.window_hours ?? 24)}
          onChange={(e) => set("window_hours", Number(e.target.value) || 24)}
        />
      )}

      {kind === "SHORTLIST_TODAY" && (
        <p className="text-caption text-fg-subtle">
          No knobs — pulls today's shortlist with outcomes WATCHING/TRIGGERED/TRADED.
        </p>
      )}
    </div>
  );
}


/* =================================================================== */
/* Grouped signals                                                      */
/* =================================================================== */

const GROUP_BY_OPTIONS: { key: SignalGroupBy; label: string }[] = [
  { key: "symbol",   label: "Symbol"   },
  { key: "strategy", label: "Strategy" },
  { key: "source",   label: "Source"   },
  { key: "day",      label: "Day"      },
];

function GroupedSignalsCard() {
  const [by, setBy] = React.useState<SignalGroupBy>("symbol");
  const [days, setDays] = React.useState(7);
  const [watchlistFilter, setWatchlistFilter] = React.useState<string>("");
  const { data: watchlists = [] } = useTradingViewWatchlists();
  const { data, isLoading, dataUpdatedAt } = useGroupedSignals({
    by, days, watchlist: watchlistFilter || undefined,
  });

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-3 flex-wrap">
        <div>
          <CardTitle>Signals</CardTitle>
          <CardDescription>
            Aggregated incoming signals — from TradingView, screener, OK scanner,
            premarket basket — within the chosen time window.
          </CardDescription>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <FreshnessIndicator
            label="Refreshed"
            timestamp={dataUpdatedAt}
            freshMs={30_000}
            staleMs={120_000}
          />
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="h-8 rounded-xs bg-surface border border-border px-2 text-body-sm"
            aria-label="Time window"
          >
            <option value={1}>Last 24h</option>
            <option value={7}>Last 7d</option>
            <option value={30}>Last 30d</option>
            <option value={90}>Last 90d</option>
          </select>
          {watchlists.length > 0 && (
            <select
              value={watchlistFilter}
              onChange={(e) => setWatchlistFilter(e.target.value)}
              className="h-8 rounded-xs bg-surface border border-border px-2 text-body-sm"
              aria-label="Filter by watchlist"
            >
              <option value="">All symbols</option>
              {watchlists.map((w) => (
                <option key={w.id} value={w.id}>{w.name}</option>
              ))}
            </select>
          )}
        </div>
      </CardHeader>

      <Tabs value={by} onValueChange={(v) => setBy(v as SignalGroupBy)} className="px-5">
        <TabsList>
          {GROUP_BY_OPTIONS.map((opt) => (
            <TabsTrigger key={opt.key} value={opt.key}>{opt.label}</TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        ) : !data || data.rows.length === 0 ? (
          <EmptyState
            icon={<Zap />}
            title="No signals yet"
            description={`Nothing fired in the last ${days} day${days === 1 ? "" : "s"}. Connect a TradingView alert above, or wait for the screener to pick something up.`}
          />
        ) : (
          <GroupedRowsTable rows={data.rows} by={by} />
        )}
      </CardContent>
    </Card>
  );
}

function GroupedRowsTable({ rows, by }: { rows: GroupedSignalRow[]; by: SignalGroupBy }) {
  const max = Math.max(...rows.map((r) => r.count), 1);
  return (
    <ul className="space-y-1" role="list">
      {rows.map((r) => (
        <li
          key={r.key}
          className="grid grid-cols-[1fr_auto_auto_auto_auto] items-center gap-3 px-2 py-2 rounded-xs hover:bg-surface-2"
        >
          {/* Bar + label */}
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-body-sm text-fg font-mono truncate" title={r.key}>
                {by === "day" ? formatDayLabel(r.key) : r.key || "(unknown)"}
              </span>
              {r.latest_action && (
                <Badge tone={r.latest_action === "SELL" ? "warning" : "success"}>
                  latest {r.latest_action}
                </Badge>
              )}
            </div>
            <div
              className="mt-1 h-1 rounded-full bg-surface overflow-hidden"
              aria-label={`${r.count} signals`}
            >
              <div
                className="h-full bg-accent"
                style={{ width: `${(r.count / max) * 100}%` }}
              />
            </div>
          </div>
          <span className="text-body-sm font-mono text-fg tabular w-12 text-right">{r.count}</span>
          <span className="text-caption text-pnl-up font-mono tabular w-8 text-right">+{r.buys}</span>
          <span className="text-caption text-pnl-down font-mono tabular w-8 text-right">−{r.sells}</span>
          <span className="text-caption text-fg-subtle font-mono w-20 text-right">
            {r.latest_at ? `${fmtRel(r.latest_at)} ago` : "—"}
          </span>
        </li>
      ))}
    </ul>
  );
}

function formatDayLabel(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString("en-IN", {
      month: "short", day: "numeric", year: "2-digit",
    });
  } catch {
    return iso;
  }
}


/* =================================================================== */
/* KPI helper                                                           */
/* =================================================================== */

function KpiCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="px-3 py-1.5">
      <dt className="text-caption text-fg-subtle uppercase tracking-wider">{label}</dt>
      <dd className="text-body-sm text-fg font-mono tabular mt-0.5">{value}</dd>
    </div>
  );
}
