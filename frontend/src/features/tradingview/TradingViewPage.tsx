/**
 * TradingView Manager — sidebar + main pane layout.
 *
 *   ┌─Sidebar 260px ──────┬─Main pane ──────────────────────────┐
 *   │ WATCHLISTS    [+]   │ Signals · group by · filters         │
 *   │ ● All signals       │                                       │
 *   │ ─ Top-N rank        │ GROUPED ROWS (click → drill-in)       │
 *   │ ─ TV hot            │                                       │
 *   │                     │                                       │
 *   │ WEBHOOK LINKS [+]   │                                       │
 *   │ ─ TV alerts ●       │                                       │
 *   └─────────────────────┴───────────────────────────────────────┘
 *
 * Sidebar = filter + management list (compact rows). Main pane = signals
 * feed (primary). Sheets handle settings + drill-in so the main pane never
 * loses focus. Replaces the old three-section stack which forced operators
 * to scroll past management UI to see what was firing.
 */
import * as React from "react";
import { toast } from "sonner";
import {
  AlertCircle, ChevronDown, ChevronRight, Copy, ExternalLink, Plus,
  RefreshCw, Settings, Sparkles, Trash2, X, Zap,
} from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import {
  Dialog, DialogContent, DialogDescription, DialogTitle,
} from "@/components/ui/Dialog";
import { EmptyState } from "@/components/ui/EmptyState";
import { FreshnessIndicator } from "@/components/ui/FreshnessIndicator";
import { Input } from "@/components/ui/Input";
import { Skeleton } from "@/components/ui/Skeleton";
import {
  Sheet, SheetBody, SheetContent, SheetFooter, SheetHeader, SheetTitle,
} from "@/components/ui/Sheet";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/Tabs";

import {
  type GroupedSignalRow, type SignalGroupBy,
  type TradingViewLink, type TradingViewSignalRow,
  type WatchlistKind,
  WATCHLIST_KIND_META,
  useAddSymbolsToWatchlist, useCreateTradingViewLink, useCreateWatchlist,
  useDeleteTradingViewLink, useDeleteWatchlist,
  useGroupedSignals, useGroupedSignalsDetail,
  usePineScript, usePineStrategies, usePortfolios,
  useRefreshWatchlist, useRemoveSymbolsFromWatchlist,
  useRotateTradingViewSecret, useTradingViewLinks, useTradingViewRecent,
  useUpdateTradingViewLink, useUpdateWatchlist,
  useWatchlistKinds, useWatchlists,
} from "@/lib/v2";
import { cn, fmtRel, safeStringify } from "@/lib/utils";

const WATCHLIST_FILTER_ALL = "__all__";


export function TradingViewPage() {
  // Sidebar selection — "all signals" by default. Picking a watchlist
  // filters the main signals feed to its symbols.
  const [watchlistFilter, setWatchlistFilter] = React.useState<string>(WATCHLIST_FILTER_ALL);
  const [editingLinkId, setEditingLinkId] = React.useState<string | undefined>();
  const [editingWatchlistId, setEditingWatchlistId] = React.useState<string | undefined>();
  const [drillKey, setDrillKey] = React.useState<string | undefined>();
  const [newWatchlistOpen, setNewWatchlistOpen] = React.useState(false);
  const [newLinkOpen, setNewLinkOpen] = React.useState(false);

  // Signal-feed filters live here (not inside MainPane) so the drill-in sheet
  // can reuse the *exact* group-by / window / source the operator is looking
  // at. Otherwise clicking a row under the Strategy / Source / Day tabs would
  // query by=symbol with that row's key and return wrong or empty detail.
  const [groupBy, setGroupBy] = React.useState<SignalGroupBy>("symbol");
  const [days, setDays] = React.useState(7);
  const [source, setSource] = React.useState<string>("");

  const symbolFilter =
    watchlistFilter === WATCHLIST_FILTER_ALL ? undefined : watchlistFilter;

  return (
    <div className="grid grid-cols-[260px_1fr] min-h-[calc(100vh-3.5rem)]">
      <SidebarPane
        watchlistFilter={watchlistFilter}
        onWatchlistFilterChange={setWatchlistFilter}
        onEditWatchlist={setEditingWatchlistId}
        onEditLink={setEditingLinkId}
        onNewWatchlist={() => setNewWatchlistOpen(true)}
        onNewLink={() => setNewLinkOpen(true)}
      />

      <MainPane
        watchlistFilter={symbolFilter}
        groupBy={groupBy}
        onGroupByChange={setGroupBy}
        days={days}
        onDaysChange={setDays}
        source={source}
        onSourceChange={setSource}
        onRowDrill={setDrillKey}
      />

      {/* ── Sheets + Dialogs (rendered outside main flow) ─────────── */}
      <NewWatchlistDialog open={newWatchlistOpen} onClose={() => setNewWatchlistOpen(false)} />
      <NewLinkDialog open={newLinkOpen} onClose={() => setNewLinkOpen(false)} />
      <LinkSettingsSheet
        linkId={editingLinkId}
        onClose={() => setEditingLinkId(undefined)}
      />
      <WatchlistSheet
        watchlistId={editingWatchlistId}
        onClose={() => setEditingWatchlistId(undefined)}
      />
      <SignalDrillSheet
        drillKey={drillKey}
        by={groupBy}
        days={days}
        source={source}
        watchlistFilter={symbolFilter}
        onClose={() => setDrillKey(undefined)}
      />
    </div>
  );
}


/* =================================================================== */
/* Sidebar                                                              */
/* =================================================================== */

function SidebarPane({
  watchlistFilter, onWatchlistFilterChange,
  onEditWatchlist, onEditLink,
  onNewWatchlist, onNewLink,
}: {
  watchlistFilter: string;
  onWatchlistFilterChange: (id: string) => void;
  onEditWatchlist: (id: string) => void;
  onEditLink: (id: string) => void;
  onNewWatchlist: () => void;
  onNewLink: () => void;
}) {
  const { data: watchlists = [], isLoading: wlLoading } = useWatchlists();
  const { data: links = [], isLoading: linksLoading } = useTradingViewLinks();

  return (
    <aside className="border-r border-border bg-surface/50 flex flex-col min-h-0 overflow-auto">
      <div className="h-12 px-4 flex items-center border-b border-border sticky top-0 bg-surface/80 backdrop-blur z-sticky">
        <h2 className="text-body-sm font-semibold text-fg">TradingView</h2>
      </div>

      <SidebarSection
        title="Watchlists"
        count={watchlists.length}
        onAdd={onNewWatchlist}
        addLabel="New watchlist"
      >
        <SidebarRow
          selected={watchlistFilter === WATCHLIST_FILTER_ALL}
          onClick={() => onWatchlistFilterChange(WATCHLIST_FILTER_ALL)}
          title="All signals"
          subtitle="No symbol filter"
        />
        {wlLoading && <SidebarSkeleton n={3} />}
        {watchlists.map((wl) => (
          <SidebarRow
            key={wl.id}
            selected={watchlistFilter === wl.id}
            onClick={() => onWatchlistFilterChange(wl.id)}
            onEdit={() => onEditWatchlist(wl.id)}
            title={wl.name}
            kindBadge={wl.kind}
            subtitle={`${wl.symbol_count} symbol${wl.symbol_count === 1 ? "" : "s"}`}
            extraNote={wl.is_auto && wl.symbols_refreshed_at
              ? `· refresh ${fmtRel(wl.symbols_refreshed_at)} ago`
              : undefined}
          />
        ))}
        {!wlLoading && watchlists.length === 0 && (
          <p className="px-4 py-2 text-caption text-fg-subtle">No watchlists yet.</p>
        )}
      </SidebarSection>

      <SidebarSection
        title="Webhook links"
        count={links.length}
        onAdd={onNewLink}
        addLabel="Connect TradingView"
      >
        {linksLoading && <SidebarSkeleton n={2} />}
        {links.map((l) => (
          <SidebarRow
            key={l.id}
            onClick={() => onEditLink(l.id)}
            title={l.display_name || "Unnamed link"}
            statusDot={l.is_active ? "active" : "inactive"}
            subtitle={
              `${l.receive_count} alert${l.receive_count === 1 ? "" : "s"}` +
              (l.last_received_at ? ` · ${fmtRel(l.last_received_at)} ago` : "")
            }
            extraNote={l.autofire_enabled ? "· autofire on" : undefined}
          />
        ))}
        {!linksLoading && links.length === 0 && (
          <p className="px-4 py-2 text-caption text-fg-subtle">
            Connect a webhook to start receiving TradingView alerts.
          </p>
        )}
      </SidebarSection>
    </aside>
  );
}

function SidebarSection({
  title, count, onAdd, addLabel, children,
}: {
  title: string;
  count?: number;
  onAdd: () => void;
  addLabel: string;
  children: React.ReactNode;
}) {
  return (
    <div className="border-b border-border">
      <div className="px-4 py-2 flex items-center justify-between bg-surface-2/40">
        <span className="text-caption uppercase tracking-wider text-fg-subtle">
          {title}{count != null && ` · ${count}`}
        </span>
        <button
          type="button"
          onClick={onAdd}
          aria-label={addLabel}
          className="inline-flex h-6 w-6 items-center justify-center rounded-xs text-fg-subtle hover:text-fg hover:bg-surface-2"
        >
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>
      <ul className="py-1" role="list">{children}</ul>
    </div>
  );
}

function SidebarRow({
  title, subtitle, extraNote, kindBadge,
  selected, onClick, onEdit, statusDot,
}: {
  title: string;
  subtitle?: string;
  extraNote?: string;
  kindBadge?: WatchlistKind;
  selected?: boolean;
  onClick: () => void;
  onEdit?: () => void;
  statusDot?: "active" | "inactive";
}) {
  const meta = kindBadge ? WATCHLIST_KIND_META[kindBadge] : null;
  return (
    <li className={cn(
      "group flex items-center gap-2 px-4 py-1.5 text-body-sm cursor-pointer",
      selected ? "bg-accent/10 border-l-2 border-l-accent" : "hover:bg-surface-2",
    )}>
      {statusDot && (
        <span
          aria-hidden
          className={cn(
            "h-1.5 w-1.5 rounded-full shrink-0",
            statusDot === "active" ? "bg-pnl-up" : "bg-fg-subtle",
          )}
        />
      )}
      <button
        type="button"
        onClick={onClick}
        className="flex-1 min-w-0 text-left"
      >
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="truncate text-fg">{title}</span>
          {meta?.isAuto && (
            <Sparkles className="h-3 w-3 text-brand shrink-0" aria-label="auto" />
          )}
        </div>
        {(subtitle || extraNote) && (
          <div className="text-caption text-fg-subtle truncate">
            {subtitle}{extraNote && ` ${extraNote}`}
          </div>
        )}
      </button>
      {onEdit && (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onEdit(); }}
          aria-label={`Edit ${title}`}
          className="opacity-0 group-hover:opacity-100 h-6 w-6 inline-flex items-center justify-center rounded-xs text-fg-subtle hover:text-fg hover:bg-surface"
        >
          <Settings className="h-3.5 w-3.5" />
        </button>
      )}
    </li>
  );
}

function SidebarSkeleton({ n }: { n: number }) {
  return (
    <>
      {Array.from({ length: n }).map((_, i) => (
        <li key={i} className="px-4 py-1.5"><Skeleton className="h-8 w-full" /></li>
      ))}
    </>
  );
}


/* =================================================================== */
/* Main pane — signals feed                                             */
/* =================================================================== */

const GROUP_BY_OPTIONS: { key: SignalGroupBy; label: string }[] = [
  { key: "symbol",   label: "Symbol"   },
  { key: "strategy", label: "Strategy" },
  { key: "source",   label: "Source"   },
  { key: "day",      label: "Day"      },
];

function MainPane({
  watchlistFilter,
  groupBy, onGroupByChange,
  days, onDaysChange,
  source, onSourceChange,
  onRowDrill,
}: {
  watchlistFilter: string | undefined;
  groupBy: SignalGroupBy;
  onGroupByChange: (v: SignalGroupBy) => void;
  days: number;
  onDaysChange: (v: number) => void;
  source: string;
  onSourceChange: (v: string) => void;
  onRowDrill: (key: string) => void;
}) {
  const { data, isLoading, dataUpdatedAt } = useGroupedSignals({
    by: groupBy,
    days,
    source: source || undefined,
    watchlist: watchlistFilter,
  });

  const totalCount = (data?.rows || []).reduce((s, r) => s + r.count, 0);

  return (
    <section className="flex flex-col min-w-0">
      <header className="h-12 px-5 border-b border-border flex items-center gap-3 sticky top-0 bg-bg/80 backdrop-blur z-sticky">
        <h1 className="text-body-sm font-semibold text-fg flex items-center gap-2">
          Signals
          <Badge tone="neutral">{totalCount}</Badge>
        </h1>
        <div className="ml-auto flex items-center gap-2 flex-wrap">
          <FreshnessIndicator
            label="Refreshed"
            timestamp={dataUpdatedAt}
            freshMs={30_000}
            staleMs={120_000}
            compact
          />
          <select
            value={days}
            onChange={(e) => onDaysChange(Number(e.target.value))}
            aria-label="Time window"
            className="h-8 rounded-xs bg-surface border border-border px-2 text-body-sm"
          >
            <option value={1}>24h</option>
            <option value={7}>7d</option>
            <option value={30}>30d</option>
            <option value={90}>90d</option>
          </select>
          <select
            value={source}
            onChange={(e) => onSourceChange(e.target.value)}
            aria-label="Source"
            className="h-8 rounded-xs bg-surface border border-border px-2 text-body-sm"
          >
            <option value="">All sources</option>
            <option value="TRADINGVIEW">TradingView</option>
            <option value="SCREENER">Screener</option>
            <option value="OK_SCANNER">OK Scanner</option>
            <option value="PREMARKET">Premarket</option>
          </select>
        </div>
      </header>

      <Tabs value={groupBy} onValueChange={(v) => onGroupByChange(v as SignalGroupBy)}>
        <TabsList className="px-5 mt-3">
          {GROUP_BY_OPTIONS.map((opt) => (
            <TabsTrigger key={opt.key} value={opt.key}>{opt.label}</TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      <div className="flex-1 min-h-0 overflow-auto px-5 py-3">
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
            description={`Nothing fired in the last ${days} day${days === 1 ? "" : "s"}${
              watchlistFilter ? " for this watchlist" : ""
            }. Connect a TradingView webhook or wait for the screener to pick something up.`}
          />
        ) : (
          <GroupedRowsTable
            rows={data.rows}
            by={groupBy}
            onRowClick={onRowDrill}
          />
        )}
      </div>
    </section>
  );
}

function GroupedRowsTable({
  rows, by, onRowClick,
}: {
  rows: GroupedSignalRow[];
  by: SignalGroupBy;
  onRowClick: (key: string) => void;
}) {
  const max = Math.max(...rows.map((r) => r.count), 1);
  return (
    <ul className="space-y-1" role="list">
      {rows.map((r) => (
        <li key={r.key}>
          <button
            type="button"
            onClick={() => onRowClick(r.key)}
            className="w-full grid grid-cols-[1fr_auto_auto_auto_auto] items-center gap-3 px-2 py-2 rounded-xs hover:bg-surface-2 text-left"
            aria-label={`Open detail for ${r.key}`}
          >
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
                <div className="h-full bg-accent" style={{ width: `${(r.count / max) * 100}%` }} />
              </div>
            </div>
            <span className="text-body-sm font-mono text-fg tabular w-12 text-right">{r.count}</span>
            <span className="text-caption text-pnl-up font-mono tabular w-8 text-right">+{r.buys}</span>
            <span className="text-caption text-pnl-down font-mono tabular w-8 text-right">−{r.sells}</span>
            <span className="text-caption text-fg-subtle font-mono w-20 text-right">
              {r.latest_at ? `${fmtRel(r.latest_at)} ago` : "—"}
            </span>
          </button>
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
/* Signal drill-in sheet                                                */
/* =================================================================== */

function SignalDrillSheet({
  drillKey, by, days, source, watchlistFilter, onClose,
}: {
  drillKey: string | undefined;
  by: SignalGroupBy;
  days: number;
  source: string;
  watchlistFilter: string | undefined;
  onClose: () => void;
}) {
  // Mirror the feed's exact filters (lifted to TradingViewPage) so the raw
  // rows shown here are the ones that aggregated into the clicked bucket —
  // same group-by dimension, window, source, and watchlist gate.
  const { data, isLoading, isError } = useGroupedSignalsDetail({
    by,
    key: drillKey,
    days,
    source: source || undefined,
    watchlist: watchlistFilter,
  });
  const open = drillKey != null;

  return (
    <Sheet open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>
            {drillKey
              ? (by === "day" ? formatDayLabel(drillKey) : drillKey)
              : "Signal detail"}
          </SheetTitle>
        </SheetHeader>
        <SheetBody>
          {isLoading && <Skeleton className="h-40 w-full" />}
          {isError && (
            <div role="alert" className="text-body-sm text-danger">Couldn't load signal detail.</div>
          )}
          {data && data.rows.length === 0 && (
            <p className="text-body-sm text-fg-subtle">No raw signals in the window.</p>
          )}
          {data && data.rows.length > 0 && (
            <ul className="space-y-2" role="list">
              {data.rows.map((row) => (
                <li key={row.id} className="rounded-sm border border-border bg-surface-2 p-3 text-body-sm">
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge tone={row.side === "SELL" ? "warning" : "success"}>{row.side}</Badge>
                    <Badge tone="neutral">{row.source}</Badge>
                    {row.strategy && <span className="text-fg-muted">{row.strategy}</span>}
                    <span className="text-caption text-fg-subtle ml-auto" title={row.signal_time}>
                      {fmtRel(row.signal_time)} ago
                    </span>
                  </div>
                  <div className="mt-1 text-caption text-fg-muted font-mono">
                    entry {row.entry_price} · SL {row.stoploss} · TP {row.target}
                    {row.outcome && row.outcome !== "PENDING" && (
                      <> · <Badge tone="neutral">{row.outcome}</Badge></>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </SheetBody>
        <SheetFooter>
          <Button variant="secondary" size="sm" onClick={onClose}>Close</Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}


/* =================================================================== */
/* Link settings sheet (progressive)                                    */
/* =================================================================== */

function LinkSettingsSheet({
  linkId, onClose,
}: { linkId: string | undefined; onClose: () => void }) {
  const { data: links = [] } = useTradingViewLinks();
  const link = links.find((l) => l.id === linkId);
  const update = useUpdateTradingViewLink();
  const rotate = useRotateTradingViewSecret();
  const remove = useDeleteTradingViewLink();
  const { data: watchlists = [] } = useWatchlists();
  const { data: portfolios = [] } = usePortfolios();
  const { data: recent = [] } = useTradingViewRecent(linkId);

  const [draft, setDraft] = React.useState<TradingViewLink | undefined>(link);
  React.useEffect(() => setDraft(link), [link?.id, link?.updated_at]);

  // Section visibility — collapsed by default for "Auto-fire" unless it's
  // already on (operator clearly cares) and "Recent alerts" always.
  const [autofireOpen, setAutofireOpen] = React.useState(false);
  const [recentOpen, setRecentOpen] = React.useState(false);
  React.useEffect(() => {
    if (link) setAutofireOpen(link.autofire_enabled);
  }, [link?.id, link?.autofire_enabled]);

  if (!draft) {
    return (
      <Sheet open={false} onOpenChange={(o) => { if (!o) onClose(); }}>
        <SheetContent><div /></SheetContent>
      </Sheet>
    );
  }

  const patch = async (changes: Partial<TradingViewLink>) => {
    const next = { ...draft, ...changes };
    setDraft(next);
    await update.mutateAsync({ id: next.id, ...changes });
  };

  const onCopy = async () => {
    await navigator.clipboard.writeText(draft.webhook_url);
    toast.success("Webhook URL copied");
  };

  const onRotate = async () => {
    if (!window.confirm("Rotate the webhook secret? The old URL stops working immediately.")) return;
    await rotate.mutateAsync(draft.id);
    toast.success("Rotated — paste the new URL into TradingView.");
  };

  const onDelete = async () => {
    if (!window.confirm(`Delete "${draft.display_name || "unnamed"}"?`)) return;
    await remove.mutateAsync(draft.id);
    toast.success("Link removed");
    onClose();
  };

  return (
    <Sheet open={!!link} onOpenChange={(o) => { if (!o) onClose(); }}>
      <SheetContent>
        <SheetHeader>
          <div className="flex items-center gap-2 flex-wrap">
            <SheetTitle className="flex-1 min-w-0 truncate">{draft.display_name || "Unnamed link"}</SheetTitle>
            <ToggleRow
              compact
              label="Active"
              checked={draft.is_active}
              onChange={(v) => patch({ is_active: v })}
            />
          </div>
        </SheetHeader>

        <SheetBody>
          <div className="space-y-4">
            {/* Basics */}
            <Input
              label="Name"
              value={draft.display_name}
              onChange={(e) => setDraft({ ...draft, display_name: e.target.value })}
              onBlur={() => patch({ display_name: draft.display_name })}
            />

            {/* Webhook URL */}
            <div>
              <div className="text-caption uppercase tracking-wider text-fg-subtle mb-1">Webhook URL</div>
              <div className="flex items-center gap-2 rounded-sm border border-border bg-surface px-2 py-1.5">
                <code className="flex-1 text-caption font-mono text-fg-muted truncate" title={draft.webhook_url}>
                  {draft.webhook_url}
                </code>
                <Button size="sm" variant="secondary" onClick={onCopy} leading={<Copy className="h-3.5 w-3.5" />}>
                  Copy
                </Button>
                <Button
                  size="sm" variant="secondary"
                  onClick={onRotate} loading={rotate.isPending}
                  leading={<RefreshCw className="h-3.5 w-3.5" />}
                >
                  Rotate
                </Button>
              </div>
              {draft.last_error && (
                <div role="alert" className="flex items-start gap-2 text-caption text-warn mt-2">
                  <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" aria-hidden />
                  <span className="break-all">{draft.last_error}</span>
                </div>
              )}
            </div>

            {/* Auto-fire — collapsible */}
            <CollapsibleSection
              title="Auto-fire workflow"
              statusBadge={draft.autofire_enabled ? <Badge tone="warning">on</Badge> : undefined}
              open={autofireOpen}
              onOpenChange={setAutofireOpen}
            >
              <ToggleRow
                label="Spawn an AgentRun on every parsed alert"
                hint="RiskGuard still gates execution."
                checked={draft.autofire_enabled}
                onChange={(v) => patch({ autofire_enabled: v })}
              />
              {draft.autofire_enabled && (
                <div className="mt-3 space-y-3">
                  {/* Portfolio is mandatory for autofire — fire_workflow()
                      silently skips with reason="no_portfolio" when it's
                      unset, so surface a warning rather than letting the
                      operator think autofire is armed when it isn't. */}
                  <div>
                    <div className="text-body-sm text-fg mb-1">Portfolio</div>
                    <select
                      value={draft.portfolio || ""}
                      onChange={(e) => patch({ portfolio: e.target.value || null })}
                      className="h-9 w-full rounded-xs bg-surface border border-border px-2 text-body-sm"
                    >
                      <option value="">Select a portfolio…</option>
                      {portfolios.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.name} · {p.mode}
                        </option>
                      ))}
                    </select>
                    {!draft.portfolio && (
                      <div role="alert" className="flex items-start gap-2 text-caption text-warn mt-2">
                        <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" aria-hidden />
                        <span>Autofire won't fire until a portfolio is selected.</span>
                      </div>
                    )}
                  </div>
                  <Input
                    label="Strategy name"
                    hint="One of: directional, short_straddle, pyramid, ..."
                    value={draft.default_strategy_name}
                    onChange={(e) => setDraft({ ...draft, default_strategy_name: e.target.value })}
                    onBlur={() => patch({ default_strategy_name: draft.default_strategy_name })}
                  />
                  <Input
                    label="Allowed actions (comma-separated)"
                    hint="Empty = allow all. Example: BUY,SELL"
                    value={(draft.allowed_actions || []).join(",")}
                    onChange={(e) => setDraft({
                      ...draft,
                      allowed_actions: e.target.value.split(",").map((s) => s.trim().toUpperCase()).filter(Boolean),
                    })}
                    onBlur={() => patch({ allowed_actions: draft.allowed_actions })}
                  />
                  <div>
                    <div className="text-body-sm text-fg mb-1">Restrict to watchlist (optional)</div>
                    <select
                      value={draft.watchlist || ""}
                      onChange={(e) => patch({ watchlist: e.target.value || null })}
                      className="h-9 w-full rounded-xs bg-surface border border-border px-2 text-body-sm"
                    >
                      <option value="">No symbol gate</option>
                      {watchlists.map((w) => (
                        <option key={w.id} value={w.id}>
                          {w.name} · {w.symbol_count} symbol{w.symbol_count === 1 ? "" : "s"}
                          {w.is_auto ? " (auto)" : ""}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              )}
            </CollapsibleSection>

            {/* Recent alerts — collapsible */}
            <CollapsibleSection
              title="Recent alerts"
              statusBadge={<Badge tone="neutral">{recent.length}</Badge>}
              open={recentOpen}
              onOpenChange={setRecentOpen}
            >
              {recent.length === 0 ? (
                <p className="text-caption text-fg-subtle">No alerts received yet.</p>
              ) : (
                <ul className="space-y-2" role="list">
                  {recent.slice(0, 10).map((r) => <RecentAlertRow key={r.id} row={r} />)}
                </ul>
              )}
            </CollapsibleSection>

            {/* Pine Script export — generate a TradingView indicator from one of
                our screener strategies, pre-wired to this link's webhook. */}
            <PineScriptSection linkId={draft.id} />
          </div>
        </SheetBody>

        <SheetFooter>
          <Button
            size="sm" variant="secondary"
            onClick={onDelete} loading={remove.isPending}
            leading={<Trash2 className="h-3.5 w-3.5" />}
          >
            Delete link
          </Button>
          <Button variant="secondary" size="sm" onClick={onClose}>Close</Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}

function PineScriptSection({ linkId }: { linkId: string }) {
  const [open, setOpen] = React.useState(false);
  const { data: strategies = [] } = usePineStrategies();
  const [strategy, setStrategy] = React.useState<string>("");

  // Default to the first strategy once the list resolves.
  React.useEffect(() => {
    if (!strategy && strategies.length) setStrategy(strategies[0].key);
  }, [strategies, strategy]);

  // Only generate while the section is expanded — keeps the network quiet
  // for operators who never open it.
  const { data, isFetching } = usePineScript(open ? strategy : undefined, linkId);

  const onCopy = async () => {
    if (!data?.code) return;
    await navigator.clipboard.writeText(data.code);
    toast.success("Pine Script copied");
  };

  return (
    <CollapsibleSection
      title="Pine Script"
      statusBadge={<Badge tone="brand">{strategies.length}</Badge>}
      open={open}
      onOpenChange={setOpen}
    >
      <p className="text-caption text-fg-subtle mb-2">
        Export a screener strategy as a TradingView indicator. Paste it into the
        Pine editor, add it to a chart, then create an alert on{" "}
        <span className="text-fg-muted">“Any alert() function call”</span> with{" "}
        <span className="text-fg-muted">Once Per Bar Close</span> and this link’s
        webhook URL — alerts feed straight back into AlphaDesk.
      </p>
      <select
        value={strategy}
        onChange={(e) => setStrategy(e.target.value)}
        aria-label="Strategy to export"
        className="h-9 w-full rounded-xs bg-surface border border-border px-2 text-body-sm mb-2"
      >
        {strategies.map((s) => (
          <option key={s.key} value={s.key}>{s.label} · {s.side}</option>
        ))}
      </select>
      {data?.code ? (
        <div className="space-y-2">
          <div className="flex justify-end">
            <Button
              size="sm" variant="secondary"
              onClick={onCopy}
              leading={<Copy className="h-3.5 w-3.5" />}
            >
              Copy
            </Button>
          </div>
          <pre className="max-h-72 overflow-auto rounded-sm border border-border bg-surface-2 p-3 text-caption font-mono whitespace-pre">
            {data.code}
          </pre>
        </div>
      ) : (
        <p className="text-caption text-fg-subtle">
          {isFetching ? "Generating…" : "Select a strategy to generate Pine Script."}
        </p>
      )}
    </CollapsibleSection>
  );
}

function RecentAlertRow({ row }: { row: TradingViewSignalRow }) {
  const parsed = row.parsed as { symbol?: string; action?: string; price?: number };
  return (
    <li className="rounded-sm border border-border bg-surface px-3 py-2">
      <div className="flex items-center gap-2 text-body-sm">
        <span className="text-fg-subtle font-mono">{fmtRel(row.received_at)} ago</span>
        {row.parse_error ? (
          <Badge tone="danger">parse failed</Badge>
        ) : (
          <>
            <Badge tone={parsed.action === "SELL" ? "warning" : "success"}>{parsed.action || "?"}</Badge>
            <span className="text-fg">{parsed.symbol || "?"}</span>
            {parsed.price != null && (
              <span className="text-fg-subtle font-mono">@ {parsed.price}</span>
            )}
          </>
        )}
        {row.workflow_run && <Badge tone="brand">Run fired</Badge>}
      </div>
      {row.parse_error && (
        <div className="text-caption text-warn mt-1 break-all">{row.parse_error}</div>
      )}
    </li>
  );
}


/* =================================================================== */
/* Watchlist edit sheet                                                 */
/* =================================================================== */

function WatchlistSheet({
  watchlistId, onClose,
}: { watchlistId: string | undefined; onClose: () => void }) {
  const { data: watchlists = [] } = useWatchlists();
  const watchlist = watchlists.find((w) => w.id === watchlistId);
  const update = useUpdateWatchlist();
  const remove = useDeleteWatchlist();
  const add = useAddSymbolsToWatchlist();
  const rm = useRemoveSymbolsFromWatchlist();
  const refresh = useRefreshWatchlist();

  const [name, setName] = React.useState(watchlist?.name || "");
  const [newSymbol, setNewSymbol] = React.useState("");
  React.useEffect(() => setName(watchlist?.name || ""), [watchlist?.id, watchlist?.name]);

  if (!watchlist) {
    return (
      <Sheet open={false} onOpenChange={(o) => { if (!o) onClose(); }}>
        <SheetContent><div /></SheetContent>
      </Sheet>
    );
  }

  const meta = WATCHLIST_KIND_META[watchlist.kind];

  const onAddSymbol = async (e: React.FormEvent) => {
    e.preventDefault();
    const sym = newSymbol.trim();
    if (!sym) return;
    await add.mutateAsync({ id: watchlist.id, symbols: [sym] });
    setNewSymbol("");
  };

  const onRemoveOne = (sym: string) => rm.mutateAsync({ id: watchlist.id, symbols: [sym] });

  const onDelete = async () => {
    if (!window.confirm(`Delete watchlist "${watchlist.name}"?`)) return;
    await remove.mutateAsync(watchlist.id);
    toast.success("Watchlist deleted");
    onClose();
  };

  const onRename = async () => {
    if (name.trim() && name.trim() !== watchlist.name) {
      await update.mutateAsync({ id: watchlist.id, name: name.trim() });
    }
  };

  const onRefresh = async () => {
    await refresh.mutateAsync(watchlist.id);
    toast.success("Watchlist refreshed");
  };

  return (
    <Sheet open={!!watchlistId} onOpenChange={(o) => { if (!o) onClose(); }}>
      <SheetContent>
        <SheetHeader>
          <div className="flex items-center gap-2">
            <Badge tone={watchlist.is_auto ? "brand" : "neutral"}>
              {watchlist.is_auto && <Sparkles className="h-3 w-3 mr-1" aria-hidden />}
              {meta.label}
            </Badge>
            <SheetTitle className="flex-1 truncate">{watchlist.name}</SheetTitle>
          </div>
        </SheetHeader>

        <SheetBody>
          <div className="space-y-4">
            <Input
              label="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onBlur={onRename}
            />

            {watchlist.is_auto && (
              <div className="rounded-md border border-border bg-surface-2 p-3 text-caption text-fg-subtle">
                <div className="font-mono break-all">config {JSON.stringify(watchlist.config)}</div>
                {watchlist.symbols_refreshed_at && (
                  <div className="mt-1">
                    Refreshed {fmtRel(watchlist.symbols_refreshed_at)} ago
                  </div>
                )}
              </div>
            )}

            <div>
              <div className="text-caption uppercase tracking-wider text-fg-subtle mb-2">
                Symbols · {watchlist.symbol_count}
              </div>
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
                    {newSymbol && <Button type="submit" size="sm" loading={add.isPending}>Add</Button>}
                  </form>
                )}
              </div>
            </div>
          </div>
        </SheetBody>

        <SheetFooter>
          {watchlist.is_auto && (
            <Button
              size="sm" variant="secondary"
              onClick={onRefresh} loading={refresh.isPending}
              leading={<RefreshCw className="h-3.5 w-3.5" />}
            >
              Refresh
            </Button>
          )}
          <Button
            size="sm" variant="secondary"
            onClick={onDelete} loading={remove.isPending}
            leading={<Trash2 className="h-3.5 w-3.5" />}
          >
            Delete
          </Button>
          <Button variant="secondary" size="sm" onClick={onClose}>Close</Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
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


/* =================================================================== */
/* New-watchlist dialog                                                 */
/* =================================================================== */

function NewWatchlistDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [name, setName] = React.useState("");
  const [kind, setKind] = React.useState<WatchlistKind>("MANUAL");
  const [symbolsRaw, setSymbolsRaw] = React.useState("");
  const [config, setConfig] = React.useState<Record<string, unknown>>({});
  const create = useCreateWatchlist();
  const { data: kindsMeta } = useWatchlistKinds();

  const defaultsByKind = React.useMemo(() => {
    const m: Partial<Record<WatchlistKind, Record<string, unknown>>> = {};
    for (const k of kindsMeta || []) m[k.kind] = k.defaults;
    return m;
  }, [kindsMeta]);

  React.useEffect(() => {
    setConfig({ ...(defaultsByKind[kind] || {}) });
  }, [kind, defaultsByKind]);

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
      const msg = detail?.name || detail?.config || detail?.detail || "Failed to create";
      toast.error(typeof msg === "string" ? msg : safeStringify(msg));
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
          min={1} max={90}
          value={String(config.window_days ?? 7)}
          onChange={(e) => set("window_days", Number(e.target.value) || 7)}
        />
      )}

      {(kind === "SIGNAL_RANK" || kind === "SOURCE_HOT") && (
        <Input
          label="Top N"
          type="number"
          min={1} max={200}
          value={String(config.top_n ?? 20)}
          onChange={(e) => set("top_n", Number(e.target.value) || 20)}
        />
      )}

      {kind === "RECENT_ACTIVE" && (
        <Input
          label="Window (hours)"
          type="number"
          min={1} max={24 * 30}
          value={String(config.window_hours ?? 24)}
          onChange={(e) => set("window_hours", Number(e.target.value) || 24)}
        />
      )}

      {kind === "SHORTLIST_TODAY" && (
        <p className="text-caption text-fg-subtle">
          No knobs — pulls today's shortlist (WATCHING / TRIGGERED / TRADED).
        </p>
      )}
    </div>
  );
}


/* =================================================================== */
/* New-link dialog (replaces the old ConnectDialog)                     */
/* =================================================================== */

function NewLinkDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [name, setName] = React.useState("");
  const create = useCreateTradingViewLink();
  const [created, setCreated] = React.useState<TradingViewLink | null>(null);

  React.useEffect(() => {
    if (!open) {
      const t = setTimeout(() => { setName(""); setCreated(null); }, 250);
      return () => clearTimeout(t);
    }
  }, [open]);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const link = await create.mutateAsync({ display_name: name.trim() || "TradingView alert" });
    setCreated(link);
  };

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="w-[min(92vw,560px)] max-h-[85vh] overflow-auto">
        <DialogTitle>Connect TradingView</DialogTitle>
        <DialogDescription>
          Generates a private webhook URL. Paste it into TradingView's alert dialog under
          Notifications → Webhook URL.
        </DialogDescription>

        {!created ? (
          <form onSubmit={onSubmit} className="mt-4 space-y-3">
            <Input
              label="Name"
              hint="Operator-facing label."
              placeholder="My TradingView strategy"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
            />
            <div className="flex items-center justify-end gap-2 pt-2">
              <Button type="button" variant="secondary" onClick={onClose}>Cancel</Button>
              <Button type="submit" loading={create.isPending}>Create webhook URL</Button>
            </div>
          </form>
        ) : (
          <CreatedView link={created} onDone={onClose} />
        )}
      </DialogContent>
    </Dialog>
  );
}

function CreatedView({ link, onDone }: { link: TradingViewLink; onDone: () => void }) {
  const [copied, setCopied] = React.useState(false);
  const onCopy = async () => {
    await navigator.clipboard.writeText(link.webhook_url);
    setCopied(true);
    toast.success("Webhook URL copied");
  };
  return (
    <div className="mt-4 space-y-4">
      <div className="rounded-md border border-pnl-up/30 bg-pnl-up/5 p-3 text-body-sm text-fg">
        Webhook created. <strong>Copy the URL now</strong> and paste it into TradingView.
      </div>
      <div>
        <div className="text-caption uppercase tracking-wider text-fg-subtle mb-1">Webhook URL</div>
        <div className="flex items-center gap-2 rounded-sm border border-border bg-surface-2 px-2 py-2">
          <code className="flex-1 text-caption font-mono text-fg break-all">{link.webhook_url}</code>
          <Button size="sm" onClick={onCopy} leading={<Copy className="h-3.5 w-3.5" />}>
            {copied ? "Copied" : "Copy"}
          </Button>
        </div>
      </div>
      <details className="text-caption text-fg-muted">
        <summary className="cursor-pointer text-fg-subtle hover:text-fg select-none">
          Sample TradingView alert message
        </summary>
        <pre className="mt-2 rounded-sm bg-surface-2 border border-border p-3 whitespace-pre-wrap break-all font-mono">
{`{
  "symbol": "{{ticker}}",
  "action": "{{strategy.order.action}}",
  "price": {{close}},
  "comment": "{{strategy.order.comment}}"
}`}
        </pre>
        <p className="mt-2">
          Plain text like <code>BUY HDFCBANK @ 1500</code> is also accepted.
        </p>
      </details>
      <div className="flex items-center justify-end gap-2">
        <Button onClick={onDone} leading={<ExternalLink className="h-3.5 w-3.5" />}>Done</Button>
      </div>
    </div>
  );
}


/* =================================================================== */
/* Shared helpers                                                       */
/* =================================================================== */

function CollapsibleSection({
  title, statusBadge, open, onOpenChange, children,
}: {
  title: string;
  statusBadge?: React.ReactNode;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-md border border-border">
      <button
        type="button"
        onClick={() => onOpenChange(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 text-body-sm text-fg hover:bg-surface-2"
        aria-expanded={open}
      >
        {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        <span className="font-semibold">{title}</span>
        {statusBadge && <span className="ml-auto">{statusBadge}</span>}
      </button>
      {open && <div className="px-3 pb-3 pt-1">{children}</div>}
    </div>
  );
}

function ToggleRow({
  label, hint, checked, onChange, compact,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  compact?: boolean;
}) {
  return (
    <label className={cn("flex cursor-pointer gap-3", compact ? "items-center" : "items-start")}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className={cn(
          "h-4 w-4 rounded-xs border-border bg-surface",
          "checked:bg-accent checked:border-accent",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
          !compact && "mt-1",
        )}
      />
      <div className="flex-1">
        <div className="text-body-sm text-fg">{label}</div>
        {hint && <div className="text-caption text-fg-subtle mt-0.5">{hint}</div>}
      </div>
    </label>
  );
}
