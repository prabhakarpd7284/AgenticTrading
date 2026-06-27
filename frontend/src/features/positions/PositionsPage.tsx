import * as React from "react";
import { Briefcase, Download, LineChart as LineIcon, PauseCircle, X } from "lucide-react";

import { connect } from "@/lib/ws";
import { clsPnl, fmtInr } from "@/lib/utils";
import {
  usePositions, useTrades, useOptionsPositions,
  type EquityPosition, type OptionPosition, type Trade,
  type OptionsPositionRow,
} from "@/lib/v2";

import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import { DataTable, type Column } from "@/components/ui/DataTable";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import { FreshnessIndicator } from "@/components/ui/FreshnessIndicator";

type EnrichedEquity = EquityPosition & { ltp: number | null; liveUnrealized: number };

export function PositionsPage() {
  const { data: positions, isLoading: posLoading, dataUpdatedAt: posUpdatedAt } =
    usePositions();
  const { data: tradesBundle } = useTrades({ limit: 200 });
  const { data: straddleBundle } = useOptionsPositions();

  const equityOpen  = positions?.equity  ?? [];
  const optionsOpen = positions?.options ?? [];
  const closedTrades = (tradesBundle?.results ?? []).filter(
    (t) => t.status === "CLOSED" || t.pnl != null,
  );
  const closedStraddles = (straddleBundle?.results ?? []).filter(
    (s) => s.status === "CLOSED",
  );

  /* -------- live ticks -> ltp map -------- */
  const [ltps, setLtps] = React.useState<Record<string, number>>({});
  // Tracks the wall-clock time of the most recent tick — drives the
  // FreshnessIndicator in the header so the operator can see at a glance
  // if the broker tape has stalled.
  const [lastTickAt, setLastTickAt] = React.useState<number | null>(null);
  // Cleared LTPs are honest; a stale-but-shown LTP could mislead an exit
  // decision, so we drop the map on disconnect rather than silently keeping
  // old values around.
  const [wsLive, setWsLive] = React.useState(false);
  // Re-subscribe when the SET of symbols changes, not just the count — swapping
  // one open symbol for another (same count) must still re-subscribe the ticks.
  // Sorted so the key is order-independent (the subscription is set-based): a
  // backend row-order change can't churn it into a re-subscribe storm.
  const equitySymbolsKey = equityOpen.map((p) => p.symbol).sort().join(",");
  React.useEffect(() => {
    if (equityOpen.length === 0) return;
    const ws = connect(
      "/ws/ticks/",
      (m) => {
        const t = m as { token: string; symbol?: string; ltp: number };
        const key = t.symbol ?? t.token;
        if (key) {
          setLtps((prev) => ({ ...prev, [key]: t.ltp }));
          setLastTickAt(Date.now());
        }
      },
      {
        onOpen: () => {
          setWsLive(true);
          ws.send({ op: "subscribe", tokens: equityOpen.map((p) => p.symbol) });
        },
        onClose: () => {
          setWsLive(false);
          setLtps({});
        },
      },
    );
    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [equitySymbolsKey]);

  const equityRows: EnrichedEquity[] = React.useMemo(
    () =>
      equityOpen.map((p) => {
        const ltp = ltps[p.symbol] ?? null;
        const live = ltp
          ? (ltp - p.entry_price) * p.quantity * (p.side === "BUY" ? 1 : -1)
          : p.pnl ?? 0;
        return { ...p, ltp, liveUnrealized: live };
      }),
    [equityOpen, ltps],
  );

  const totalUnrealized =
    equityRows.reduce((s, r) => s + r.liveUnrealized, 0) +
    optionsOpen.reduce((s, r) => s + r.pnl_inr, 0);

  /* -------- equity columns -------- */
  const equityColumns: Column<EnrichedEquity>[] = [
    {
      key: "symbol", header: "Symbol", sortable: true,
      render: (p) => <span className="font-medium text-fg">{p.symbol}</span>,
    },
    {
      key: "side", header: "Side",
      render: (p) => <Badge tone={p.side === "BUY" ? "success" : "danger"}>{p.side}</Badge>,
    },
    { key: "quantity", header: "Qty", kind: "num", sortable: true },
    {
      key: "entry_price", header: "Entry", kind: "money", sortable: true,
      render: (p) => <span className="font-mono tabular">{fmtInr(p.entry_price)}</span>,
    },
    {
      key: "stop_loss", header: "Stop",
      render: (p) => <span className="font-mono tabular text-fg-muted">{fmtInr(p.stop_loss)}</span>,
    },
    {
      key: "target", header: "Target",
      render: (p) => <span className="font-mono tabular text-fg-muted">{fmtInr(p.target)}</span>,
    },
    {
      key: "ltp", header: "LTP", kind: "money",
      render: (p) => p.ltp == null
        ? <span className="text-fg-subtle">—</span>
        : <span className="font-mono tabular" aria-live="polite">{fmtInr(p.ltp)}</span>,
    },
    {
      key: "liveUnrealized", header: "Unrealized", kind: "money", sortable: true,
      render: (p) => (
        <span className={`font-mono tabular ${clsPnl(p.liveUnrealized)}`}>
          {fmtInr(p.liveUnrealized)}
        </span>
      ),
    },
    {
      key: "action" as any, header: "", align: "right",
      render: (p: EnrichedEquity) => (
        <div className="flex justify-end gap-1">
          <Button variant="ghost" size="sm" leading={<PauseCircle className="h-4 w-4" />}>Hedge</Button>
          <Button
            variant="ghost"
            size="sm"
            leading={<X className="h-4 w-4" />}
            className="text-pnl-down hover:text-pnl-down"
            onClick={() => console.log("close", p.id)}
          >
            Close
          </Button>
        </div>
      ),
    },
  ];

  /* -------- options columns -------- */
  const optionColumns: Column<OptionPosition>[] = [
    {
      key: "underlying", header: "Position", sortable: true,
      render: (p) => (
        <div>
          <div className="font-medium text-fg">{p.underlying} {p.strike} SHORT STRADDLE</div>
          <div className="text-caption text-fg-subtle font-mono">exp {p.expiry} · {p.dte}d</div>
        </div>
      ),
    },
    {
      key: "lots", header: "Size",
      render: (p) => <span className="font-mono tabular text-fg">{p.lots}×{p.lot_size}</span>,
    },
    {
      key: "ce_sell" as any, header: "Premium sold",
      render: (p: OptionPosition) => (
        <span className="font-mono tabular">{fmtInr((p.ce_sell + p.pe_sell) * p.lot_size * p.lots)}</span>
      ),
    },
    {
      key: "ce_current" as any, header: "Combined LTP",
      render: (p: OptionPosition) => (
        <span className="font-mono tabular">{fmtInr(p.ce_current + p.pe_current)}</span>
      ),
    },
    {
      key: "net_delta", header: "Δ",
      render: (p) => (
        <span className={`font-mono tabular ${Math.abs(p.net_delta) > 0.3 ? "text-warn" : "text-fg-muted"}`}>
          {p.net_delta.toFixed(2)}
        </span>
      ),
    },
    {
      key: "pnl_inr", header: "P&L", kind: "money", sortable: true,
      render: (p) => (
        <span className={`font-mono tabular ${clsPnl(p.pnl_inr)}`}>{fmtInr(p.pnl_inr)}</span>
      ),
    },
    {
      key: "status", header: "Status",
      render: (p) => <Badge tone={p.status === "ACTIVE" ? "info" : "neutral"}>{p.status}</Badge>,
    },
  ];

  /* -------- closed trades columns -------- */
  const closedColumns: Column<Trade>[] = [
    { key: "trade_date", header: "Date", sortable: true },
    {
      key: "symbol", header: "Symbol",
      render: (t) => <span className="font-medium text-fg">{t.symbol}</span>,
    },
    {
      key: "side", header: "Side",
      render: (t) => <Badge tone={t.side === "BUY" ? "success" : "danger"}>{t.side}</Badge>,
    },
    { key: "quantity", header: "Qty", kind: "num" },
    {
      key: "fill_price", header: "Fill",
      render: (t) => <span className="font-mono tabular">{t.fill_price ? fmtInr(t.fill_price) : "—"}</span>,
    },
    {
      key: "pnl", header: "P&L", kind: "money", sortable: true,
      render: (t) => (
        <span className={`font-mono tabular ${clsPnl(t.pnl ?? 0)}`}>
          {t.pnl == null ? "—" : fmtInr(t.pnl)}
        </span>
      ),
    },
    {
      key: "close_reason", header: "Exit reason",
      render: (t) => <span className="text-body-sm text-fg-muted">{t.close_reason || "—"}</span>,
    },
  ];

  const closedStraddleColumns: Column<OptionsPositionRow>[] = [
    { key: "trade_date", header: "Date", sortable: true },
    {
      key: "underlying", header: "Position",
      render: (p) => (
        <div>
          <div className="font-medium text-fg">{p.underlying} {p.strike}</div>
          <div className="text-caption text-fg-subtle font-mono">exp {p.expiry}</div>
        </div>
      ),
    },
    { key: "lots", header: "Lots", kind: "num" },
    {
      key: "premium_sold", header: "Premium",
      render: (p) => <span className="font-mono tabular">{fmtInr(p.premium_sold)}</span>,
    },
    {
      key: "pnl_inr", header: "P&L", kind: "money", sortable: true,
      render: (p) => (
        <span className={`font-mono tabular ${clsPnl(p.pnl_inr)}`}>{fmtInr(p.pnl_inr)}</span>
      ),
    },
    {
      key: "action_taken", header: "Closed by",
      render: (p) => <Badge tone="neutral">{p.action_taken || "—"}</Badge>,
    },
  ];

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <p className="text-caption uppercase tracking-wider text-fg-subtle">Positions</p>
          <h1 className="text-h1 text-fg">Open book</h1>
          <p className="text-body-sm text-fg-muted mt-1">
            {equityOpen.length} equity · {optionsOpen.length} options · unrealized{" "}
            <span className={`font-mono tabular ${clsPnl(totalUnrealized)}`}>
              {fmtInr(totalUnrealized)}
            </span>
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {/* Two freshness signals: the broker tick stream and the legacy
              positions snapshot. Both matter for an exit decision. */}
          {equityOpen.length > 0 && (
            <FreshnessIndicator
              label={wsLive ? "Last tick" : "Tick stream offline"}
              timestamp={lastTickAt}
              freshMs={3_000}
              staleMs={30_000}
            />
          )}
          <FreshnessIndicator
            label="Snapshot"
            timestamp={posUpdatedAt}
            freshMs={20_000}
            staleMs={60_000}
          />
          <Button variant="secondary" leading={<Download className="h-4 w-4" />}>Export CSV</Button>
        </div>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Positions</CardTitle>
          <CardDescription>
            Live from the legacy trading journal. Marks update every broker tick.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <Tabs defaultValue="equity" className="px-5 pt-4">
            <TabsList>
              <TabsTrigger value="equity">Equity ({equityOpen.length})</TabsTrigger>
              <TabsTrigger value="options">Options ({optionsOpen.length})</TabsTrigger>
              <TabsTrigger value="history">
                History ({closedTrades.length + closedStraddles.length})
              </TabsTrigger>
            </TabsList>
            <TabsContent value="equity">
              {equityRows.length === 0 ? (
                <EmptyState
                  className="my-4"
                  icon={<LineIcon />}
                  title="No open equity positions"
                  description="Intraday equity trades created by the desk or manually will appear here."
                />
              ) : (
                <DataTable<EnrichedEquity>
                  columns={equityColumns}
                  rows={equityRows}
                  loading={posLoading}
                  rowKey={(r) => String(r.id)}
                />
              )}
            </TabsContent>
            <TabsContent value="options">
              {optionsOpen.length === 0 ? (
                <EmptyState
                  className="my-4"
                  icon={<LineIcon />}
                  title="No active straddle positions"
                  description="Short-straddle positions registered via manage_straddle will appear here."
                />
              ) : (
                <DataTable<OptionPosition>
                  columns={optionColumns}
                  rows={optionsOpen}
                  rowKey={(r) => String(r.id)}
                />
              )}
            </TabsContent>
            <TabsContent value="history">
              {closedTrades.length === 0 && closedStraddles.length === 0 ? (
                <EmptyState
                  className="my-4"
                  icon={<Briefcase />}
                  title="No closed positions yet"
                  description="Realized trades and closed straddles will show up here with final P&L."
                />
              ) : (
                <div className="space-y-6 py-4">
                  {closedTrades.length > 0 && (
                    <div>
                      <h3 className="text-caption uppercase tracking-wider text-fg-subtle mb-2">
                        Equity trades ({closedTrades.length})
                      </h3>
                      <DataTable<Trade>
                        columns={closedColumns}
                        rows={closedTrades}
                        rowKey={(r) => String(r.id)}
                      />
                    </div>
                  )}
                  {closedStraddles.length > 0 && (
                    <div>
                      <h3 className="text-caption uppercase tracking-wider text-fg-subtle mb-2">
                        Closed straddles ({closedStraddles.length})
                      </h3>
                      <DataTable<OptionsPositionRow>
                        columns={closedStraddleColumns}
                        rows={closedStraddles}
                        rowKey={(r) => String(r.id)}
                      />
                    </div>
                  )}
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
