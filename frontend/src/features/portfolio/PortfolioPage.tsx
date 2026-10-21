import * as React from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { AlertTriangle, PieChart, PlugZap } from "lucide-react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip as ReTooltip, XAxis, YAxis,
} from "recharts";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorState } from "@/components/ui/ErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import { PositionsPage } from "@/features/positions/PositionsPage";
import OptionsDashboardPage from "@/features/options/OptionsDashboardPage";
import { MutualFundsPage } from "@/features/mutual-funds/MutualFundsPage";
import { MonthlyPage } from "@/features/monthly/MonthlyPage";
import { clsPnl, cn, fmtInr, fmtRel } from "@/lib/utils";
import {
  usePortfolioAttribution, usePortfolioOverview,
  type AttributionGroup, type ComponentStatus, type Freshness, type PortfolioOverview,
} from "./usePortfolio";

export const PORTFOLIO_TABS = [
  { key: "overview", label: "Overview" },
  { key: "positions", label: "Positions" },
  { key: "options", label: "Options" },
  { key: "mutual-funds", label: "Mutual Funds" },
  { key: "performance", label: "Performance" },
  { key: "attribution", label: "Attribution" },
] as const;
type TabKey = (typeof PORTFOLIO_TABS)[number]["key"];

const STATUS_TEXT: Record<ComponentStatus, string> = {
  ok: "connected", empty: "nothing held", not_connected: "not connected",
};
const STATUS_TONE: Record<ComponentStatus, "success" | "neutral" | "warning"> = {
  ok: "success", empty: "neutral", not_connected: "warning",
};

/**
 * /portfolio[/:tab] — the Portfolio destination (UX redesign R2, plan §5).
 * Overview and Attribution are new; the other tabs embed the existing pages
 * unchanged (Positions, Options Desk, Mutual Funds, Monthly).
 */
export function PortfolioPage() {
  const { tab } = useParams<{ tab?: string }>();
  const navigate = useNavigate();
  const active: TabKey = PORTFOLIO_TABS.some((t) => t.key === tab) ? (tab as TabKey) : "overview";

  return (
    <div className="space-y-2">
      <Tabs value={active}
            onValueChange={(v) => navigate(v === "overview" ? "/portfolio" : `/portfolio/${v}`)}>
        <div className="px-6 pt-6">
          <TabsList aria-label="Portfolio views">
            {PORTFOLIO_TABS.map((t) => <TabsTrigger key={t.key} value={t.key}>{t.label}</TabsTrigger>)}
          </TabsList>
        </div>
        <TabsContent value="overview"><OverviewTab /></TabsContent>
        <TabsContent value="positions"><PositionsPage /></TabsContent>
        <TabsContent value="options"><OptionsDashboardPage /></TabsContent>
        <TabsContent value="mutual-funds"><MutualFundsPage /></TabsContent>
        <TabsContent value="performance"><MonthlyPage /></TabsContent>
        <TabsContent value="attribution"><AttributionTab /></TabsContent>
      </Tabs>
    </div>
  );
}

/* ================================================================== */
/* Overview                                                            */
/* ================================================================== */

function OverviewTab() {
  const q = usePortfolioOverview();
  if (q.isLoading) {
    return <div className="p-6 grid gap-4 md:grid-cols-3">{[0, 1, 2, 3, 4, 5].map((i) =>
      <Skeleton key={i} className="h-24 w-full" />)}</div>;
  }
  if (q.isError || !q.data) {
    return <div className="p-6"><ErrorState title="Could not load the portfolio overview"
                                            description={(q.error as Error | undefined)?.message}
                                            onRetry={() => q.refetch()} /></div>;
  }
  const d = q.data;
  return (
    <div className="p-6 pt-2 space-y-6 max-w-[1400px]">
      <header className="flex flex-wrap items-end gap-3">
        <div className="flex-1 min-w-[240px]">
          <h2 className="text-h2 text-fg">Portfolio</h2>
          <p className="text-body-sm text-fg-muted">
            Everything you hold, in one place. Missing sources show as <em>not connected</em>, never as zero.
          </p>
        </div>
        <Badge tone={d.mode === "live" ? "danger" : "warning"} dot>{d.mode.toUpperCase()}</Badge>
        <span className="text-caption text-fg-subtle">as of {fmtRel(d.as_of)} ago</span>
      </header>

      {d.total.components_missing.length > 0 && (
        <div role="note" className="flex items-start gap-2 rounded-sm border border-warn/40 bg-warn/10 px-3 py-2 text-body-sm text-fg">
          <PlugZap className="h-4 w-4 mt-0.5 text-warn" aria-hidden />
          <span>
            Totals exclude {d.total.components_missing.map(labelOf).join(", ")} (not connected).
            {d.components.mutual_funds.status === "not_connected" && d.components.mutual_funds.reason
              ? ` ${d.components.mutual_funds.reason}` : ""}
          </span>
        </div>
      )}

      <section aria-label="Headline numbers" className="grid gap-3 grid-cols-2 lg:grid-cols-3 2xl:grid-cols-6">
        <ValueTile label="Total value" value={d.total.value}
                   hint={d.total.mutual_funds_value == null ? "trading book only" : "trading book + MFs"} />
        <ValueTile label="Cash" value={d.total.cash} hint={`capital ${fmtInr(d.total.capital)}`} />
        <ValueTile label="Invested" value={d.total.invested}
                   hint={d.components.equity.open_count + d.components.options.open_count === 0
                     ? "nothing deployed" : "open cost basis"} />
        <PnlTile label="Day P&L" value={d.pnl.day.total} emptyLabel={d.pnl.label} />
        <PnlTile label="Month P&L" value={d.pnl.month.total} emptyLabel={d.pnl.label} />
        <PnlTile label="Since start" value={d.pnl.since_start.total} emptyLabel={d.pnl.label} />
      </section>

      <section aria-label="Holdings by asset class" className="grid gap-4 lg:grid-cols-3">
        <ComponentCard title="Equity" to="/portfolio/positions" status={d.components.equity.status}
                       label={d.components.equity.label} freshness={d.components.equity.freshness}
                       rows={[
                         ["Market value", fmtInr(d.components.equity.market_value)],
                         ["Open P&L", fmtInr(d.components.equity.unrealized_pnl), d.components.equity.unrealized_pnl],
                         ["Pending orders", String(d.components.equity.pending_orders)],
                       ]} />
        <ComponentCard title="Options" to="/portfolio/options" status={d.components.options.status}
                       label={d.components.options.label} freshness={d.components.options.freshness}
                       rows={[
                         ["Open P&L", fmtInr(d.components.options.unrealized_pnl), d.components.options.unrealized_pnl],
                         ["Premium sold", fmtInr(d.components.options.short_premium_sold)],
                         ["Premium paid", fmtInr(d.components.options.long_premium_paid)],
                       ]} />
        <ComponentCard title="Mutual funds" to="/portfolio/mutual-funds" status={d.components.mutual_funds.status}
                       label={d.components.mutual_funds.label} freshness={d.components.mutual_funds.freshness}
                       rows={[
                         ["Current value", fmtInr(d.components.mutual_funds.current_value)],
                         ["Gain", fmtInr(d.components.mutual_funds.gain), d.components.mutual_funds.gain],
                         ["NAV as of", d.components.mutual_funds.catalog_nav_date ?? "—"],
                       ]}
                       extra={d.components.mutual_funds.stale_holdings > 0
                         ? `${d.components.mutual_funds.stale_holdings} holding(s) on a stale NAV` : undefined} />
      </section>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <EquityCurveCard data={d} />
        <RiskCard data={d} />
      </div>

      <section aria-label="Allocation" className="grid gap-4 lg:grid-cols-3">
        <AllocationCard title="By asset class"
                        rows={d.allocation.by_asset_class.map((r) => ({
                          key: r.label, value: r.value, pct: r.pct,
                          note: r.value == null ? STATUS_TEXT[r.status as ComponentStatus] ?? r.status : undefined,
                        }))} />
        <AllocationCard title="By strategy" rows={d.allocation.by_strategy}
                        empty="No open positions or holdings to attribute." />
        <AllocationCard title="By sector" rows={d.allocation.by_sector}
                        empty="No open equity or option positions." />
      </section>

      <BrokerCard data={d} />
    </div>
  );
}

function labelOf(key: string) {
  return { equity: "equity", options: "options", mutual_funds: "mutual funds" }[key] ?? key;
}

function ValueTile({ label, value, hint }: { label: string; value: number | null; hint?: string }) {
  return (
    <div className="rounded-md border border-border bg-surface px-4 py-3 min-w-0">
      <span className="text-caption uppercase tracking-wider text-fg-subtle">{label}</span>
      <div className="font-mono tabular text-num-lg mt-1.5 text-fg truncate">
        {value == null ? <span className="text-body text-fg-subtle">not connected</span> : fmtInr(value)}
      </div>
      {hint && <p className="text-caption text-fg-subtle mt-0.5 truncate">{hint}</p>}
    </div>
  );
}

function PnlTile({ label, value, emptyLabel }: { label: string; value: number | null; emptyLabel: string }) {
  return (
    <div className="rounded-md border border-border bg-surface px-4 py-3 min-w-0">
      <span className="text-caption uppercase tracking-wider text-fg-subtle">{label}</span>
      <div className={cn("font-mono tabular text-num-lg mt-1.5 truncate", value == null ? "text-fg-subtle" : clsPnl(value))}>
        {value == null ? <span className="text-body">no trades yet</span> : fmtInr(value)}
      </div>
      {value == null && emptyLabel && <p className="text-caption text-fg-subtle mt-0.5 truncate" title={emptyLabel}>{emptyLabel}</p>}
    </div>
  );
}

function FreshnessBadge({ f }: { f: Freshness }) {
  if (!f.as_of) {
    return <span className="text-caption text-fg-subtle" title={f.note ?? f.source}>{f.source}</span>;
  }
  return (
    <Badge tone={f.stale ? "warning" : "success"} title={`${f.source}${f.note ? ` — ${f.note}` : ""}`}>
      {f.stale ? "stale" : "fresh"} · {fmtRel(f.as_of)}
    </Badge>
  );
}

type StatRow = [label: string, value: string, signed?: number | null];

function ComponentCard({ title, to, status, label, freshness, rows, extra }: {
  title: string; to: string; status: ComponentStatus; label: string; freshness: Freshness;
  rows: StatRow[]; extra?: string;
}) {
  return (
    <Card className="min-w-0" data-testid={`component-${title.toLowerCase().replace(" ", "-")}`}>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardTitle className="flex-1">
            <Link to={to} className="hover:underline">{title}</Link>
          </CardTitle>
          <Badge tone={STATUS_TONE[status]} dot>{STATUS_TEXT[status]}</Badge>
        </div>
        <CardDescription>{label}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        {status === "not_connected" ? (
          <p className="text-body-sm text-fg-muted">No data source yet — values are not shown as 0.</p>
        ) : (
          <dl className="grid grid-cols-2 gap-y-1 text-body-sm">
            {rows.map(([k, v, signed]) => (
              <React.Fragment key={k}>
                <dt className="text-fg-subtle">{k}</dt>
                <dd className={cn("text-right font-mono", signed != null && clsPnl(signed))}>{v}</dd>
              </React.Fragment>
            ))}
          </dl>
        )}
        {extra && <p className="text-caption text-warn">{extra}</p>}
        <div className="pt-1"><FreshnessBadge f={freshness} /></div>
      </CardContent>
    </Card>
  );
}

const TOOLTIP_STYLE = {
  background: "rgb(var(--surface))", border: "1px solid rgb(var(--border))", borderRadius: 6, fontSize: 12,
};

function EquityCurveCard({ data }: { data: PortfolioOverview }) {
  const h = data.history;
  const dd = h.max_drawdown;
  return (
    <Card className="min-w-0">
      <CardHeader>
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="flex-1">Equity curve</CardTitle>
          {dd && <span className="text-caption text-fg-subtle">
            Max drawdown <span className="font-mono text-pnl-down">{fmtInr(dd.abs)}</span> ({dd.pct.toFixed(2)}%, {dd.note})
          </span>}
          <FreshnessBadge f={h.freshness} />
        </div>
        <CardDescription>{h.basis ?? h.label}</CardDescription>
      </CardHeader>
      <CardContent>
        {h.status !== "ok" || h.points.length === 0 ? (
          <EmptyState title="No equity history yet" description={h.label ?? "Portfolio snapshots have not been recorded."} />
        ) : h.points.length < 2 ? (
          <p className="text-body-sm text-fg-muted">One day of history so far ({fmtInr(h.points[0].equity)}).</p>
        ) : (
          <div className="h-56" data-testid="portfolio-equity-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={h.points} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="rgb(var(--border))" vertical={false} strokeDasharray="3 3" />
                <XAxis dataKey="date" stroke="rgb(var(--fg-subtle))" fontSize={11} minTickGap={40} tickLine={false} />
                <YAxis stroke="rgb(var(--fg-subtle))" fontSize={11} width={72} domain={["auto", "auto"]}
                       tickFormatter={(v: number) => fmtInr(v, { compact: true })} />
                <ReTooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => fmtInr(v)} />
                <Line type="monotone" dataKey="equity" name="Equity" stroke="rgb(var(--accent))"
                      dot={false} strokeWidth={1.5} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function RiskCard({ data }: { data: PortfolioOverview }) {
  const r = data.risk;
  const pct = r.daily_loss.pct_used;
  return (
    <Card className="min-w-0">
      <CardHeader>
        <CardTitle>Risk</CardTitle>
        <CardDescription>{r.note || "Open risk is the sum of distances to each position's current stop."}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <div className="flex justify-between text-body-sm">
            <span className="text-fg-subtle">Daily loss budget</span>
            <span className="font-mono">
              {r.daily_loss.used == null ? "no trades today" : `${fmtInr(r.daily_loss.used)} / ${fmtInr(r.daily_loss.limit)}`}
            </span>
          </div>
          <div className="mt-1 h-2 rounded-full bg-surface-2 overflow-hidden" role="meter"
               aria-label="Daily loss budget used" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct ?? 0}>
            <div className={cn("h-full", (pct ?? 0) >= 80 ? "bg-pnl-down" : (pct ?? 0) >= 50 ? "bg-warn" : "bg-pnl-up")}
                 style={{ width: `${Math.min(100, pct ?? 0)}%` }} />
          </div>
          <p className="text-caption text-fg-subtle mt-0.5">
            limit {r.daily_loss.limit_pct_of_capital}% of capital · {r.daily_loss.basis}
          </p>
        </div>
        <dl className="grid grid-cols-2 gap-y-1 text-body-sm">
          <dt className="text-fg-subtle">Open risk</dt>
          <dd className="text-right font-mono">{r.open_risk == null ? "—" : fmtInr(r.open_risk)}
            {r.open_risk_pct_of_capital != null && <span className="text-fg-subtle"> ({r.open_risk_pct_of_capital}%)</span>}</dd>
          <dt className="text-fg-subtle">Margin used (book)</dt>
          <dd className="text-right font-mono">{fmtInr(r.margin_used_book)}</dd>
          <dt className="text-fg-subtle">Margin used (broker)</dt>
          <dd className="text-right font-mono">{r.margin_used_broker == null ? "not connected" : fmtInr(r.margin_used_broker)}</dd>
          <dt className="text-fg-subtle">Without a stop</dt>
          <dd className={cn("text-right font-mono", r.positions_without_stop > 0 && "text-warn")}>{r.positions_without_stop}</dd>
        </dl>
      </CardContent>
    </Card>
  );
}

function AllocationCard({ title, rows, empty }: {
  title: string; empty?: string;
  rows: { key: string; value: number | null; pct: number | null; note?: string }[];
}) {
  return (
    <Card className="min-w-0">
      <CardHeader><CardTitle>{title}</CardTitle></CardHeader>
      <CardContent>
        {rows.length === 0 ? (
          <p className="text-body-sm text-fg-muted">{empty ?? "Nothing to show."}</p>
        ) : (
          <ul className="space-y-2">
            {rows.map((r) => (
              <li key={r.key} className="text-body-sm">
                <div className="flex justify-between gap-2">
                  <span className="truncate text-fg">{r.key}</span>
                  <span className="font-mono text-fg-muted whitespace-nowrap">
                    {r.value == null ? (r.note ?? "—") : `${fmtInr(r.value)}${r.pct != null ? ` · ${r.pct}%` : ""}`}
                  </span>
                </div>
                {r.pct != null && (
                  <div className="mt-1 h-1.5 rounded-full bg-surface-2 overflow-hidden">
                    <div className="h-full bg-accent" style={{ width: `${Math.min(100, r.pct)}%` }} />
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function BrokerCard({ data }: { data: PortfolioOverview }) {
  const b = data.broker_reconciliation;
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardTitle className="flex-1">Held at broker (reconciliation)</CardTitle>
          <Badge tone={b.status === "ok" ? "success" : "warning"} dot>{b.label}</Badge>
        </div>
        <CardDescription>
          What the broker account reports, shown beside the book — it never replaces it.
          {b.discrepancy && <span className="ml-1 text-warn"><AlertTriangle className="inline h-3.5 w-3.5" aria-hidden /> {b.discrepancy}</span>}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <table className="w-full text-body-sm">
          <thead>
            <tr className="text-left text-caption uppercase tracking-wider text-fg-subtle border-b border-border">
              <th className="px-5 py-2 font-medium">Broker</th>
              <th className="px-2 py-2 font-medium">Link</th>
              <th className="px-2 py-2 font-medium">Positions</th>
              <th className="px-2 py-2 font-medium">Holdings</th>
              <th className="px-2 py-2 font-medium">Margin used</th>
              <th className="px-2 py-2 font-medium">Snapshot</th>
            </tr>
          </thead>
          <tbody>
            {b.links.length === 0 ? (
              <tr><td colSpan={6} className="px-5 py-3 text-fg-muted">
                No broker linked. <Link to="/brokers" className="text-accent hover:underline">Link one</Link>.
              </td></tr>
            ) : b.links.map((l) => (
              <tr key={l.broker} className="border-b border-border/60">
                <td className="px-5 py-2 font-mono">{l.broker}</td>
                <td className="px-2 py-2"><Badge tone={l.link_status === "active" ? "success" : "warning"}>{l.link_status}</Badge></td>
                <td className="px-2 py-2 font-mono">{l.snapshot ? l.snapshot.positions : "—"}</td>
                <td className="px-2 py-2 font-mono">{l.snapshot ? l.snapshot.holdings : "—"}</td>
                <td className="px-2 py-2 font-mono">{fmtInr(l.snapshot?.margin_used ?? null)}</td>
                <td className="px-2 py-2">{l.snapshot ? <FreshnessBadge f={l.snapshot.freshness} /> : "never"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

/* ================================================================== */
/* Attribution                                                         */
/* ================================================================== */

function AttributionTab() {
  const [months, setMonths] = React.useState(12);
  const q = usePortfolioAttribution(months);
  return (
    <div className="p-6 pt-2 space-y-6 max-w-[1400px]">
      <header className="flex flex-wrap items-end gap-3">
        <div className="flex-1 min-w-[240px]">
          <h2 className="text-h2 text-fg">Attribution</h2>
          <p className="text-body-sm text-fg-muted">Realized P&amp;L from closed trades, by strategy, symbol and month.</p>
        </div>
        <label className="flex items-center gap-2 text-caption text-fg-subtle">
          Window
          <select aria-label="Window" value={months} onChange={(e) => setMonths(Number(e.target.value))}
                  className="h-8 rounded-sm border border-border bg-surface px-2 text-body-sm text-fg">
            {[3, 6, 12, 24].map((m) => <option key={m} value={m}>{m} months</option>)}
          </select>
        </label>
      </header>
      {q.isLoading ? <Skeleton className="h-40 w-full" /> : q.isError || !q.data ? (
        <ErrorState title="Could not load attribution" onRetry={() => q.refetch()} />
      ) : q.data.status === "empty" ? (
        <EmptyState icon={<PieChart />} title="No closed trades yet"
                    description={`Nothing closed since ${q.data.since}. Attribution fills in as paper or live trades close.`} />
      ) : (
        <>
          <p className="text-body-sm text-fg-muted">
            {q.data.trade_count} closed trade(s) since {q.data.since} · total{" "}
            <span className={cn("font-mono", clsPnl(q.data.total_pnl))}>{fmtInr(q.data.total_pnl)}</span>
          </p>
          <div className="grid gap-4 xl:grid-cols-3">
            <GroupTable title="By strategy" rows={q.data.by_strategy} />
            <GroupTable title="By symbol" rows={q.data.by_symbol} />
            <GroupTable title="By month" rows={q.data.by_month} />
          </div>
        </>
      )}
    </div>
  );
}

function GroupTable({ title, rows }: { title: string; rows: AttributionGroup[] }) {
  return (
    <Card className="min-w-0">
      <CardHeader><CardTitle>{title}</CardTitle></CardHeader>
      <CardContent className="p-0">
        <table className="w-full text-body-sm">
          <thead>
            <tr className="text-left text-caption uppercase tracking-wider text-fg-subtle border-b border-border">
              <th className="px-5 py-2 font-medium">{title.replace("By ", "")}</th>
              <th className="px-2 py-2 font-medium text-right">P&amp;L</th>
              <th className="px-2 py-2 font-medium text-right">Trades</th>
              <th className="px-2 py-2 font-medium text-right">Win %</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} className="border-b border-border/60">
                <td className="px-5 py-2 truncate max-w-[180px]">{r.key}</td>
                <td className={cn("px-2 py-2 text-right font-mono", clsPnl(r.pnl))}>{fmtInr(r.pnl)}</td>
                <td className="px-2 py-2 text-right font-mono">{r.trades}</td>
                <td className="px-2 py-2 text-right font-mono">{r.win_rate == null ? "—" : `${r.win_rate}%`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
