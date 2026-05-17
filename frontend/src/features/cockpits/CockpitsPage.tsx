/**
 * Cockpits — 10 trader-facing aggregations on one page.
 *
 * Each tab fetches a single `/api/v1/portfolios/<slug>/` endpoint and
 * renders the response as KPIs + a table or chart. Everything is read-only.
 */
import * as React from "react";
import {
  Activity, AlertTriangle, BarChart3, Briefcase, Clock, Gauge, GitCompareArrows,
  LineChart, Sigma, Target, TrendingDown,
} from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart as RLineChart,
  ResponsiveContainer, Tooltip as ReTooltip, XAxis, YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import { cn, fmtInr, fmtNum, fmtPct } from "@/lib/utils";

type KPITone = "neutral" | "success" | "warning" | "danger";

function KPI({ label, value, hint, tone = "neutral" }: {
  label: string; value: string; hint?: string; tone?: KPITone;
}) {
  const color =
    tone === "success" ? "text-pnl-up"
    : tone === "warning" ? "text-warning"
    : tone === "danger" ? "text-pnl-down"
    : "text-fg";
  return (
    <div className="rounded-md border border-border bg-surface p-3">
      <div className="text-caption uppercase tracking-wider text-fg-subtle">{label}</div>
      <div className={cn("text-h3 font-mono mt-0.5", color)}>{value}</div>
      {hint ? <div className="text-caption text-fg-subtle mt-0.5">{hint}</div> : null}
    </div>
  );
}
import {
  useBrokerRecon, useCapitalCockpit, useEdgeDecay, useExpiryCockpit,
  useGreeksHeatmap, usePlanVsActual, useRegimeHeatmap, useRiskBudget,
  useSignalFunnel, useThetaForecast,
} from "@/lib/cockpits";

const TABS = [
  { id: "capital",       label: "Capital",      icon: Briefcase },
  { id: "plan-vs-actual",label: "Plan vs Actual", icon: GitCompareArrows },
  { id: "greeks",        label: "Greeks",       icon: Sigma },
  { id: "signal-funnel", label: "Signal Funnel",icon: Target },
  { id: "risk-budget",   label: "Risk Budget",  icon: Gauge },
  { id: "expiry",        label: "Expiry",       icon: Clock },
  { id: "broker-recon",  label: "Broker Recon", icon: GitCompareArrows },
  { id: "edge-decay",    label: "Edge Decay",   icon: TrendingDown },
  { id: "theta",         label: "Theta",        icon: LineChart },
  { id: "regime",        label: "Regime",       icon: Activity },
] as const;

export function CockpitsPage() {
  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      <header>
        <p className="text-caption uppercase tracking-wider text-fg-subtle">Trader cockpits</p>
        <h1 className="text-h1 text-fg flex items-center gap-2">
          <BarChart3 className="h-6 w-6 text-accent" aria-hidden />
          Cockpits
        </h1>
        <p className="text-body-sm text-fg-muted mt-1">
          Ten read-only aggregations across capital, plan vs actual, Greeks, risk,
          expiry, broker reconciliation, edge decay, theta burn, and market regime.
        </p>
      </header>

      <Tabs defaultValue="capital">
        <TabsList className="flex-wrap h-auto">
          {TABS.map(({ id, label, icon: Icon }) => (
            <TabsTrigger key={id} value={id} className="gap-1.5">
              <Icon className="h-3.5 w-3.5" aria-hidden /> {label}
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value="capital"><CapitalPanel /></TabsContent>
        <TabsContent value="plan-vs-actual"><PlanVsActualPanel /></TabsContent>
        <TabsContent value="greeks"><GreeksPanel /></TabsContent>
        <TabsContent value="signal-funnel"><SignalFunnelPanel /></TabsContent>
        <TabsContent value="risk-budget"><RiskBudgetPanel /></TabsContent>
        <TabsContent value="expiry"><ExpiryPanel /></TabsContent>
        <TabsContent value="broker-recon"><BrokerReconPanel /></TabsContent>
        <TabsContent value="edge-decay"><EdgeDecayPanel /></TabsContent>
        <TabsContent value="theta"><ThetaPanel /></TabsContent>
        <TabsContent value="regime"><RegimePanel /></TabsContent>
      </Tabs>
    </div>
  );
}

/* ------------------------------ helpers ----------------------------------- */
function LoadingPanel() {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
      {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-20" />)}
    </div>
  );
}

function PanelWrap({ children }: { children: React.ReactNode }) {
  return <div className="mt-4 space-y-6">{children}</div>;
}

/* ------------------------------ 1. Capital -------------------------------- */
function CapitalPanel() {
  const { data, isLoading } = useCapitalCockpit();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KPI label="Total capital" value={fmtInr(data.total_capital)} />
        <KPI label="Deployed" value={fmtInr(data.deployed_capital)} />
        <KPI label="Free margin" value={fmtInr(data.free_margin)} tone="success" />
        <KPI label="Leverage" value={`${fmtNum(data.leverage_ratio, 2)}×`}
             tone={data.leverage_ratio > 3 ? "warning" : "neutral"} />
        <KPI label="Margin used" value={fmtInr(data.margin_used)} />
        <KPI label="Notional" value={fmtInr(data.notional_exposure, { compact: true })} />
        <KPI label="Delta-adjusted" value={fmtInr(data.delta_adjusted_exposure, { compact: true })} />
        <KPI label="Premium received" value={fmtInr(data.premium_received)} />
      </section>
      <Card>
        <CardHeader>
          <CardTitle>Buckets</CardTitle>
          <CardDescription>Margin and notional split by instrument bucket.</CardDescription>
        </CardHeader>
        <CardContent>
          {data.buckets.length === 0 ? (
            <EmptyState title="No open positions" description="Deploy a strategy to populate." />
          ) : (
            <table className="w-full text-body-sm">
              <thead className="text-fg-subtle border-b border-border">
                <tr><th className="text-left py-1.5">Bucket</th><th className="text-right">Margin</th><th className="text-right">Notional</th></tr>
              </thead>
              <tbody>
                {data.buckets.map((b) => (
                  <tr key={b.name} className="border-b border-border/40">
                    <td className="py-1.5">{b.name}</td>
                    <td className="text-right tabular-nums">{fmtInr(b.margin)}</td>
                    <td className="text-right tabular-nums">{fmtInr(b.notional)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 2. Plan vs Actual ------------------------- */
function PlanVsActualPanel() {
  const { data, isLoading } = usePlanVsActual();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <section className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <KPI label="Trades analysed" value={String(data.count)} />
        <KPI label="Avg |slippage|" value={`${fmtNum(data.avg_abs_slippage_bps, 1)} bps`}
             tone={data.avg_abs_slippage_bps > 30 ? "warning" : "neutral"} />
        <KPI label="SL hits" value={String(data.rows.filter((r) => r.sl_hit).length)} />
      </section>
      <Card>
        <CardHeader><CardTitle>Slippage breakdown</CardTitle></CardHeader>
        <CardContent>
          {data.rows.length === 0 ? (
            <EmptyState title="No journal rows yet" />
          ) : (
            <div className="overflow-auto max-h-[480px]">
              <table className="w-full text-body-sm">
                <thead className="text-fg-subtle border-b border-border sticky top-0 bg-surface">
                  <tr>
                    <th className="text-left py-1.5">Symbol</th>
                    <th className="text-left">Side</th>
                    <th className="text-right">Planned</th>
                    <th className="text-right">Actual</th>
                    <th className="text-right">Slippage (bps)</th>
                    <th className="text-left">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.slice(0, 200).map((r) => (
                    <tr key={String(r.trade_id)} className="border-b border-border/40">
                      <td className="py-1.5">{r.symbol}</td>
                      <td>{r.side}</td>
                      <td className="text-right tabular-nums">{fmtNum(r.planned_entry, 2)}</td>
                      <td className="text-right tabular-nums">{fmtNum(r.actual_entry, 2)}</td>
                      <td className={`text-right tabular-nums ${Math.abs(r.slippage_bps) > 50 ? "text-warning" : ""}`}>
                        {fmtNum(r.slippage_bps, 1)}
                      </td>
                      <td>{r.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 3. Greeks --------------------------------- */
function GreeksPanel() {
  const { data, isLoading } = useGreeksHeatmap();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      {data.note ? (
        <Card><CardContent className="py-3 text-body-sm text-fg-muted">{data.note}</CardContent></Card>
      ) : null}
      <Card>
        <CardHeader>
          <CardTitle>Greeks heatmap</CardTitle>
          <CardDescription>Δ / Γ / Θ / V aggregated per underlying × expiry.</CardDescription>
        </CardHeader>
        <CardContent>
          {data.rows.length === 0 ? (
            <EmptyState title="No active option positions" />
          ) : (
            <table className="w-full text-body-sm">
              <thead className="text-fg-subtle border-b border-border">
                <tr>
                  <th className="text-left py-1.5">Underlying</th>
                  <th className="text-left">Expiry</th>
                  <th className="text-right">Δ</th>
                  <th className="text-right">Γ</th>
                  <th className="text-right">Θ</th>
                  <th className="text-right">V</th>
                  <th className="text-right">#Pos</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((r, i) => (
                  <tr key={`${r.underlying}-${r.expiry}-${i}`} className="border-b border-border/40">
                    <td className="py-1.5">{r.underlying}</td>
                    <td>{r.expiry}</td>
                    <td className="text-right tabular-nums">{fmtNum(r.delta, 2)}</td>
                    <td className="text-right tabular-nums">{fmtNum(r.gamma, 4)}</td>
                    <td className="text-right tabular-nums">{fmtNum(r.theta, 2)}</td>
                    <td className="text-right tabular-nums">{fmtNum(r.vega, 2)}</td>
                    <td className="text-right tabular-nums">{r.positions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 4. Signal Funnel -------------------------- */
function SignalFunnelPanel() {
  const { data, isLoading } = useSignalFunnel();
  if (isLoading || !data) return <LoadingPanel />;
  const t = data.totals;
  return (
    <PanelWrap>
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KPI label="Fired" value={String(t.fired)} />
        <KPI label="Risk passed" value={`${t.risk_passed} (${fmtPct(safeDiv(t.risk_passed, t.fired), 0)})`} />
        <KPI label="Executed" value={`${t.executed} (${fmtPct(safeDiv(t.executed, t.risk_passed), 0)})`} />
        <KPI label="Profitable" value={`${t.profitable} (${fmtPct(safeDiv(t.profitable, t.executed), 0)})`} tone="success" />
      </section>
      <div className="grid md:grid-cols-2 gap-4">
        <Card>
          <CardHeader><CardTitle>By strategy</CardTitle></CardHeader>
          <CardContent>
            {data.by_strategy.length === 0 ? <EmptyState title="No signals logged" /> : (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={data.by_strategy}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis dataKey="strategy" tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                  <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                  <ReTooltip />
                  <Bar dataKey="fired" fill="#64748b" />
                  <Bar dataKey="executed" fill="#22c55e" />
                  <Bar dataKey="profitable" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Rejection reasons</CardTitle></CardHeader>
          <CardContent>
            {data.rejection_reasons.length === 0 ? <EmptyState title="No rejections recorded" /> : (
              <ul className="space-y-1.5 text-body-sm">
                {data.rejection_reasons.map((r) => (
                  <li key={r.reason} className="flex items-center justify-between">
                    <span className="text-fg-muted">{r.reason}</span>
                    <Badge tone="danger">{r.count}</Badge>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>
    </PanelWrap>
  );
}

/* ------------------------------ 5. Risk Budget ---------------------------- */
function RiskBudgetPanel() {
  const { data, isLoading } = useRiskBudget();
  if (isLoading || !data) return <LoadingPanel />;
  const used = data.used_risk_pct;
  return (
    <PanelWrap>
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KPI label="Capital" value={fmtInr(data.capital)} />
        <KPI label="Daily P&amp;L" value={fmtInr(data.daily_pnl)}
             tone={data.daily_pnl >= 0 ? "success" : "danger"} />
        <KPI label="Used risk" value={fmtPct(used / 100, 2)}
             tone={used > 80 ? "danger" : used > 50 ? "warning" : "success"} />
        <KPI label="Open risk @ SL" value={fmtInr(data.open_risk_at_stop)} />
      </section>
      <Card>
        <CardHeader>
          <CardTitle>30-day drawdown</CardTitle>
          <CardDescription>Daily P&amp;L stacked into cumulative drawdown.</CardDescription>
        </CardHeader>
        <CardContent>
          {data.drawdown_waterfall.length === 0 ? <EmptyState title="No realised P&amp;L yet" /> : (
            <ResponsiveContainer width="100%" height={260}>
              <RLineChart data={data.drawdown_waterfall}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="date" tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                <ReTooltip />
                <Line type="monotone" dataKey="cum_pnl" stroke="#22c55e" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="drawdown" stroke="#ef4444" dot={false} strokeWidth={2} />
              </RLineChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ 6. Expiry --------------------------------- */
function ExpiryPanel() {
  const [underlying, setUnderlying] = React.useState("NIFTY");
  const { data, isLoading } = useExpiryCockpit(underlying);
  return (
    <PanelWrap>
      <div className="flex items-center gap-2">
        {["NIFTY", "BANKNIFTY", "SENSEX"].map((u) => (
          <button
            key={u}
            onClick={() => setUnderlying(u)}
            className={`px-3 h-8 text-body-sm rounded-sm border ${underlying === u ? "border-accent text-fg bg-accent/10" : "border-border text-fg-muted hover:bg-surface-2"}`}
          >
            {u}
          </button>
        ))}
      </div>
      {isLoading || !data ? <LoadingPanel /> : (
        <>
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <KPI label="Expiry day" value={data.is_expiry_day ? "YES" : "No"}
                 tone={data.is_expiry_day ? "warning" : "neutral"} />
            <KPI label="Countdown" value={fmtCountdown(data.countdown_seconds)} />
            <KPI label="Pin strike" value={data.pin_strike ? String(data.pin_strike) : "—"} />
            <KPI label="Active positions" value={String(data.active_count)} />
          </section>
          <Card>
            <CardHeader>
              <CardTitle>Close list ({data.close_list.length})</CardTitle>
              <CardDescription>
                {data.is_expiry_day
                  ? "Must close before 15:15 IST — gamma risk explodes after."
                  : "No expiry-day positions on this underlying."}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {data.close_list.length === 0 ? <EmptyState title="Nothing to close" /> : (
                <table className="w-full text-body-sm">
                  <thead className="text-fg-subtle border-b border-border">
                    <tr>
                      <th className="text-left py-1.5">Symbol</th>
                      <th className="text-left">CE</th>
                      <th className="text-left">PE</th>
                      <th className="text-right">P&amp;L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.close_list.map((row, i) => (
                      <tr key={i} className="border-b border-border/40">
                        <td className="py-1.5">{row.symbol}</td>
                        <td className="text-fg-muted">{row.ce_symbol ?? "—"}</td>
                        <td className="text-fg-muted">{row.pe_symbol ?? "—"}</td>
                        <td className="text-right tabular-nums">{fmtInr(row.pnl ?? 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </PanelWrap>
  );
}

/* ------------------------------ 7. Broker Recon --------------------------- */
function BrokerReconPanel() {
  const { data, isLoading } = useBrokerRecon();
  if (isLoading || !data) return <LoadingPanel />;
  return (
    <PanelWrap>
      <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KPI label="Broker rows" value={String(data.broker_count)} />
        <KPI label="Journal rows" value={String(data.journal_count)} />
        <KPI label="P&amp;L delta" value={fmtInr(data.pnl_delta)}
             tone={Math.abs(data.pnl_delta) < 1 ? "success" : "warning"} />
        <KPI label="Mismatched" value={String(data.mismatched_count)}
             tone={data.mismatched_count > 0 ? "warning" : "neutral"} />
      </section>
      {data.broker_note ? (
        <Card><CardContent className="py-3 text-body-sm text-fg-muted flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-warning" /> {data.broker_note}
        </CardContent></Card>
      ) : null}
      <div className="grid md:grid-cols-2 gap-4">
        <ReconList title="Only in broker" rows={data.only_in_broker} />
        <ReconList title="Only in journal" rows={data.only_in_journal} />
      </div>
    </PanelWrap>
  );
}

function ReconList({ title, rows }: { title: string; rows: { symbol: string; side?: string; qty?: number; pnl?: number }[] }) {
  return (
    <Card>
      <CardHeader><CardTitle>{title}</CardTitle></CardHeader>
      <CardContent>
        {rows.length === 0 ? <EmptyState title="No discrepancies" /> : (
          <ul className="space-y-1 text-body-sm">
            {rows.map((r, i) => (
              <li key={i} className="flex items-center justify-between">
                <span>{r.symbol} {r.side ? `· ${r.side}` : ""} {r.qty != null ? `· ${r.qty}` : ""}</span>
                {r.pnl != null ? <span className="tabular-nums">{fmtInr(r.pnl)}</span> : null}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

/* ------------------------------ 8. Edge Decay ----------------------------- */
function EdgeDecayPanel() {
  const { data, isLoading } = useEdgeDecay();
  if (isLoading || !data) return <LoadingPanel />;
  if (data.series.length === 0) return <EmptyState title="Not enough trades to compute edge decay" />;
  const palette = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];
  return (
    <PanelWrap>
      {data.series.map((s, idx) => (
        <Card key={s.strategy}>
          <CardHeader>
            <CardTitle>{s.strategy}</CardTitle>
            <CardDescription>Rolling {data.window}-trade expectancy.</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <RLineChart data={s.points}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="trade_idx" tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                <ReTooltip />
                <Line type="monotone" dataKey="expectancy" stroke={palette[idx % palette.length]} dot={false} strokeWidth={2} />
              </RLineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      ))}
    </PanelWrap>
  );
}

/* ------------------------------ 9. Theta Forecast ------------------------- */
function ThetaPanel() {
  const { data, isLoading } = useThetaForecast();
  if (isLoading || !data) return <LoadingPanel />;
  if (data.count === 0) return <EmptyState title="No active short-premium positions" />;
  return (
    <PanelWrap>
      {data.positions.map((p) => (
        <Card key={p.position_id}>
          <CardHeader>
            <CardTitle>{p.underlying} · {p.expiry}</CardTitle>
            <CardDescription>
              {fmtNum(p.minutes_to_expiry, 0)} min to expiry · Θ/min ≈ {fmtNum(p.theta_per_minute, 2)}
              · Premium remaining {fmtInr(p.premium_remaining)}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <RLineChart data={p.projection}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="minute" tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                <YAxis tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }} />
                <ReTooltip />
                <Line type="monotone" dataKey="premium" stroke="#3b82f6" dot={false} strokeWidth={2} />
              </RLineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      ))}
    </PanelWrap>
  );
}

/* ------------------------------ 10. Regime -------------------------------- */
function RegimePanel() {
  const { data, isLoading } = useRegimeHeatmap();
  if (isLoading || !data) return <LoadingPanel />;
  const cellMap = new Map<string, { expectancy: number; trades: number }>();
  for (const c of data.cells) cellMap.set(`${c.strategy}|${c.regime}`, c);
  return (
    <PanelWrap>
      <div className="flex items-center gap-2">
        <span className="text-body-sm text-fg-muted">Current regime:</span>
        <Badge tone="info">{data.current_regime || "unknown"}</Badge>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Strategy × regime expectancy</CardTitle>
          <CardDescription>Mean P&amp;L per trade per regime bucket. Green = positive edge.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-auto">
            <table className="w-full text-body-sm">
              <thead className="text-fg-subtle border-b border-border">
                <tr>
                  <th className="text-left py-1.5">Strategy</th>
                  {data.regimes.map((r) => <th key={r} className="text-right px-2">{r}</th>)}
                </tr>
              </thead>
              <tbody>
                {data.strategies.map((s) => (
                  <tr key={s} className="border-b border-border/40">
                    <td className="py-1.5 font-medium">{s}</td>
                    {data.regimes.map((r) => {
                      const cell = cellMap.get(`${s}|${r}`);
                      const e = cell?.expectancy ?? 0;
                      const n = cell?.trades ?? 0;
                      const bg = e > 0 ? `rgba(34,197,94,${Math.min(0.6, Math.abs(e) / 1000)})`
                              : e < 0 ? `rgba(239,68,68,${Math.min(0.6, Math.abs(e) / 1000)})`
                              : "transparent";
                      return (
                        <td key={r} className="text-right px-2 tabular-nums" style={{ background: bg }}>
                          {n > 0 ? <>{fmtInr(e)}<div className="text-caption text-fg-subtle">n={n}</div></> : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </PanelWrap>
  );
}

/* ------------------------------ misc -------------------------------------- */
function safeDiv(n: number, d: number): number {
  return d > 0 ? n / d : 0;
}
function fmtCountdown(seconds: number): string {
  if (seconds <= 0) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}
