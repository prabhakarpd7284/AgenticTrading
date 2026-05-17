/**
 * "Plan stock" — orchestrator UI.
 *
 * Button on the StockPage header → opens a dialog with capital + horizon
 * inputs → POSTs /api/v1/agents/plan-stock/ → renders consolidated card.
 *
 * Each strategy's plan shows side-by-side so the operator can compare:
 *   directional (equity), vertical_spread (defined-risk options),
 *   pyramid (intraday momentum).
 */
import * as React from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Bot, ChevronRight, Layers, Play, ShieldCheck, Triangle, Wallet } from "lucide-react";

import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { Portfolio } from "@/types";
import {
  Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger,
} from "@/components/ui/Dialog";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";

type Horizon = "intraday" | "swing" | "monthly";

type Allocation = { strategy: string; bucket: string; pct: number; capital: number };

type PlanResult = {
  strategy: string;
  bucket: string;
  capital: number;
  status: "succeeded" | "failed";
  summary: Record<string, unknown>;
};

type PlanResponse = {
  symbol: string;
  horizon: Horizon;
  side_hint: "BULL" | "BEAR" | null;
  total_capital: number;
  dry_run: boolean;
  allocations: Allocation[];
  results: PlanResult[];
  summary: { strategies_succeeded: number; strategies_failed: number; total_runtime_ms: number };
};

const STORAGE_KEY = (symbol: string) => `alphadesk:plan-stock:${symbol}`;

/** Read the last orchestrator result for a symbol from localStorage. */
export function getStoredPlan(symbol: string): { plan: PlanResponse; savedAt: string } | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY(symbol));
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function PlanStockButton({ symbol, onPlanned }: { symbol: string; onPlanned?: () => void }) {
  const [open, setOpen] = React.useState(false);
  const [plan, setPlan] = React.useState<PlanResponse | null>(null);

  const { data: portfolios = [] } = useQuery({
    queryKey: ["portfolios"],
    queryFn: () => api.get<Portfolio[]>("/portfolios/").then((r) => r.data),
  });

  const handleComplete = (r: PlanResponse) => {
    setPlan(r);
    try {
      localStorage.setItem(STORAGE_KEY(symbol), JSON.stringify({ plan: r, savedAt: new Date().toISOString() }));
    } catch { /* quota exceeded — silent */ }
    onPlanned?.();
  };

  return (
    <>
      <Dialog open={open} onOpenChange={(o) => { setOpen(o); if (!o) setPlan(null); }}>
        <DialogTrigger asChild>
          <Button leading={<Play className="h-4 w-4" />}>Plan stock</Button>
        </DialogTrigger>
        <DialogContent className="w-[min(95vw,720px)] max-h-[90vh] overflow-auto">
          <DialogTitle>Plan {symbol} across all strategies</DialogTitle>
          <DialogDescription>
            The orchestrator splits the capital across directional, vertical spread, and pyramid
            (and reserves the rest), fires each strategy in parallel, and shows a consolidated
            plan you can act on. Result is saved to the Plan report tab.
          </DialogDescription>
          {plan ? (
            <PlanResults plan={plan} onReset={() => setPlan(null)} />
          ) : (
            <PlanForm symbol={symbol} portfolios={portfolios} onComplete={handleComplete} />
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

/** Standalone display of a stored plan — used inside the Stock-page Plan tab. */
export function PlanReport({ plan, savedAt, onReplan }: { plan: PlanResponse; savedAt: string; onReplan?: () => void }) {
  return (
    <div className="space-y-3">
      <div className="text-caption text-fg-subtle font-mono">
        Saved {new Date(savedAt).toLocaleString("en-IN", { timeZone: "Asia/Kolkata", hour12: false })}
      </div>
      <PlanResults plan={plan} onReset={() => onReplan?.()} />
    </div>
  );
}

function PlanForm({
  symbol, portfolios, onComplete,
}: {
  symbol: string;
  portfolios: Portfolio[];
  onComplete: (r: PlanResponse) => void;
}) {
  const [capital, setCapital] = React.useState(500_000);
  const [horizon, setHorizon] = React.useState<Horizon>("monthly");
  const [sideHint, setSideHint] = React.useState<"BULL" | "BEAR">("BULL");
  const [portfolioId, setPortfolioId] = React.useState(portfolios[0]?.id ?? "");

  React.useEffect(() => {
    if (portfolios.length && !portfolioId) setPortfolioId(portfolios[0].id);
  }, [portfolios, portfolioId]);

  const planMut = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      api.post<PlanResponse>("/agents/plan-stock/", body).then((r) => r.data),
    onSuccess: (r) => onComplete(r),
  });

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (!portfolioId) return;
        planMut.mutate({
          symbol,
          portfolio: portfolioId,
          total_capital: capital,
          horizon,
          side_hint: sideHint,
          dry_run: true,
        });
      }}
      className="mt-4 space-y-4"
    >
      <Input
        label="Total capital for this stock"
        type="number"
        min={10000}
        step={10000}
        value={capital}
        onChange={(e) => setCapital(parseFloat(e.target.value) || 0)}
        hint="The orchestrator splits this across buckets per the horizon policy."
      />

      <div>
        <label className="text-body-sm text-fg mb-1.5 inline-block">Horizon</label>
        <div className="grid grid-cols-3 gap-1.5">
          {(["intraday", "swing", "monthly"] as Horizon[]).map((h) => (
            <button
              key={h}
              type="button"
              onClick={() => setHorizon(h)}
              className={cn(
                "rounded-sm border p-3 text-body-sm capitalize",
                horizon === h ? "border-accent/60 bg-accent/5 text-fg" : "border-border bg-surface hover:bg-surface-2 text-fg-muted",
              )}
            >
              <div className="font-medium">{h}</div>
              <div className="text-caption text-fg-subtle mt-0.5">{horizonHint(h)}</div>
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="text-body-sm text-fg mb-1.5 inline-block">Side hint</label>
        <div className="inline-flex rounded-sm border border-border overflow-hidden">
          {(["BULL", "BEAR"] as const).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setSideHint(s)}
              className={cn(
                "px-4 py-1.5 text-body-sm",
                sideHint === s
                  ? (s === "BULL" ? "bg-pnl-up/15 text-pnl-up" : "bg-pnl-down/15 text-pnl-down")
                  : "text-fg-muted hover:bg-surface-2",
              )}
            >
              {s}
            </button>
          ))}
        </div>
        <p className="text-caption text-fg-subtle mt-1">
          Pyramid uses CE for BULL / PE for BEAR. Vertical spread uses bull-call vs bear-put.
        </p>
      </div>

      {portfolios.length > 1 && (
        <div>
          <label className="text-body-sm text-fg mb-1.5 inline-block">Portfolio</label>
          <select
            value={portfolioId}
            onChange={(e) => setPortfolioId(e.target.value)}
            className="w-full bg-surface border border-border rounded-sm px-3 py-2 text-body-sm text-fg"
          >
            {portfolios.map((p) => (
              <option key={p.id} value={p.id}>{p.name} · ₹{p.capital}</option>
            ))}
          </select>
        </div>
      )}

      <div className="rounded-sm border border-border bg-surface-2/40 p-3 flex gap-2 items-start">
        <ShieldCheck className="h-4 w-4 text-accent mt-0.5" />
        <p className="text-caption text-fg-muted">
          This is a paper rehearsal — no orders are placed. Each strategy passes through @RiskGuard;
          their plans are returned together so you can choose what to fire live.
        </p>
      </div>

      {planMut.isError && (
        <p className="text-body-sm text-pnl-down">Plan failed: {(planMut.error as any)?.message ?? "unknown error"}</p>
      )}

      <div className="flex justify-end">
        <Button
          type="submit"
          loading={planMut.isPending}
          leading={<Layers className="h-4 w-4" />}
          disabled={!portfolioId || planMut.isPending}
        >
          {planMut.isPending ? "Planning across strategies… ~15s" : "Plan stock"}
        </Button>
      </div>
    </form>
  );
}

function PlanResults({ plan, onReset }: { plan: PlanResponse; onReset: () => void }) {
  return (
    <div className="mt-4 space-y-4">
      <header className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <div className="text-body-sm text-fg">
            <span className="font-mono">{plan.symbol}</span> · {plan.horizon} ·{" "}
            <Badge tone={plan.side_hint === "BULL" ? "success" : "danger"}>{plan.side_hint}</Badge>
          </div>
          <div className="text-caption text-fg-subtle font-mono">
            ₹{plan.total_capital.toLocaleString("en-IN")} total · {plan.summary.strategies_succeeded}/{plan.summary.strategies_succeeded + plan.summary.strategies_failed} succeeded ·
            {" "}{plan.summary.total_runtime_ms}ms
          </div>
        </div>
        <Button size="sm" variant="secondary" onClick={onReset}>Plan again</Button>
      </header>

      {/* Capital allocation bars */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Wallet className="h-4 w-4 text-accent" /> Capital allocation</CardTitle>
          <CardDescription>How the orchestrator split your capital across buckets.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {plan.allocations.map((a) => (
            <div key={a.strategy}>
              <div className="flex items-center justify-between text-body-sm">
                <span className="font-mono">{a.strategy} <span className="text-fg-subtle">· {a.bucket}</span></span>
                <span className="font-mono">₹{a.capital.toLocaleString("en-IN")} <span className="text-fg-subtle">({(a.pct * 100).toFixed(0)}%)</span></span>
              </div>
              <div className="h-1.5 bg-surface-2 rounded-full overflow-hidden mt-1">
                <div className={cn("h-full", bucketColor(a.bucket))} style={{ width: `${a.pct * 100}%` }} />
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Per-strategy result cards */}
      <div className="grid grid-cols-1 gap-3">
        {plan.results.map((r) => (
          <ResultCard key={r.strategy} result={r} />
        ))}
      </div>
    </div>
  );
}

function ResultCard({ result }: { result: PlanResult }) {
  const summary: any = result.summary ?? {};
  const failed = result.status === "failed";

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 flex-wrap">
          {result.strategy === "directional" ? <Bot className="h-4 w-4 text-accent" /> : <Triangle className="h-4 w-4 text-accent" />}
          <span className="font-mono">{result.strategy}</span>
          <Badge tone="info">{result.bucket}</Badge>
          <Badge tone="neutral">₹{result.capital.toLocaleString("en-IN")}</Badge>
          <Badge tone={failed ? "danger" : "success"}>{result.status}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {failed ? (
          <p className="text-body-sm text-pnl-down font-mono">{summary.error}</p>
        ) : (
          <StrategySummary strategy={result.strategy} summary={summary} />
        )}
      </CardContent>
    </Card>
  );
}

function StrategySummary({ strategy, summary }: { strategy: string; summary: any }) {
  if (strategy === "directional") {
    return (
      <div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
          <Kv label="Side" value={<Badge tone={summary.side === "BUY" ? "success" : "danger"}>{summary.side ?? "—"}</Badge>} />
          <Kv label="Qty" value={summary.quantity ?? "—"} />
          <Kv label="Entry" value={fmtNum(summary.entry)} />
          <Kv label="Conf" value={summary.confidence != null ? summary.confidence.toFixed(2) : "—"} />
          <Kv label="Stop" value={fmtNum(summary.stop_loss)} tone="danger" />
          <Kv label="Target" value={fmtNum(summary.target)} tone="success" />
          <Kv label="Risk" value={summary.risk_approved === true ? "approved" : summary.risk_approved === false ? "rejected" : "—"}
              tone={summary.risk_approved ? "success" : summary.risk_approved === false ? "danger" : "neutral"} />
        </div>
        {summary.reasoning && (
          <div className="mb-2">
            <div className="text-caption uppercase tracking-wider text-fg-subtle mb-0.5">Reasoning</div>
            <p className="text-body-sm text-fg-muted whitespace-pre-wrap">{summary.reasoning}</p>
          </div>
        )}
        {summary.risk_approved === false && (
          <div className="rounded-sm border border-danger/40 bg-pnl-down/5 p-2 text-body-sm text-fg-muted">
            <strong className="text-fg">RiskGuard blocked:</strong> {summary.risk_reason}
          </div>
        )}
        {summary.indicators && (
          <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-1 text-caption font-mono">
            {Object.entries(summary.indicators).slice(0, 8).map(([k, v]) => (
              <div key={k} className="rounded-sm border border-border bg-surface-2/40 px-2 py-1">
                <span className="text-fg-subtle">{k}</span> <span className="text-fg ml-1">{String(v ?? "—")}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (strategy === "vertical_spread") {
    return (
      <div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
          <Kv label="Structure" value={summary.side ? `${summary.side} ${summary.option_type}` : "—"} />
          <Kv label="Strikes" value={summary.long_strike ? `${summary.long_strike}/${summary.short_strike}` : "—"} />
          <Kv label="Expiry" value={summary.expiry} />
          <Kv label="Spot" value={fmtNum(summary.spot)} />
          <Kv label="Long LTP" value={fmtNum(summary.long_ltp)} />
          <Kv label="Short LTP" value={fmtNum(summary.short_ltp)} />
          <Kv label="Net debit" value={fmtNum(summary.net_debit)} />
          <Kv label="Lots" value={summary.lots ?? "—"} />
          <Kv label="Max profit" value={fmtINR(summary.max_profit_inr)} tone="success" />
          <Kv label="Max loss" value={fmtINR(summary.max_loss_inr)} tone="danger" />
          <Kv label="R:R" value={summary.rr_ratio != null ? `${summary.rr_ratio}×` : "—"} />
          <Kv label="Breakeven" value={fmtNum(summary.breakeven)} />
        </div>
        {summary.error && (
          <div className="rounded-sm border border-warning/40 bg-warning/5 p-2.5 text-body-sm text-fg-muted">
            <strong className="text-fg">Couldn't size the spread:</strong> {summary.error}
          </div>
        )}
      </div>
    );
  }

  if (strategy === "pyramid") {
    return (
      <div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
          <Kv label="Symbol" value={summary.symbol ?? "—"} />
          <Kv label="Strike" value={summary.strike != null ? `${summary.strike}` : "—"} />
          <Kv label="Expiry" value={summary.expiry ?? "—"} />
          <Kv label="Spot" value={fmtNum(summary.spot)} />
          <Kv label="Candles" value={summary.candles ?? 0} />
          <Kv label="Entries" value={summary.entries ?? 0} />
          <Kv label="Peak lots" value={summary.peak_lots ?? "—"} />
          <Kv label="Total lots" value={summary.total_lots ?? "—"} />
          <Kv label="Avg entry" value={fmtNum(summary.avg_entry)} />
          <Kv label="Exit price" value={fmtNum(summary.exit_price)} />
          <Kv label="P&L" value={fmtINR(summary.pnl_inr)} tone={(summary.pnl_inr ?? 0) >= 0 ? "success" : "danger"} />
          <Kv label="Exit reason" value={summary.exit_reason ?? "—"} />
        </div>
        {summary.first_entry && (
          <div className="text-caption font-mono text-fg-muted mb-2">
            First entry @ {fmtNum(summary.first_entry.price)} at {summary.first_entry.timestamp}
            <span className="text-fg-subtle"> · {summary.first_entry.reason}</span>
          </div>
        )}
        {summary.error && (
          <div className="rounded-sm border border-warning/40 bg-warning/5 p-2.5 text-body-sm text-fg-muted">
            <strong className="text-fg">No simulation run:</strong> {summary.error}
          </div>
        )}
      </div>
    );
  }
  return null;
}

function Kv({ label, value, tone = "neutral" }: { label: string; value: React.ReactNode; tone?: "neutral" | "success" | "danger" | "warning" | "info" }) {
  const color = tone === "success" ? "text-pnl-up" : tone === "danger" ? "text-pnl-down" : tone === "warning" ? "text-warning" : tone === "info" ? "text-info" : "text-fg";
  return (
    <div className="rounded-sm border border-border bg-surface-2/40 px-2 py-1.5">
      <div className="text-caption uppercase tracking-wider text-fg-subtle">{label}</div>
      <div className={cn("text-body-sm font-mono mt-0.5", color)}>{value ?? "—"}</div>
    </div>
  );
}

function horizonHint(h: Horizon): string {
  if (h === "intraday") return "MIS · today-only";
  if (h === "swing") return "2-10 sessions";
  return "expiry-to-expiry";
}

function bucketColor(bucket: string): string {
  if (bucket.startsWith("equity")) return "bg-info";
  if (bucket.includes("monthly") || bucket.includes("swing")) return "bg-accent";
  if (bucket.includes("intraday")) return "bg-warning";
  if (bucket === "reserve") return "bg-fg-subtle";
  return "bg-brand";
}

function fmtNum(n?: number | null): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

function fmtINR(n?: number | null): string {
  if (n == null || !Number.isFinite(n)) return "—";
  const v = Math.round(n);
  return `${v < 0 ? "-" : ""}₹${Math.abs(v).toLocaleString("en-IN")}`;
}
