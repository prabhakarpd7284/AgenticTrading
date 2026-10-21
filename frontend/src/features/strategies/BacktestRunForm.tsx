/**
 * Run form for a Studio backtest: period, universe, capital, slippage.
 * Starts the job (202) and hands the backtest id back; progress and the
 * report are rendered by BacktestReport.
 */
import * as React from "react";
import { Calendar, Play } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { apiErrors, useRunBacktest, useUniverse, type Spec, type StrategyRow } from "./useStrategies";

const LARGE_CAPS = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN", "BHARTIARTL", "LT", "ITC", "AXISBANK"];

function isoDaysBefore(iso: string, days: number) {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() - days);
  return d.toISOString().slice(0, 10);
}

export function BacktestRunForm({
  strategy, definition, onStarted,
}: {
  strategy: Pick<StrategyRow, "id" | "kind" | "current_version" | "backtest" | "backtest_reason">;
  definition?: Spec | Record<string, unknown> | null;
  onStarted: (backtestId: string) => void;
}) {
  const universe = useUniverse();
  const run = useRunBacktest();
  const specSyms = strategy.kind === "spec" ? ((definition as Spec | undefined)?.universe?.symbols ?? []) : [];
  const daily = strategy.kind === "spec" && (definition as Spec | undefined)?.timeframe === "1d";
  const last = (daily ? universe.data?.daily.last : universe.data?.intraday.last) ?? new Date().toISOString().slice(0, 10);
  const [from, setFrom] = React.useState<string | null>(null);
  const [to, setTo] = React.useState<string | null>(null);
  const [symbols, setSymbols] = React.useState<string>(specSyms.length ? "" : LARGE_CAPS.slice(0, 5).join(", "));
  const [capital, setCapital] = React.useState(500000);
  const [slippage, setSlippage] = React.useState<string>("");
  const fromV = from ?? isoDaysBefore(last, daily ? 150 : 30);
  const toV = to ?? last;

  if (strategy.backtest !== "backtest") {
    return <p className="text-body-sm text-fg-muted" data-testid="not-backtestable">{strategy.backtest_reason}</p>;
  }
  const err = run.isError ? apiErrors(run.error) : null;
  const list = symbols.split(/[\s,]+/).map((s) => s.trim().toUpperCase()).filter(Boolean);

  return (
    <form
      className="space-y-3"
      aria-label="Backtest settings"
      onSubmit={(e) => {
        e.preventDefault();
        run.mutate(
          {
            instance: strategy.id, from_date: fromV, to_date: toV, version: strategy.current_version,
            symbols: list.length ? list : null, capital,
            slippage_pct: slippage === "" ? null : Number(slippage),
          },
          { onSuccess: (bt) => onStarted(bt.id) },
        );
      }}
    >
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 items-end">
        <Input label="From" type="date" value={fromV} onChange={(e) => setFrom(e.target.value)}
               leading={<Calendar className="h-4 w-4" />} />
        <Input label="To" type="date" value={toV} onChange={(e) => setTo(e.target.value)}
               leading={<Calendar className="h-4 w-4" />} />
        <Input label="Capital ₹" type="number" min={10000} value={capital}
               onChange={(e) => setCapital(Number(e.target.value))} />
        <Input label="Slippage % / side" type="number" step="0.01" min={0} max={2} value={slippage}
               placeholder="strategy default" onChange={(e) => setSlippage(e.target.value)} />
      </div>
      <div>
        <label htmlFor="bt-symbols" className="text-body-sm text-fg mb-1 inline-block">
          Stocks {specSyms.length > 0 && <span className="text-fg-subtle">(empty = the strategy's own {specSyms.length})</span>}
        </label>
        <textarea id="bt-symbols" rows={2} value={symbols} onChange={(e) => setSymbols(e.target.value)}
                  placeholder={specSyms.join(", ") || "RELIANCE, HDFCBANK, …"}
                  className="w-full rounded-sm border border-border bg-surface px-3 py-2 text-body-sm text-fg font-mono focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60" />
        <div className="flex flex-wrap gap-1 mt-1">
          {specSyms.length > 0 && <Button type="button" size="sm" variant="ghost" onClick={() => setSymbols("")}>Strategy universe</Button>}
          <Button type="button" size="sm" variant="ghost" onClick={() => setSymbols(LARGE_CAPS.slice(0, 5).join(", "))}>5 large caps</Button>
          <Button type="button" size="sm" variant="ghost" onClick={() => setSymbols(LARGE_CAPS.join(", "))}>10 large caps</Button>
        </div>
        <p className="text-caption text-fg-subtle mt-1">
          {daily
            ? `Daily bars: NSE bhavcopy ${universe.data?.daily.first ?? ""} → ${universe.data?.daily.last ?? ""} (${universe.data?.daily.symbol_count ?? "?"} stocks, unadjusted).`
            : `1m bars: ${universe.data?.intraday.symbols.length ?? "?"} stocks with Angel One candles ${universe.data?.intraday.first ?? ""} → ${universe.data?.intraday.last ?? ""}. Start small (5 stocks × 1 month); runs go one at a time on the backtests queue.`}
          {" "}Dates {universe.data?.research_oos.from} → {universe.data?.research_oos.to} are the research out-of-sample window.
        </p>
      </div>
      {err && (
        <div role="alert" className="text-body-sm text-pnl-down">
          {err.errors.length ? err.errors.map((e) => <p key={e.path + e.message}>{e.path ? `${e.path}: ` : ""}{e.message}</p>) : err.detail}
        </div>
      )}
      <Button type="submit" loading={run.isPending} leading={<Play className="h-4 w-4" />}>
        Run backtest (v{strategy.current_version})
      </Button>
    </form>
  );
}
