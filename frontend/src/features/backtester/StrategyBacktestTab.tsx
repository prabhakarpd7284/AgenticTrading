/**
 * "Your strategies" on the Backtester page: pick any backtestable strategy,
 * run it through the Studio engine, read the report here. The Studio page
 * (/strategies/:id) has the same Test tab plus the definition and versions.
 */
import * as React from "react";
import { Link } from "react-router-dom";
import { ExternalLink } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { BacktestReportById } from "@/features/strategies/BacktestReport";
import { BacktestRunForm } from "@/features/strategies/BacktestRunForm";
import { BACKTEST_LABEL, useStrategies, useStrategy } from "@/features/strategies/useStrategies";

export function StrategyBacktestTab() {
  const { data: rows = [], isLoading } = useStrategies();
  const [id, setId] = React.useState<string>("");
  const [btId, setBtId] = React.useState<string | null>(null);
  const detail = useStrategy(id || undefined);
  const testable = rows.filter((r) => r.backtest === "backtest");
  const other = rows.filter((r) => r.backtest !== "backtest");

  if (!isLoading && rows.length === 0) {
    return (
      <EmptyState title="No strategies yet"
                  description="Create one on the Strategies page (from a template), then backtest it here." />
    );
  }
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Backtest one of your strategies</CardTitle>
          <CardDescription>
            Deterministic engine: signal on a bar's close, fill at the next open, live risk caps, NSE costs, last 30% of
            sessions held out. Results carry their data provenance.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label htmlFor="bt-strategy" className="text-body-sm text-fg mb-1.5 inline-block">Strategy</label>
            <select id="bt-strategy" value={id} onChange={(e) => { setId(e.target.value); setBtId(null); }}
                    className="h-9 w-full max-w-md rounded-sm border border-border bg-surface px-3 text-body-sm text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60">
              <option value="">Pick a strategy…</option>
              {testable.map((r) => <option key={r.id} value={r.id}>{r.name} (v{r.current_version})</option>)}
              {other.length > 0 && (
                <optgroup label="Not backtestable">
                  {other.map((r) => <option key={r.id} value={r.id} disabled>{r.name} — {BACKTEST_LABEL[r.backtest]}</option>)}
                </optgroup>
              )}
            </select>
          </div>
          {detail.data && (
            <>
              <BacktestRunForm strategy={detail.data} definition={detail.data.definition} onStarted={setBtId} />
              <Link to={`/strategies/${detail.data.id}?tab=test`}
                    className="inline-flex items-center gap-1 text-body-sm text-accent hover:underline">
                Open in Studio <ExternalLink className="h-3.5 w-3.5" />
              </Link>
            </>
          )}
        </CardContent>
      </Card>
      {btId && <BacktestReportById id={btId} />}
    </div>
  );
}
