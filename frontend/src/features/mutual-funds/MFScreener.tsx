/**
 * Screener tab on /mutual-funds (MF-S S9). Read-only view of the
 * deterministic within-category ranking computed by the backend
 * (apps.mutual_funds.services.screener). Personal use only.
 */
import * as React from "react";
import { AlertTriangle, Scale } from "lucide-react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip as ReTooltip, XAxis, YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { DataTable, type Column } from "@/components/ui/DataTable";
import { Skeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/ErrorState";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/Select";
import { cn, fmtNum } from "@/lib/utils";

import {
  COMPONENT_LABEL, FLAG_LABEL, useMFNavHistory, useMFScreener,
  type MFIndexCheck, type MFScreenRow, type ScoreComponent,
} from "./useMutualFunds";

const DEFAULT_CATEGORY = "Equity Scheme - Flexi Cap Fund";
const COMPONENTS: ScoreComponent[] = ["consistency", "risk_adjusted", "drawdown", "cost", "size_age"];
const TOOLTIP_STYLE = {
  background: "rgb(var(--surface))",
  border: "1px solid rgb(var(--border-strong))",
  borderRadius: 6,
  color: "rgb(var(--fg))",
  fontSize: 12,
};

/** Fraction -> "12.3%" (no forced sign; "n/a" when missing). */
export function pctOrNa(x: number | null | undefined, digits = 1): string {
  return x == null || !isFinite(x) ? "n/a" : `${(x * 100).toFixed(digits)}%`;
}

function numOrNa(x: number | null | undefined, digits = 0): string {
  return x == null || !isFinite(x) ? "n/a" : x.toFixed(digits);
}

type SortKey = keyof MFScreenRow | `c_${ScoreComponent}`;

function sortValue(r: MFScreenRow, key: string): number | string | null {
  if (key.startsWith("c_")) return r.components[key.slice(2) as ScoreComponent];
  const v = (r as unknown as Record<string, unknown>)[key];
  return typeof v === "number" || typeof v === "string" ? v : null;
}

export function MFScreener() {
  const [category, setCategory] = React.useState<string | null>(null);
  const [minYears, setMinYears] = React.useState(3);
  const [excludeIssues, setExcludeIssues] = React.useState(false);
  const [sort, setSort] = React.useState<{ key: string; dir: "asc" | "desc" }>({ key: "rank", dir: "asc" });
  const [selected, setSelected] = React.useState<string | null>(null);

  const q = useMFScreener({ category, minYears, excludeDataIssues: excludeIssues });
  const cats = q.data?.available_categories ?? [];

  // Pick a sensible default once the category list is known.
  React.useEffect(() => {
    if (category || cats.length === 0) return;
    setCategory(cats.find((c) => c.category === DEFAULT_CATEGORY)?.category ?? cats[0].category);
  }, [category, cats]);

  const result = q.data?.result ?? null;
  const rows = React.useMemo(() => {
    const list = [...(result?.rows ?? [])];
    list.sort((a, b) => {
      const va = sortValue(a, sort.key), vb = sortValue(b, sort.key);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;           // missing data always sinks
      if (vb == null) return -1;
      const c = va < vb ? -1 : va > vb ? 1 : 0;
      return sort.dir === "asc" ? c : -c;
    });
    return list;
  }, [result, sort]);
  const selectedRow = rows.find((r) => r.scheme_code === selected) ?? null;

  const onSort = (key: string) =>
    setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: key === "rank" ? "asc" : "desc" }));

  if (q.isError) {
    return (
      <ErrorState title="Couldn't load the screener" description="The screener endpoint failed."
                  onRetry={() => q.refetch()} />
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between gap-3 flex-wrap">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Scale className="h-4 w-4 text-accent" aria-hidden /> Fund screener
              </CardTitle>
              <CardDescription>
                Ranks Direct-Growth funds within one category by rolling-return consistency, risk-adjusted
                return, drawdown and age. Deterministic, personal use only — {q.data?.personal_use_note ?? ""}
              </CardDescription>
            </div>
            <div className="flex flex-col items-end gap-1" data-testid="screener-freshness">
              {q.isLoading ? <Skeleton className="h-6 w-40" /> : q.data?.metrics_as_of ? (
                <Badge tone="info" dot>Metrics as of {q.data.metrics_as_of}</Badge>
              ) : (
                <Badge tone="warning" dot>No metrics computed yet</Badge>
              )}
              <Badge tone="warning">TER / AUM data missing</Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-end gap-4 flex-wrap">
            <div className="min-w-[280px]">
              <span className="block text-caption text-fg-subtle mb-1" id="mf-cat-label">Category</span>
              <Select value={category ?? ""} onValueChange={(v) => { setCategory(v); setSelected(null); }}>
                <SelectTrigger aria-labelledby="mf-cat-label" placeholder="Pick a category" />
                <SelectContent>
                  {cats.map((c) => (
                    <SelectItem key={c.category} value={c.category}>
                      {c.category} ({c.count})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-1" role="group" aria-label="Minimum history">
              <span className="text-caption text-fg-subtle mr-1">Min history</span>
              {[1, 3, 5].map((y) => (
                <button key={y} type="button" onClick={() => setMinYears(y)} aria-pressed={minYears === y}
                        className={cn("rounded-full border px-2.5 py-0.5 text-caption",
                          minYears === y ? "border-accent bg-accent/15 text-accent" : "border-border text-fg-muted hover:text-fg")}>
                  {y}y+
                </button>
              ))}
            </div>
            <label className="flex items-center gap-2 text-body-sm text-fg-muted">
              <input type="checkbox" checked={excludeIssues} onChange={(e) => setExcludeIssues(e.target.checked)} />
              Exclude funds with data issues in the last 3y
            </label>
          </div>

          {result && (
            <div className="text-caption text-fg-subtle space-y-1" data-testid="screener-meta">
              <p>
                {result.eligible_count} of {result.universe_count} funds eligible
                {Object.keys(result.excluded).length > 0 && (
                  <> · excluded: {Object.entries(result.excluded).map(([k, v]) => `${v} (${k})`).join(", ")}</>
                )}
                {" "}· weights used:{" "}
                {Object.entries(result.weights_used).map(([k, v]) => `${COMPONENT_LABEL[k as ScoreComponent]} ${v}`).join(", ")}
              </p>
              {result.notes.map((n) => (
                <p key={n} className="flex items-center gap-1.5 text-warn">
                  <AlertTriangle className="h-3.5 w-3.5" aria-hidden /> {n}
                </p>
              ))}
            </div>
          )}

          <ScreenTable rows={rows} loading={q.isLoading || (!!category && !result && q.isFetching)}
                       sort={sort} onSort={onSort} onSelect={(r) => setSelected(r.scheme_code)} />
        </CardContent>
      </Card>

      {result?.index_check && <IndexCheckCard ic={result.index_check} />}
      {selectedRow && <FundBreakdown row={selectedRow} />}
    </div>
  );
}

function ScreenTable({
  rows, loading, sort, onSort, onSelect,
}: {
  rows: MFScreenRow[];
  loading?: boolean;
  sort: { key: string; dir: "asc" | "desc" };
  onSort: (k: string) => void;
  onSelect: (r: MFScreenRow) => void;
}) {
  const comp = (c: ScoreComponent, header: string): Column<MFScreenRow> => ({
    key: `c_${c}` as SortKey, header, kind: "num", sortable: true,
    render: (r) => (
      <span className={cn("font-mono tabular", r.components[c] == null && "text-fg-subtle")}>
        {numOrNa(r.components[c])}
      </span>
    ),
  });
  const columns: Column<MFScreenRow>[] = [
    { key: "rank", header: "#", kind: "num", sortable: true, render: (r) => <span className="font-mono tabular">{r.rank}</span> },
    {
      key: "scheme_name", header: "Fund", sortable: true,
      render: (r) => (
        <div className="py-1.5 min-w-[220px]">
          <div className="text-fg">{r.scheme_name}</div>
          <div className="text-caption text-fg-subtle flex items-center gap-1.5 flex-wrap">
            <span>{r.amc_name}</span>
            {(r.quality_flags ?? []).filter((f) => f !== "no_benchmark").map((f) => (
              <Badge key={f} tone="warning">{FLAG_LABEL[f] ?? f}</Badge>
            ))}
          </div>
        </div>
      ),
    },
    {
      key: "score", header: "Score", kind: "num", sortable: true,
      render: (r) => (
        <span className="font-mono tabular font-semibold"
              title={COMPONENTS.map((c) => `${COMPONENT_LABEL[c]}: ${numOrNa(r.components[c])}`).join("\n")}>
          {numOrNa(r.score)}{r.partial_score ? "*" : ""}
        </span>
      ),
    },
    comp("consistency", "Cons"),
    comp("risk_adjusted", "Risk"),
    comp("drawdown", "DD"),
    comp("size_age", "Age"),
    { key: "cagr_3y", header: "3y CAGR", kind: "num", sortable: true, render: (r) => <span className="font-mono tabular">{pctOrNa(r.cagr_3y)}</span> },
    { key: "rolling_3y_median", header: "Roll 3y med", kind: "num", sortable: true, render: (r) => <span className="font-mono tabular">{pctOrNa(r.rolling_3y_median)}</span> },
    { key: "max_drawdown", header: "Max DD", kind: "num", sortable: true, render: (r) => <span className="font-mono tabular">{pctOrNa(r.max_drawdown)}</span> },
    { key: "sortino", header: "Sortino", kind: "num", sortable: true, render: (r) => <span className="font-mono tabular">{numOrNa(r.sortino, 2)}</span> },
    { key: "expense_ratio_pct", header: "TER", kind: "num", render: (r) => <span className="font-mono tabular text-fg-subtle">{r.expense_ratio_pct == null ? "n/a" : `${r.expense_ratio_pct}%`}</span> },
  ];
  return (
    <DataTable
      columns={columns}
      rows={rows}
      loading={loading}
      rowKey={(r) => r.scheme_code}
      caption="Screener ranking"
      sort={sort}
      onSort={onSort}
      onRowClick={onSelect}
      emptyTitle="No funds to rank"
      emptyDescription="No metric snapshots for this category, or nothing passes the filters. Run compute_mf_metrics."
    />
  );
}

function IndexCheckCard({ ic }: { ic: MFIndexCheck }) {
  const tone = ic.verdict === "yes" ? "success" : ic.verdict === "no" ? "danger" : "warning";
  return (
    <Card>
      <CardHeader>
        <CardTitle>Active vs index — {ic.category}</CardTitle>
        <CardDescription>Rolling 3-year windows, median active fund (dead funds included) vs the passive alternative.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2" data-testid="index-check">
        <div className="flex items-center gap-2 flex-wrap">
          <Badge tone={tone}>{ic.verdict === "yes" ? "Active earned its fee" : ic.verdict === "no" ? "Index fund wins" : ic.verdict}</Badge>
          {ic.basis === "tri" && <Badge tone="warning">Gross TRI fallback</Badge>}
        </div>
        <p className="text-body-sm text-fg">{ic.text}</p>
        {ic.windows > 0 && (
          <p className="text-caption text-fg-subtle">
            Reference: {ic.reference}. Median active fund beat it in {ic.pct_windows_median_active_beat}% of {ic.windows} windows
            ({ic.first_window_end} – {ic.last_window_end}; last 3y: {ic.pct_windows_recent_3y ?? "n/a"}%), median gap{" "}
            {ic.median_gap_pp} pp/yr. Funds beating it in most of their own windows: {ic.funds_beating_reference_majority}.
          </p>
        )}
        {ic.alternative && (
          <p className="text-caption text-fg-subtle">
            Against the {ic.alternative.basis === "tri" ? "gross TRI" : "index-fund NAVs"} instead: median active fund won{" "}
            {ic.alternative.pct_windows_median_active_beat}% of {ic.alternative.windows} windows.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function FundBreakdown({ row }: { row: MFScreenRow }) {
  const hist = useMFNavHistory(row.scheme_code);
  const chart = React.useMemo(() => {
    if (!hist.data) return [];
    const bench = new Map((hist.data.benchmark?.points ?? []).map((p) => [p.date, p.value]));
    return hist.data.navs.map((n) => ({ date: n.date, nav: n.nav, bench: bench.get(n.date) ?? null }));
  }, [hist.data]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>{row.scheme_name}</CardTitle>
        <CardDescription>Why it ranks #{row.rank}: each component is a 0–100 percentile within the category.</CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-3">
          <ul className="space-y-2" aria-label="Score breakdown">
            {COMPONENTS.map((c) => {
              const v = row.components[c];
              return (
                <li key={c} className="text-body-sm">
                  <div className="flex justify-between">
                    <span className="text-fg-muted">{COMPONENT_LABEL[c]}</span>
                    <span className="font-mono tabular text-fg">{v == null ? "n/a" : fmtNum(v, 0)}</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-surface-2 overflow-hidden">
                    <div className="h-full bg-accent" style={{ width: `${v ?? 0}%` }} />
                  </div>
                </li>
              );
            })}
          </ul>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-body-sm">
            <dt className="text-fg-subtle">1y / 3y / 5y CAGR</dt>
            <dd className="font-mono tabular">{pctOrNa(row.cagr_1y)} / {pctOrNa(row.cagr_3y)} / {pctOrNa(row.cagr_5y)}</dd>
            <dt className="text-fg-subtle">Rolling 3y: median / worst</dt>
            <dd className="font-mono tabular">{pctOrNa(row.rolling_3y_median)} / {pctOrNa(row.rolling_3y_min)}</dd>
            <dt className="text-fg-subtle">Windows above category median</dt>
            <dd className="font-mono tabular">{numOrNa(row.rolling_3y_pct_above_cat_median)}% of {row.rolling_3y_windows}</dd>
            <dt className="text-fg-subtle">Max drawdown / worst 1y</dt>
            <dd className="font-mono tabular">{pctOrNa(row.max_drawdown)} / {pctOrNa(row.worst_1y)}</dd>
            <dt className="text-fg-subtle">Sharpe / Sortino</dt>
            <dd className="font-mono tabular">{numOrNa(row.sharpe, 2)} / {numOrNa(row.sortino, 2)}</dd>
            <dt className="text-fg-subtle">3y excess vs {row.benchmark_name || "benchmark"}</dt>
            <dd className="font-mono tabular">{pctOrNa(row.excess_cagr_3y_vs_benchmark)}</dd>
            <dt className="text-fg-subtle">History</dt>
            <dd className="font-mono tabular">{numOrNa(row.history_years, 1)}y (NAV to {row.last_nav_date ?? "—"})</dd>
          </dl>
          <p className="text-caption text-fg-subtle">Historical ranges only — not a forecast.</p>
        </div>
        <div>
          {hist.isLoading ? (
            <Skeleton className="h-56 w-full" />
          ) : chart.length < 2 ? (
            <p className="text-body-sm text-fg-subtle">Not enough NAV history to chart.</p>
          ) : (
            <div className="h-64" data-testid="screener-nav-chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chart} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="rgb(var(--border))" vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="date" stroke="rgb(var(--fg-subtle))" fontSize={11} minTickGap={40} tickLine={false} />
                  <YAxis stroke="rgb(var(--fg-subtle))" fontSize={11} width={56} domain={["auto", "auto"]} />
                  <ReTooltip contentStyle={TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="nav" name="NAV" stroke="rgb(var(--accent))" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                  {hist.data?.benchmark && (
                    <Line type="monotone" dataKey="bench" name={`${hist.data.benchmark.index_name} TRI (rebased)`}
                          stroke="rgb(var(--fg-subtle))" dot={false} strokeWidth={1} connectNulls isAnimationActive={false} />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
