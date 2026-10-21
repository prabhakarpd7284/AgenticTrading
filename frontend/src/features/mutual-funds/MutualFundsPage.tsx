import * as React from "react";
import { toast } from "sonner";
import { FileUp, PieChart, Play, Search, Sparkles } from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer,
  Tooltip as ReTooltip, XAxis, YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { KPI } from "@/components/ui/KPI";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { DataTable, type Column } from "@/components/ui/DataTable";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorState } from "@/components/ui/ErrorState";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import { cn, clsPnl, fmtDateTime, fmtInr, fmtNum, fmtPct } from "@/lib/utils";

import {
  CAS_IMPORT_COMMAND, FINDING_KIND_LABEL,
  useMFFindings, useMFHoldings, useMFSchemeDetail, useMFSchemeSearch, useMFStatus,
  useRunMFAdvisor,
  type MFAllocationRow, type MFFinding, type MFHoldingRow, type MFStatus,
} from "./useMutualFunds";
import { MFScreener } from "./MFScreener";

const TOOLTIP_STYLE = {
  background: "rgb(var(--surface))",
  border: "1px solid rgb(var(--border-strong))",
  borderRadius: 6,
  color: "rgb(var(--fg))",
  fontSize: 12,
};

export function MutualFundsPage() {
  const [includeClosed, setIncludeClosed] = React.useState(false);
  const holdings = useMFHoldings(includeClosed);
  const status = useMFStatus();
  const findings = useMFFindings();

  const summary = holdings.data?.summary;
  const rows = holdings.data?.results ?? [];
  const empty = !holdings.isLoading && !holdings.isError && rows.length === 0 && !includeClosed;

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <p className="text-caption uppercase tracking-wider text-fg-subtle">Personal portfolio</p>
          <h1 className="text-h1 text-fg">
            Mutual <span className="text-accent">Funds</span>
          </h1>
          <p className="text-body-sm text-fg-muted mt-1">
            Holdings from CAS imports, valued at the latest AMFI NAV.
          </p>
        </div>
        <NavFreshnessBadge status={status.data} loading={status.isLoading} />
      </header>

      <Tabs defaultValue="portfolio">
        <TabsList>
          <TabsTrigger value="portfolio">Portfolio</TabsTrigger>
          <TabsTrigger value="screener">Screener</TabsTrigger>
        </TabsList>
        <TabsContent value="screener" className="pt-4">
          <MFScreener />
        </TabsContent>
        <TabsContent value="portfolio" className="pt-4 space-y-6">
      {holdings.isError ? (
        <ErrorState
          title="Couldn't load mutual fund holdings"
          description="The holdings endpoint failed. Retry, or check that the backend is reachable."
          onRetry={() => holdings.refetch()}
        />
      ) : empty ? (
        <CasEmptyState />
      ) : (
        <>
          <section aria-label="Summary" className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <KPI label="Invested" value={summary?.invested} valueFormat="inr" loading={holdings.isLoading}
                 hint={summary?.missing_cost_count ? `${summary.missing_cost_count} holding(s) without cost basis` : undefined} />
            <KPI label="Current value" value={summary?.current_value} valueFormat="inr" loading={holdings.isLoading}
                 hint={summary?.unvalued_count ? `${summary.unvalued_count} holding(s) without a NAV` : undefined} />
            <KPI label="Gain" value={summary?.gain} valueFormat="inr" loading={holdings.isLoading}
                 delta={summary?.gain_pct ?? undefined} deltaFormat="pct" />
            <KPI label="Holdings" value={summary?.holding_count} loading={holdings.isLoading} />
          </section>

          <div className="grid grid-cols-1 xl:grid-cols-[2fr_1fr] gap-6">
            <Card>
              <CardHeader>
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <CardTitle>Holdings</CardTitle>
                    <CardDescription>One row per folio. Closed (zero-unit) folios are hidden by default.</CardDescription>
                  </div>
                  <label className="flex items-center gap-2 text-body-sm text-fg-muted whitespace-nowrap">
                    <input type="checkbox" checked={includeClosed}
                           onChange={(e) => setIncludeClosed(e.target.checked)} />
                    Show closed
                  </label>
                </div>
              </CardHeader>
              <CardContent>
                <HoldingsTable rows={rows} loading={holdings.isLoading} />
              </CardContent>
            </Card>
            <AllocationCard rows={summary?.allocation ?? []} loading={holdings.isLoading} />
          </div>
        </>
      )}

      <FindingsCard data={findings.data} loading={findings.isLoading} error={findings.isError} />
      <SchemeExplorer />
        </TabsContent>
      </Tabs>
    </div>
  );
}

/* ================================================================== */
/* NAV freshness                                                       */
/* ================================================================== */
function NavFreshnessBadge({ status, loading }: { status?: MFStatus; loading?: boolean }) {
  if (loading) return <Skeleton className="h-6 w-48" />;
  if (!status) return null;
  const stale = status.catalog_is_stale || status.stale_holdings_count > 0;
  const label = status.latest_nav_date ? `NAV as of ${status.latest_nav_date}` : "No NAV data";
  const detail = [
    status.last_pull_at ? `Last AMFI pull ${fmtDateTime(status.last_pull_at)} IST` : "AMFI pull never ran",
    status.catalog_is_stale
      ? `catalog older than ${status.stale_after_trading_days} trading days`
      : null,
    status.stale_holdings_count > 0 ? `${status.stale_holdings_count} held scheme(s) stale` : null,
  ].filter(Boolean).join(" · ");
  return (
    <div className="flex flex-col items-end gap-1" data-testid="nav-freshness">
      <Badge tone={stale ? "warning" : "success"} dot>{label}</Badge>
      <span className="text-caption text-fg-subtle">{detail}</span>
    </div>
  );
}

/* ================================================================== */
/* Empty state                                                         */
/* ================================================================== */
function CasEmptyState() {
  return (
    <EmptyState
      icon={<FileUp />}
      title="No mutual fund holdings yet"
      description={
        "Holdings come from a CAMS/KFintech Consolidated Account Statement (CAS) PDF. " +
        "Import one from the backend (dry run first, then drop --dry-run):"
      }
      action={
        <pre className="max-w-full overflow-x-auto rounded-sm border border-border bg-surface-2 px-3 py-2 text-left text-caption font-mono text-fg">
          {CAS_IMPORT_COMMAND}
        </pre>
      }
    />
  );
}

/* ================================================================== */
/* Holdings table                                                      */
/* ================================================================== */
function HoldingsTable({ rows, loading }: { rows: MFHoldingRow[]; loading?: boolean }) {
  const columns: Column<MFHoldingRow>[] = [
    {
      key: "scheme_name", header: "Scheme",
      render: (r) => (
        <div className="py-1.5 min-w-[220px]">
          <div className="text-fg">{r.scheme_name}</div>
          <div className="text-caption text-fg-subtle flex items-center gap-1.5 flex-wrap">
            <span>{r.amc_name}</span>
            <span aria-hidden>·</span>
            <span>folio {r.folio_number || "—"}</span>
            {!r.is_direct && <Badge tone="warning">Regular</Badge>}
          </div>
        </div>
      ),
    },
    { key: "units", header: "Units", kind: "num", render: (r) => <span className="font-mono tabular">{fmtNum(r.units, 3)}</span> },
    {
      key: "latest_nav", header: "NAV", kind: "num",
      render: (r) => (
        <span className="font-mono tabular">
          {r.latest_nav == null ? "—" : fmtNum(r.latest_nav, 4)}
          <span className={cn("block text-caption", r.nav_is_stale ? "text-warn" : "text-fg-subtle")}>
            {r.nav_date ?? "no NAV"}{r.nav_is_stale && r.nav_date ? " · stale" : ""}
          </span>
        </span>
      ),
    },
    { key: "invested", header: "Invested", kind: "money", render: (r) => <span className="font-mono tabular">{fmtInr(r.invested)}</span> },
    { key: "current_value", header: "Value", kind: "money", render: (r) => <span className="font-mono tabular">{fmtInr(r.current_value)}</span> },
    {
      key: "gain", header: "Gain", kind: "delta",
      render: (r) => (
        <span className={cn("font-mono tabular", clsPnl(r.gain))}>
          {fmtInr(r.gain)}
          <span className="block text-caption">{fmtPct(r.gain_pct)}</span>
        </span>
      ),
    },
  ];
  return (
    <DataTable
      columns={columns}
      rows={rows}
      loading={loading}
      rowKey={(r) => String(r.id)}
      caption="Mutual fund holdings"
      emptyTitle="No holdings"
      emptyDescription="No folios match — closed folios are hidden unless 'Show closed' is on."
    />
  );
}

/* ================================================================== */
/* Allocation by category                                              */
/* ================================================================== */
function AllocationCard({ rows, loading }: { rows: MFAllocationRow[]; loading?: boolean }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle>Allocation by category</CardTitle>
            <CardDescription>Share of current value per AMFI category.</CardDescription>
          </div>
          <PieChart className="h-5 w-5 text-fg-subtle shrink-0" aria-hidden />
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-48 w-full" />
        ) : rows.length === 0 ? (
          <p className="text-body-sm text-fg-subtle">No valued holdings to allocate.</p>
        ) : (
          <>
            <div style={{ height: Math.max(120, rows.length * 34) }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="rgb(var(--border))" horizontal={false} strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} stroke="rgb(var(--fg-subtle))" fontSize={11}
                         tickFormatter={(v: number) => `${v}%`} tickLine={false} />
                  <YAxis type="category" dataKey="category" hide />
                  <ReTooltip
                    contentStyle={TOOLTIP_STYLE}
                    cursor={{ fill: "rgb(var(--surface-2))" }}
                    formatter={(v: number, _n, item) => [`${v.toFixed(1)}% · ${fmtInr((item?.payload as MFAllocationRow)?.value)}`, "Share"]}
                  />
                  <Bar dataKey="pct" fill="rgb(var(--accent))" radius={[0, 2, 2, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <ul className="mt-3 space-y-1" aria-label="Allocation by category">
              {rows.map((r) => (
                <li key={r.category} className="flex justify-between gap-3 text-body-sm">
                  <span className="text-fg-muted truncate">{r.category}</span>
                  <span className="font-mono tabular text-fg">{r.pct.toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </>
        )}
      </CardContent>
    </Card>
  );
}

/* ================================================================== */
/* Advisor findings                                                    */
/* ================================================================== */
function FindingsCard({
  data, loading, error,
}: {
  data?: { latest_run_at: string | null; results: MFFinding[] };
  loading?: boolean;
  error?: boolean;
}) {
  const run = useRunMFAdvisor();
  const onRun = () =>
    run.mutate(undefined, {
      onSuccess: (r) => toast.success(`Advisor ran — ${r.count} finding(s)`),
      onError: () => toast.error("Advisor run failed (owner role required)"),
    });

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-accent" aria-hidden /> Advisor findings
            </CardTitle>
            <CardDescription>
              Deterministic rule checks, personal use only — not investment advice.
              {data?.latest_run_at && <> Latest run {fmtDateTime(data.latest_run_at)} IST.</>}
            </CardDescription>
          </div>
          <Button variant="secondary" size="sm" onClick={onRun} loading={run.isPending}
                  leading={<Play className="h-4 w-4" />}>
            Run advisor
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-24 w-full" />
        ) : error ? (
          <p className="text-body-sm text-danger">Couldn't load advisor findings.</p>
        ) : !data || data.results.length === 0 ? (
          <p className="text-body-sm text-fg-subtle">
            No advisor findings yet. They appear after the advisor runs over imported holdings
            (<code className="font-mono">manage.py run_mf_advisor</code> or “Run advisor”).
          </p>
        ) : (
          <ul className="space-y-2" aria-label="Advisor findings">
            {data.results.map((f, i) => (
              <li key={f.id ?? i} className="rounded-sm border border-border px-3 py-2">
                <div className="flex items-center gap-2 flex-wrap mb-1">
                  <Badge tone={f.severity === "warn" ? "warning" : "info"}>
                    {FINDING_KIND_LABEL[f.kind] ?? f.kind}
                  </Badge>
                  {f.is_estimate && <Badge tone="neutral">Estimate</Badge>}
                </div>
                <p className="text-body-sm text-fg">{f.text}</p>
                {f.kind === "direct_vs_regular" && (
                  <p className="text-caption text-fg-subtle mt-1">
                    The drag is an ESTIMATE from an assumed regular-vs-direct spread — AMFI data has no
                    expense ratios. Verify against the fund's factsheet before acting.
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

/* ================================================================== */
/* Scheme search + NAV history                                         */
/* ================================================================== */
function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = React.useState(value);
  React.useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return v;
}

function SchemeExplorer() {
  const [q, setQ] = React.useState("");
  const [selected, setSelected] = React.useState<string | null>(null);
  const debounced = useDebounced(q, 300);
  const search = useMFSchemeSearch(debounced);
  const detail = useMFSchemeDetail(selected);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Scheme search</CardTitle>
        <CardDescription>Search the AMFI catalog by scheme name, AMC or scheme code.</CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-3">
          <Input
            label="Search schemes"
            labelSrOnly
            placeholder="e.g. parag parikh flexi direct"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            leading={<Search />}
          />
          {debounced.trim().length < 2 ? (
            <p className="text-caption text-fg-subtle">Type at least 2 characters.</p>
          ) : search.isLoading ? (
            <Skeleton className="h-32 w-full" />
          ) : (search.data?.results.length ?? 0) === 0 ? (
            <p className="text-body-sm text-fg-subtle">No schemes match.</p>
          ) : (
            <>
              <p className="text-caption text-fg-subtle">
                {search.data!.count} match(es){search.data!.count > search.data!.results.length ? ", showing first 20" : ""}
              </p>
              <ul className="max-h-80 overflow-auto divide-y divide-border/60 rounded-sm border border-border" aria-label="Scheme results">
                {search.data!.results.map((s) => (
                  <li key={s.scheme_code}>
                    <button
                      type="button"
                      onClick={() => setSelected(s.scheme_code)}
                      className={cn(
                        "w-full text-left px-3 py-2 hover:bg-surface-2 focus-visible:outline-none focus-visible:bg-surface-2",
                        selected === s.scheme_code && "bg-accent/10",
                      )}
                    >
                      <div className="text-body-sm text-fg">{s.scheme_name}</div>
                      <div className="text-caption text-fg-subtle">
                        {s.scheme_code} · {s.amc_name} · NAV {s.latest_nav ?? "—"} ({s.latest_nav_date ?? "—"})
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>

        <div>
          {!selected ? (
            <p className="text-body-sm text-fg-subtle">Select a scheme to see its NAV history.</p>
          ) : detail.isLoading ? (
            <Skeleton className="h-56 w-full" />
          ) : detail.isError || !detail.data ? (
            <p className="text-body-sm text-danger">Couldn't load scheme detail.</p>
          ) : (
            <div className="space-y-2">
              <div>
                <div className="text-body text-fg font-semibold">{detail.data.scheme.scheme_name}</div>
                <div className="text-caption text-fg-subtle">
                  {detail.data.scheme.category_label} · {detail.data.scheme.plan || "—"}
                  {detail.data.held_units > 0 && <> · you hold {fmtNum(detail.data.held_units, 3)} units</>}
                </div>
              </div>
              {detail.data.navs.length < 2 ? (
                <p className="text-body-sm text-fg-subtle">
                  Only {detail.data.navs.length} NAV point(s) stored — run{" "}
                  <code className="font-mono">backfill_amfi_history --scheme {detail.data.scheme.scheme_code}</code>{" "}
                  for history.
                </p>
              ) : (
                <div className="h-56" data-testid="nav-history-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={detail.data.navs} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke="rgb(var(--border))" vertical={false} strokeDasharray="3 3" />
                      <XAxis dataKey="date" stroke="rgb(var(--fg-subtle))" fontSize={11} minTickGap={32} tickLine={false} />
                      <YAxis stroke="rgb(var(--fg-subtle))" fontSize={11} width={56} domain={["auto", "auto"]} />
                      <ReTooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [fmtNum(v, 4), "NAV"]} />
                      <Line type="monotone" dataKey="nav" stroke="rgb(var(--accent))" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
