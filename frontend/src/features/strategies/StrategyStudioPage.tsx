/**
 * Strategy Studio — /strategies/:id (Studio plan §8, P3).
 *
 *   header: name · stage · version · "what's next"
 *   Define    rule builder (spec) or schema form (plugin) + plain-English
 *             summary + live validation; save = new version. Instant
 *             signal preview on one stock.
 *   Test      run form → progress → report (provenance, metrics, IS/OOS,
 *             equity, trades); past backtests of every version.
 *   Versions  history with what changed.
 * Agent Sim / Paper / Promote are later Studio phases and say so.
 */
import * as React from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { AlertTriangle, ArrowLeft, ArrowRight, CheckCircle2, Eye, Info, Pencil, Save } from "lucide-react";
import { CartesianGrid, ComposedChart, Line, ResponsiveContainer, Scatter, Tooltip as ReTooltip, XAxis, YAxis } from "recharts";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { DataTable, type Column } from "@/components/ui/DataTable";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import { cn, fmtRel } from "@/lib/utils";

import { BacktestLine, BacktestReportById } from "./BacktestReport";
import { BacktestRunForm } from "./BacktestRunForm";
import { SchemaForm, schemaErrors } from "./SchemaForm";
import { SpecBuilder } from "./SpecBuilder";
import {
  BACKTEST_LABEL, STAGE_TONE, apiErrors, useBacktests, usePreviewSignals, useRenameStrategy, useSaveVersion,
  useStrategy, useUniverse, useValidateSpec, useVersions,
  type BacktestSummary, type PreviewResult, type Spec, type StrategyDetail, type StrategyVersion, type ValidateResult,
} from "./useStrategies";

export function StrategyStudioPage() {
  const { id } = useParams<{ id: string }>();
  const [sp, setSp] = useSearchParams();
  const q = useStrategy(id);
  const tab = sp.get("tab") ?? (sp.get("backtest") ? "test" : "define");
  const setTab = (t: string) => { const n = new URLSearchParams(sp); n.set("tab", t); setSp(n, { replace: true }); };

  if (q.isLoading) return <div className="p-6 pt-2 max-w-[1400px]"><Skeleton className="h-64 w-full" /></div>;
  if (q.isError || !q.data) {
    return (
      <div className="p-6 pt-2 max-w-[1400px]">
        <EmptyState title="Strategy not found" description="It may have been deleted, or it belongs to another workspace." />
      </div>
    );
  }
  const s = q.data;
  return (
    <div className="p-6 pt-2 space-y-5 max-w-[1400px]">
      <Header s={s} onAction={(a) => setTab(a === "backtest" ? "test" : "define")} />
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="define">Define</TabsTrigger>
          <TabsTrigger value="test">Test</TabsTrigger>
          <TabsTrigger value="versions">Versions ({s.versions})</TabsTrigger>
          <TabsTrigger value="sim" disabled title="Agent simulation — Studio phase P7, not built yet">Agent Sim</TabsTrigger>
          <TabsTrigger value="paper" disabled title="Paper trading — Studio phase P6, not built yet">Paper</TabsTrigger>
          <TabsTrigger value="promote" disabled title="Promotion gates — Studio phase P8, not built yet">Promote</TabsTrigger>
        </TabsList>
        <TabsContent value="define"><DefineTab s={s} onSaved={() => undefined} /></TabsContent>
        <TabsContent value="test"><TestTab s={s} /></TabsContent>
        <TabsContent value="versions"><VersionsTab s={s} /></TabsContent>
      </Tabs>
    </div>
  );
}

/* ---- header + what's next ---- */

function Header({ s, onAction }: { s: StrategyDetail; onAction: (a: string) => void }) {
  const rename = useRenameStrategy(s.id);
  const [editing, setEditing] = React.useState(false);
  const [name, setName] = React.useState(s.name);
  React.useEffect(() => setName(s.name), [s.name]);
  return (
    <header className="space-y-3">
      <Link to="/strategies" className="inline-flex items-center gap-1 text-caption text-fg-muted hover:text-fg">
        <ArrowLeft className="h-3.5 w-3.5" /> All strategies
      </Link>
      <div className="flex flex-wrap items-center gap-3">
        {editing ? (
          <form className="flex gap-2" onSubmit={(e) => {
            e.preventDefault();
            rename.mutate({ name }, { onSuccess: () => setEditing(false), onError: (er) => toast.error(apiErrors(er).detail) });
          }}>
            <input aria-label="Strategy name" value={name} onChange={(e) => setName(e.target.value)}
                   className="h-9 rounded-sm border border-border bg-surface px-3 text-h3 text-fg" />
            <Button size="sm" type="submit" loading={rename.isPending}>Save</Button>
          </form>
        ) : (
          <h1 className="text-h1 text-fg flex items-center gap-2">
            {s.name}
            <Button size="icon" variant="ghost" aria-label="Rename" onClick={() => setEditing(true)}><Pencil className="h-3.5 w-3.5" /></Button>
          </h1>
        )}
        <Badge tone={STAGE_TONE[s.stage]} dot>{s.stage}</Badge>
        <Badge tone="neutral">v{s.current_version}</Badge>
        <Badge tone={s.backtest === "backtest" ? "success" : s.backtest === "forward_paper_only" ? "warning" : "neutral"}>
          {BACKTEST_LABEL[s.backtest]}
        </Badge>
        {s.archived && <Badge tone="neutral">archived</Badge>}
        <span className="text-caption text-fg-subtle">{s.kind === "spec" ? "Rule spec" : `Plugin · ${s.strategy_name}`}</span>
      </div>
      <div className="flex items-start gap-3 rounded-md border border-accent/30 bg-accent/5 px-4 py-3" data-testid="next-step">
        <ArrowRight className="h-4 w-4 text-accent shrink-0 mt-0.5" aria-hidden />
        <div className="flex-1 min-w-0">
          <div className="text-body-sm font-medium text-fg">{s.next_step.title}</div>
          <div className="text-body-sm text-fg-muted">{s.next_step.detail}</div>
        </div>
        {s.next_step.action === "backtest" && (
          <Button size="sm" onClick={() => onAction("backtest")}>Backtest v{s.current_version}</Button>
        )}
      </div>
      {s.live_gate && s.live_gate.passes === false && (
        <div className="flex gap-2 text-body-sm text-warn"><AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />{s.live_gate.message}</div>
      )}
    </header>
  );
}

/* ---- Define ---- */

function useDebounced<T>(value: T, ms = 400) {
  const [v, setV] = React.useState(value);
  React.useEffect(() => { const t = setTimeout(() => setV(value), ms); return () => clearTimeout(t); }, [value, ms]);
  return v;
}

function DefineTab({ s, onSaved }: { s: StrategyDetail; onSaved: () => void }) {
  const save = useSaveVersion(s.id);
  const [note, setNote] = React.useState("");
  const isSpec = s.kind === "spec";
  const [spec, setSpec] = React.useState<Spec>(s.definition as Spec);
  const [params, setParams] = React.useState<Record<string, unknown>>(s.definition as Record<string, unknown>);
  React.useEffect(() => {
    if (isSpec) setSpec(s.definition as Spec); else setParams(s.definition as Record<string, unknown>);
  }, [s.version_hash]);  // eslint-disable-line react-hooks/exhaustive-deps
  const validate = useValidateSpec();
  const debounced = useDebounced(spec);
  const [val, setVal] = React.useState<ValidateResult | null>(null);
  React.useEffect(() => {
    if (!isSpec) return;
    validate.mutate(debounced, { onSuccess: setVal });
  }, [JSON.stringify(debounced)]);  // eslint-disable-line react-hooks/exhaustive-deps

  const dirty = JSON.stringify(isSpec ? spec : params) !== JSON.stringify(s.definition);
  const clientErrs = !isSpec ? schemaErrors(s.schema, params) : [];
  const serverErr = save.isError ? apiErrors(save.error) : null;
  const errors = [...(isSpec ? val?.errors ?? [] : clientErrs), ...(serverErr?.errors ?? [])];
  const canSave = dirty && errors.length === 0 && (!isSpec || val?.valid);

  const onSave = () =>
    save.mutate(isSpec ? { spec, note } : { params, note }, {
      onSuccess: (r) => {
        toast.success(r.created ? `Saved as v${r.instance.current_version}` : "No change — still the same version");
        setNote("");
        onSaved();
      },
    });

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_360px] gap-5">
      <div className="space-y-5 min-w-0">
        {s.backtest !== "backtest" && (
          <div className="flex gap-2 rounded-md border border-warn/40 bg-warn/10 px-4 py-3 text-body-sm text-fg">
            <Info className="h-4 w-4 text-warn shrink-0 mt-0.5" />{s.backtest_reason}
          </div>
        )}
        <Card>
          <CardHeader>
            <CardTitle>{isSpec ? "Rules" : "Parameters"}</CardTitle>
            <CardDescription>
              {isSpec ? "Signals are decided on a bar's close and filled at the next bar's open." : "Rendered from the plugin's own schema()."}
              {" "}Saving writes v{s.current_version + 1}; older versions and their results stay as they were.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isSpec ? (
              <SpecBuilder spec={spec} onChange={setSpec} errors={errors} disabled={s.archived} />
            ) : (
              <SchemaForm schema={s.schema} value={params} onChange={setParams} errors={errors}
                          broken={s.broken} disabled={s.archived} />
            )}
            {!isSpec && (s.template_notes?.length ?? 0) > 0 && (
              <ul className="mt-3 space-y-1 text-caption text-fg-muted list-disc pl-4">
                {s.template_notes!.map((n) => <li key={n}>{n}</li>)}
              </ul>
            )}
          </CardContent>
        </Card>
        <div className="flex flex-wrap items-end gap-2">
          <input aria-label="Version note" placeholder="What changed? (optional)" value={note} onChange={(e) => setNote(e.target.value)}
                 className="h-9 flex-1 min-w-[12rem] rounded-sm border border-border bg-surface px-3 text-body-sm text-fg" />
          <Button leading={<Save className="h-4 w-4" />} disabled={!canSave || s.archived} loading={save.isPending} onClick={onSave}>
            Save as v{s.current_version + 1}
          </Button>
          {dirty && <Button variant="ghost" onClick={() => (isSpec ? setSpec(s.definition as Spec) : setParams(s.definition as Record<string, unknown>))}>Discard</Button>}
        </div>
        {serverErr && !serverErr.errors.length && <p role="alert" className="text-body-sm text-pnl-down">{serverErr.detail}</p>}
        {isSpec && <PreviewCard spec={spec} valid={!!val?.valid} />}
      </div>
      {isSpec && (
        <aside className="space-y-3">
          <Card>
            <CardHeader><CardTitle>In plain English</CardTitle></CardHeader>
            <CardContent>
              {val?.valid ? (
                <ul className="space-y-1.5 text-body-sm text-fg-muted" data-testid="explain">
                  {val.explain.map((l) => <li key={l}>{l}</li>)}
                </ul>
              ) : (
                <p className="text-body-sm text-pnl-down">{val ? `${val.errors.length} problem(s) to fix` : "Checking…"}</p>
              )}
              {(val?.warnings ?? []).map((w) => (
                <p key={w.path + w.message} className="mt-2 flex gap-1.5 text-caption text-warn">
                  <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-px" />{w.message}
                </p>
              ))}
              {val?.valid && (
                <p className="mt-2 flex items-center gap-1 text-caption text-pnl-up"><CheckCircle2 className="h-3.5 w-3.5" /> Valid · {val.spec_hash?.slice(0, 10)}</p>
              )}
            </CardContent>
          </Card>
        </aside>
      )}
    </div>
  );
}

function shiftDays(iso: string, days: number) {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function PreviewCard({ spec, valid }: { spec: Spec; valid: boolean }) {
  const prev = usePreviewSignals();
  const universe = useUniverse();
  const daily = spec.timeframe === "1d";
  const last = (daily ? universe.data?.daily.last : universe.data?.intraday.last) ?? new Date().toISOString().slice(0, 10);
  const [symbol, setSymbol] = React.useState(spec.universe.symbols[0] ?? "RELIANCE");
  const [fromRaw, setFrom] = React.useState<string | null>(null);
  const [toRaw, setTo] = React.useState<string | null>(null);
  const to = toRaw ?? last;
  const from = fromRaw ?? shiftDays(to, daily ? -180 : -6);
  const data: PreviewResult | undefined = prev.data;
  const chart = React.useMemo(() => {
    if (!data) return [];
    const sig = new Map(data.signals.map((x) => [x.t, x]));
    return data.bars.map((b) => {
      const x = sig.get(b.t);
      return { t: b.t.slice(5, 16).replace("T", " "), c: b.c,
               buy: x?.side === "BUY" ? x.price : null, short: x?.side === "SHORT" ? x.price : null };
    });
  }, [data]);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Eye className="h-4 w-4" />Instant preview</CardTitle>
        <CardDescription>Where this version's rules fire on one stock (signals only — no sizing or costs). Up to 31 days intraday.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <form className="flex flex-wrap items-end gap-2" onSubmit={(e) => {
          e.preventDefault();
          prev.mutate({ spec, symbol, from_date: from, to_date: to });
        }}>
          <label className="text-caption text-fg-muted">Stock
            <input aria-label="Preview symbol" value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                   className="block h-8 w-32 rounded-sm border border-border bg-surface px-2 text-body-sm text-fg font-mono" />
          </label>
          <label className="text-caption text-fg-muted">From
            <input type="date" aria-label="Preview from" value={from} onChange={(e) => setFrom(e.target.value)}
                   className="block h-8 rounded-sm border border-border bg-surface px-2 text-body-sm text-fg" />
          </label>
          <label className="text-caption text-fg-muted">To
            <input type="date" aria-label="Preview to" value={to} onChange={(e) => setTo(e.target.value)}
                   className="block h-8 rounded-sm border border-border bg-surface px-2 text-body-sm text-fg" />
          </label>
          <Button type="submit" size="sm" variant="secondary" disabled={!valid} loading={prev.isPending}>Preview signals</Button>
        </form>
        {prev.isError && <p role="alert" className="text-body-sm text-pnl-down">{apiErrors(prev.error).errors[0]?.message ?? apiErrors(prev.error).detail}</p>}
        {data && (
          <div className="space-y-2" data-testid="preview-result">
            <p className="text-caption text-fg-subtle">
              {data.message ?? `${data.signals.length} signal(s) on ${data.symbol} · ${data.bars.length} bars${data.downsampled ? " (chart downsampled)" : ""} · ${data.source ?? ""}`}
            </p>
            {chart.length > 1 && (
              <div className="h-48" role="img" aria-label="Preview chart">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chart} margin={{ top: 4, right: 8, left: 8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="t" hide />
                    <YAxis domain={["auto", "auto"]} width={60} fontSize={11} />
                    <ReTooltip />
                    <Line type="monotone" dataKey="c" dot={false} stroke="#6366f1" strokeWidth={1.2} isAnimationActive={false} />
                    <Scatter dataKey="buy" fill="#22c55e" isAnimationActive={false} />
                    <Scatter dataKey="short" fill="#ef4444" isAnimationActive={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            )}
            {data.signals.length > 0 && (
              <ul className="max-h-40 overflow-y-auto text-caption font-mono text-fg-muted space-y-0.5">
                {data.signals.slice(0, 100).map((x) => (
                  <li key={x.t}>{x.t.slice(0, 16).replace("T", " ")} {x.side} @ {x.price} · stop {x.stop}{x.target ? ` · target ${x.target}` : ""} → fills {x.fill_t?.slice(11, 16) ?? "—"}</li>
                ))}
              </ul>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ---- Test ---- */

function TestTab({ s }: { s: StrategyDetail }) {
  const [sp, setSp] = useSearchParams();
  const list = useBacktests(s.id);
  const selected = sp.get("backtest") ?? list.data?.[0]?.id ?? null;
  const select = (id: string) => { const n = new URLSearchParams(sp); n.set("backtest", id); n.set("tab", "test"); setSp(n, { replace: true }); };
  const cols: Column<BacktestSummary>[] = [
    { key: "created_at", header: "When", render: (b) => <span className="text-caption">{fmtRel(b.created_at)} ago</span> },
    { key: "version", header: "Ver", render: (b) => <Badge tone={b.version === s.current_version ? "brand" : "neutral"}>v{b.version ?? "?"}</Badge> },
    { key: "from_date", header: "Period", render: (b) => <span className="font-mono text-caption">{b.from_date} → {b.to_date}</span> },
    { key: "status", header: "Result", render: (b) => <BacktestLine b={b} /> },
  ];
  return (
    <div className="space-y-5">
      <Card>
        <CardHeader>
          <CardTitle>Run a backtest</CardTitle>
          <CardDescription>
            Deterministic replay on stored candles with the live risk caps, NSE costs and slippage. The last 30% of
            sessions is held out as out-of-sample.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <BacktestRunForm strategy={s} definition={s.definition} onStarted={select} />
        </CardContent>
      </Card>
      {selected && <BacktestReportById id={selected} />}
      {(list.data?.length ?? 0) > 0 && (
        <Card>
          <CardHeader><CardTitle>All backtests of this strategy</CardTitle></CardHeader>
          <CardContent className="p-0">
            <DataTable<BacktestSummary> columns={cols} rows={list.data ?? []} dense rowKey={(b) => b.id}
                                        onRowClick={(b) => select(b.id)} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

/* ---- Versions ---- */

export function diffPaths(a: unknown, b: unknown, path = ""): string[] {
  if (JSON.stringify(a) === JSON.stringify(b)) return [];
  const isObj = (x: unknown) => x !== null && typeof x === "object" && !Array.isArray(x);
  if (isObj(a) && isObj(b)) {
    const keys = new Set([...Object.keys(a as object), ...Object.keys(b as object)]);
    return [...keys].flatMap((k) =>
      diffPaths((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], path ? `${path}.${k}` : k));
  }
  return [path || "(root)"];
}

function VersionsTab({ s }: { s: StrategyDetail }) {
  const q = useVersions(s.id);
  const nav = useNavigate();
  const rows = q.data ?? [];
  const cols: Column<StrategyVersion>[] = [
    { key: "version", header: "Version", render: (v) => <Badge tone={v.version === s.current_version ? "brand" : "neutral"}>v{v.version}</Badge> },
    { key: "note", header: "Note", render: (v) => <span className="text-body-sm">{v.note || "—"}</span> },
    { key: "changes", header: "Changed vs previous", render: (v) => {
      const prev = rows.find((x) => x.version === v.version - 1);
      const d = prev ? diffPaths(prev.definition, v.definition) : [];
      return <span className="text-caption font-mono text-fg-muted">{prev ? (d.length ? d.slice(0, 6).join(", ") + (d.length > 6 ? ` +${d.length - 6}` : "") : "no change") : "first version"}</span>;
    } },
    { key: "backtests", header: "Backtests", kind: "num" },
    { key: "created_at", header: "Saved", render: (v) => <span className="text-caption">{fmtRel(v.created_at)} ago{v.created_by ? ` · ${v.created_by}` : ""}</span> },
    { key: "spec_hash", header: "Hash", render: (v) => <span className={cn("font-mono text-caption text-fg-subtle")}>{v.spec_hash.slice(0, 10)}</span> },
  ];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Versions</CardTitle>
        <CardDescription>Definitions are immutable once saved; every backtest points at the version it tested.</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <DataTable<StrategyVersion> columns={cols} rows={rows} loading={q.isLoading} dense rowKey={(v) => v.id}
                                    onRowClick={() => nav(`/strategies/${s.id}?tab=test`)} />
      </CardContent>
    </Card>
  );
}
