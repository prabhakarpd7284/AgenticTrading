import * as React from "react";
import { ChevronDown, ChevronRight, ExternalLink, MoonStar, ShieldAlert, Terminal } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorState } from "@/components/ui/ErrorState";
import { cn, fmtDateTime, fmtInr } from "@/lib/utils";

import {
  RUN_SCAN_COMMAND, useCooldownStatus, useNightlyScan,
  type Allowed, type CooldownRow, type NightlyScan, type ScanRow, type SentimentLabel,
} from "./useNightlyScan";

type Tone = "neutral" | "brand" | "success" | "warning" | "danger" | "info";

const LABEL_TONE: Record<SentimentLabel, Tone> = {
  strong_positive: "success",
  positive: "success",
  neutral: "neutral",
  negative: "danger",
  strong_negative: "danger",
  unscored: "warning",
};

const LABEL_TEXT: Record<SentimentLabel, string> = {
  strong_positive: "Strong +",
  positive: "Positive",
  neutral: "Neutral",
  negative: "Negative",
  strong_negative: "Strong −",
  unscored: "Unscored",
};

const ALLOWED_TONE: Record<Allowed, Tone> = { both: "neutral", long: "success", short: "danger", none: "warning" };
const ALLOWED_TEXT: Record<Allowed, string> = { both: "Both", long: "Long only", short: "Short only", none: "Skip" };

const STATUS_TONE: Record<NightlyScan["status"], Tone> = {
  success: "success", partial: "warning", failed: "danger", running: "info",
};

function fmtScore(n: number | null | undefined) {
  if (n === null || n === undefined) return "—";
  return `${n > 0 ? "+" : ""}${n.toFixed(2)}`;
}

function humanize(s: string | undefined | null) {
  return (s || "—").replace(/_/g, " ");
}

export function NightlyScanPage() {
  const scanQ = useNightlyScan();
  const cooldownQ = useCooldownStatus();
  const [query, setQuery] = React.useState("");
  const [restrictedOnly, setRestrictedOnly] = React.useState(false);

  const scan = scanQ.data?.scan ?? null;
  const rows = React.useMemo(() => {
    const all = scan?.rows ?? [];
    const q = query.trim().toUpperCase();
    return all.filter((r) => (!restrictedOnly || r.allowed !== "both") && (!q || r.symbol.includes(q)));
  }, [scan, query, restrictedOnly]);

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <p className="text-caption uppercase tracking-wider text-fg-subtle">Pre-market rules · 21:00 IST</p>
          <h1 className="text-h1 text-fg">
            Nightly <span className="text-accent">Scan</span>
          </h1>
          <p className="text-body-sm text-fg-muted mt-1">
            Next-day levels, news sentiment and the directions automated strategies may trade.
          </p>
        </div>
        {scan && (
          <div className="flex items-center gap-2 flex-wrap text-body-sm text-fg-muted">
            <span>For <span className="text-fg font-medium">{scan.for_date}</span></span>
            <Badge tone={STATUS_TONE[scan.status]}>{scan.status}</Badge>
            {scanQ.data?.running && <Badge tone="info" dot>re-running</Badge>}
            <span>· finished {fmtDateTime(scan.finished_at)}</span>
            <span>· {Math.round(scan.timings?.total ?? 0)}s</span>
            <Badge tone={scan.telegram_sent ? "success" : "neutral"}>
              {scan.telegram_sent ? "Telegram sent" : "No Telegram"}
            </Badge>
          </div>
        )}
      </header>

      <CooldownCard rows={cooldownQ.data?.results} loading={cooldownQ.isLoading}
                    error={cooldownQ.isError} />

      {scanQ.isError ? (
        <ErrorState
          title="Couldn't load the nightly scan"
          description="The nightly-scan endpoint failed. Retry, or check that the backend is reachable."
          onRetry={() => scanQ.refetch()}
        />
      ) : scanQ.isLoading ? (
        <Skeleton className="h-64 w-full" />
      ) : !scan ? (
        <EmptyState
          icon={<MoonStar />}
          title="No nightly scan yet"
          description={`The scan runs at 21:00 IST before each trading day. Run it by hand with: ${RUN_SCAN_COMMAND}`}
        />
      ) : (
        <>
          <SummaryStrip scan={scan} />
          {scan.summary?.sentiment_available === false && (
            <div role="alert" className="flex items-center gap-2 rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-body-sm text-warn">
              <ShieldAlert className="h-4 w-4" />
              Sentiment could not be computed for this scan — the sentiment filter is not blocking anything.
            </div>
          )}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-3 flex-wrap">
              <div>
                <CardTitle>Per-stock verdicts</CardTitle>
                <CardDescription>
                  Strong negative → no longs · strong positive → no shorts · results/major event due → skip.
                  Click a row for headlines and levels.
                </CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Input aria-label="Filter symbols" placeholder="Symbol…" value={query}
                       onChange={(e) => setQuery(e.target.value)} className="w-36" />
                <Button variant={restrictedOnly ? "primary" : "secondary"} size="sm"
                        onClick={() => setRestrictedOnly((v) => !v)}>
                  {restrictedOnly ? "Showing restricted" : "Restricted only"}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <ScanTable rows={rows} />
            </CardContent>
          </Card>
          <SourcesCard scan={scan} />
        </>
      )}
    </div>
  );
}

function CooldownCard({ rows, loading, error }: { rows?: CooldownRow[]; loading: boolean; error: boolean }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Losing-streak cooldown</CardTitle>
        <CardDescription>
          3 consecutive loss days → 2 trading days of no automated entries (shadow-recorded), then risk ×0.5
          until the first profitable day. Manual orders are never blocked.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-10 w-full" />
        ) : error ? (
          <p className="text-body-sm text-pnl-down">Couldn't load cooldown status.</p>
        ) : !rows || rows.length === 0 ? (
          <p className="text-body-sm text-fg-muted">No trading portfolios.</p>
        ) : (
          <ul className="space-y-3">
            {rows.map((c) => (
              <li key={c.portfolio_id} className="flex flex-wrap items-center gap-3" data-testid="cooldown-row">
                <span className="font-medium text-fg">{c.portfolio}</span>
                <Badge tone={c.status === "normal" ? "success" : c.status === "cooldown" ? "danger" : "warning"}>
                  {c.status === "cooldown" ? `Cooldown until ${c.cooldown_until}` : c.status}
                </Badge>
                <span className="text-body-sm text-fg-muted">
                  risk ×{c.risk_multiplier} · loss streak {c.loss_streak}
                </span>
                <span className="flex gap-1" aria-label="recent trade days">
                  {c.history.slice(-8).map((h) => (
                    <span key={h.date} title={`${h.date}: ${fmtInr(h.net_pnl)}${h.in_cooldown ? " (in cooldown)" : ""}`}
                          className={cn("h-3 w-3 rounded-sm",
                            h.kind === "loss" ? "bg-pnl-down" : h.kind === "profit" ? "bg-pnl-up" : "bg-fg-subtle",
                            h.in_cooldown && "opacity-40")} />
                  ))}
                </span>
                <span className="text-caption text-fg-subtle">{c.reason}</span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function SummaryStrip({ scan }: { scan: NightlyScan }) {
  const s = scan.summary ?? {};
  const m = s.market;
  const restricted = s.skip?.length ?? 0;
  const tiles: { label: string; value: React.ReactNode; hint?: string }[] = [
    { label: "Market mood", value: m ? `${humanize(m.label)} ${fmtScore(m.score)}` : "—",
      hint: m ? `${m.scored} market headlines · breadth ${m.breadth > 0 ? "+" : ""}${m.breadth}` : undefined },
    { label: "Restricted tomorrow", value: restricted, hint: `of ${s.universe ?? scan.rows.length} stocks` },
    { label: "Headlines scored", value: `${s.headlines_scored ?? 0}/${s.headlines_total ?? 0}`,
      hint: s.unscored?.length ? `${s.unscored.length} stock(s) unscored` : "Claude CLI" },
    { label: "Levels from", value: scan.session_date, hint: `pivots: ${s.pivot_source ?? "?"}` },
  ];
  return (
    <section aria-label="Summary" className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {tiles.map((t) => (
        <Card key={t.label}>
          <CardContent className="py-4">
            <p className="text-caption uppercase tracking-wider text-fg-subtle">{t.label}</p>
            <p className="text-h3 text-fg mt-1 capitalize">{t.value}</p>
            {t.hint && <p className="text-caption text-fg-muted mt-0.5">{t.hint}</p>}
          </CardContent>
        </Card>
      ))}
      {s.indices && Object.keys(s.indices).length > 0 && (
        <div className="col-span-2 lg:col-span-4 flex flex-wrap gap-4 text-body-sm text-fg-muted">
          {Object.entries(s.indices).map(([sym, ix]) => (
            <span key={sym}>
              <span className="text-fg font-medium">{sym}</span> CPR {ix.cpr_class} ({ix.cpr_width_pct.toFixed(2)}%) ·{" "}
              {humanize(ix.relationship)} → {humanize(ix.bias)}
            </span>
          ))}
        </div>
      )}
    </section>
  );
}

function ScanTable({ rows }: { rows: ScanRow[] }) {
  const [open, setOpen] = React.useState<string | null>(null);
  if (rows.length === 0) {
    return <EmptyState title="No stocks match" description="Clear the filter to see every stock." />;
  }
  return (
    <div className="overflow-x-auto rounded-md border border-border bg-surface">
      <table className="w-full border-collapse text-body-sm">
        <caption className="sr-only">Nightly scan per-stock verdicts</caption>
        <thead className="bg-surface-2 text-caption uppercase tracking-wider text-fg-subtle">
          <tr>
            <th scope="col" className="w-6" />
            <th scope="col" className="text-left px-3 py-2">Symbol</th>
            <th scope="col" className="text-left px-3 py-2">Allowed</th>
            <th scope="col" className="text-left px-3 py-2">Sentiment</th>
            <th scope="col" className="text-left px-3 py-2">Event</th>
            <th scope="col" className="text-right px-3 py-2">CPR width</th>
            <th scope="col" className="text-left px-3 py-2">Two-day</th>
            <th scope="col" className="text-right px-3 py-2">News</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const isOpen = open === r.symbol;
            return (
              <React.Fragment key={r.symbol}>
                <tr className={cn("border-t border-border cursor-pointer hover:bg-surface-2", isOpen && "bg-surface-2")}
                    onClick={() => setOpen(isOpen ? null : r.symbol)} data-testid={`row-${r.symbol}`}>
                  <td className="pl-2 text-fg-subtle">
                    {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                  </td>
                  <td className="px-3 py-2 font-medium text-fg">{r.symbol}</td>
                  <td className="px-3 py-2"><Badge tone={ALLOWED_TONE[r.allowed]}>{ALLOWED_TEXT[r.allowed]}</Badge></td>
                  <td className="px-3 py-2">
                    <Badge tone={LABEL_TONE[r.sentiment_label]}>{LABEL_TEXT[r.sentiment_label]}</Badge>
                    <span className="ml-2 font-mono text-fg-muted">{fmtScore(r.sentiment_score)}</span>
                  </td>
                  <td className="px-3 py-2">
                    {r.event_tomorrow ? <Badge tone="warning" title={r.event_purpose}>Event</Badge>
                                      : <span className="text-fg-subtle">—</span>}
                  </td>
                  <td className="px-3 py-2 text-right font-mono">
                    {r.cpr_width_pct === null ? "—" : `${r.cpr_width_pct.toFixed(2)}%`}
                    {r.cpr_class && <span className="ml-1 text-caption text-fg-subtle">{r.cpr_class}</span>}
                  </td>
                  <td className="px-3 py-2 text-fg-muted">
                    {r.two_day_relationship ? `${humanize(r.two_day_relationship)} · ${humanize(r.bias)}` : "—"}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-fg-muted">{r.scored_count}/{r.headline_count}</td>
                </tr>
                {isOpen && (
                  <tr className="border-t border-border bg-surface-2/50">
                    <td />
                    <td colSpan={7} className="px-3 py-3"><RowDetail row={r} /></td>
                  </tr>
                )}
              </React.Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RowDetail({ row }: { row: ScanRow }) {
  const L = row.levels ?? {};
  const lv = (o: Record<string, number> | undefined, k: string) => (o && o[k] !== undefined ? o[k].toFixed(2) : "—");
  return (
    <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
      <div className="space-y-2">
        {row.reasons.length > 0 && (
          <ul className="text-body-sm text-warn list-disc pl-4">
            {row.reasons.map((x) => <li key={x}>{x}</li>)}
          </ul>
        )}
        {row.headlines.length === 0 ? (
          <p className="text-body-sm text-fg-muted">No headlines in the window.</p>
        ) : (
          <ul className="space-y-1.5">
            {row.headlines.map((h) => (
              <li key={h.id} className="flex items-start gap-2 text-body-sm">
                <span className={cn("font-mono w-12 shrink-0 text-right",
                  (h.score ?? 0) > 0 ? "text-pnl-up" : (h.score ?? 0) < 0 ? "text-pnl-down" : "text-fg-subtle")}>
                  {fmtScore(h.score)}
                </span>
                <a href={h.url} target="_blank" rel="noreferrer noopener"
                   className="text-fg hover:text-accent inline-flex items-start gap-1">
                  {h.title}
                  <ExternalLink className="h-3 w-3 mt-1 shrink-0" />
                </a>
                <span className="text-caption text-fg-subtle shrink-0">
                  {h.publisher || h.source}{h.materiality ? ` · ${h.materiality}` : ""} · {fmtDateTime(h.published_at)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
      <dl className="grid grid-cols-3 gap-x-3 gap-y-1 text-body-sm font-mono">
        <dt className="text-fg-subtle col-span-3 font-sans text-caption uppercase">
          Levels {L.data_date ? `(from ${L.data_date}${L.stale ? ", stale" : ""})` : ""}
        </dt>
        <span>TC {lv(L.cpr as Record<string, number> | undefined, "tc")}</span>
        <span>PP {lv(L.cpr as Record<string, number> | undefined, "pp")}</span>
        <span>BC {lv(L.cpr as Record<string, number> | undefined, "bc")}</span>
        <span>R1 {lv(L.floor, "r1")}</span>
        <span>R2 {lv(L.floor, "r2")}</span>
        <span>R3 {lv(L.floor, "r3")}</span>
        <span>S1 {lv(L.floor, "s1")}</span>
        <span>S2 {lv(L.floor, "s2")}</span>
        <span>S3 {lv(L.floor, "s3")}</span>
        <span>H3 {lv(L.camarilla, "h3")}</span>
        <span>H4 {lv(L.camarilla, "h4")}</span>
        <span>L3 {lv(L.camarilla, "l3")}</span>
        <span className="col-span-3 font-sans text-fg-muted">
          Camarilla: {humanize(row.camarilla_relationship)}
        </span>
      </dl>
    </div>
  );
}

function SourcesCard({ scan }: { scan: NightlyScan }) {
  const entries = Object.entries(scan.sources ?? {});
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Terminal className="h-4 w-4" /> Sources & timings</CardTitle>
        <CardDescription>
          Headlines and links only. A failing source degrades the scan to “partial”; it never aborts it.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <ul className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3 text-body-sm">
          {entries.map(([k, v]) => (
            <li key={k} className="flex items-center gap-2">
              <Badge tone={v?.ok === false ? "danger" : "success"}>{v?.ok === false ? "fail" : "ok"}</Badge>
              <span className="text-fg">{humanize(k)}</span>
              <span className="text-fg-subtle">
                {typeof v?.seen === "number" ? `${v.new ?? 0} new / ${v.seen} seen` : ""}
                {typeof v?.scored === "number" ? `${v.scored as number} scored` : ""}
                {typeof v?.secs === "number" ? ` · ${Math.round(v.secs)}s` : ""}
              </span>
            </li>
          ))}
        </ul>
        <p className="text-caption text-fg-subtle">
          {Object.entries(scan.timings ?? {}).map(([k, v]) => `${k} ${Math.round(v)}s`).join(" · ")}
        </p>
        {scan.summary?.degraded && scan.summary.degraded.length > 0 && (
          <p className="text-caption text-warn">Degraded: {scan.summary.degraded.join("; ")}</p>
        )}
      </CardContent>
    </Card>
  );
}
