/**
 * Today — the home destination (UX redesign R3, plan §3).
 *
 *   [ Action Board — today's ideas from your strategies (directional, PAPER, unproven) ]
 *   [ Portfolio snapshot ]  [ Needs attention ]
 *   [ Market context (collapsed) — the Market Pulse page, still at /pulse            ]
 *
 * The status strip (mode / data / jobs) lives in the app shell's top bar.
 */
import * as React from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, ChevronDown, ChevronRight, ClipboardList, Wallet } from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn, clsPnl, fmtInr, fmtNum } from "@/lib/utils";
import { MarketPulsePage } from "@/features/market-pulse/MarketPulsePage";
import { usePortfolioOverview } from "@/features/portfolio/usePortfolio";
import { useRunsStatus } from "@/features/runs/useRuns";
import {
  CRITERION_LABEL,
  type ActionStatus,
  type TodayActionCard,
  type TodayResponse,
  type TrackRecord,
  useCooldown,
  useDecideAction,
  useTodayActions,
} from "./useToday";

const IST_TIME: Intl.DateTimeFormatOptions = { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "Asia/Kolkata" };

function istTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleTimeString("en-GB", IST_TIME);
}

function px(n: number | null | undefined): string {
  return fmtNum(n, 2);
}

function apiError(err: unknown): string {
  const e = err as { response?: { data?: { detail?: string } }; message?: string };
  return e?.response?.data?.detail ?? e?.message ?? "request failed";
}

const STATUS_TONE: Record<ActionStatus, "success" | "warning" | "danger" | "info" | "neutral" | "brand"> = {
  proposed: "brand",
  approved: "info",
  executed: "success",
  snoozed: "info",
  skipped: "neutral",
  expired: "neutral",
  rejected: "danger",
};

export function TodayPage() {
  const today = useTodayActions();
  return (
    <div className="p-6 pt-2 space-y-4 max-w-[1400px]">
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h1 className="text-h2 font-semibold">Today</h1>
          <p className="text-body-sm text-fg-muted">
            Ideas from your own strategies, the book, and what needs you. Not advice — every idea is paper-traded
            and carries its own (thin) evidence.
          </p>
        </div>
        {today.data && (
          <span className="text-caption text-fg-subtle">
            {new Date(`${today.data.date}T00:00:00`).toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "short" })}
          </span>
        )}
      </header>

      <ActionBoard query={today} />

      <div className="grid gap-4 lg:grid-cols-2">
        <PortfolioSnapshot />
        <NeedsAttention board={today.data} />
      </div>

      <MarketContext />
    </div>
  );
}

// ─── Action Board ────────────────────────────────────────────────────────

function ActionBoard({ query }: { query: ReturnType<typeof useTodayActions> }) {
  const { data, isLoading, isError, error, refetch } = query;
  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2">
              <ClipboardList className="h-4 w-4" aria-hidden /> Action Board
            </CardTitle>
            <CardDescription>Stocks to act on today — intraday, from the directional planner.</CardDescription>
          </div>
          {data && (
            <div className="flex flex-wrap items-center gap-1.5">
              <Badge tone="info">{data.strategy.mode}</Badge>
              <Badge tone="warning" data-testid="stage-badge">{data.strategy.stage}</Badge>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {isLoading && <Skeleton className="h-24 w-full" />}
        {isError && (
          <div className="text-body-sm text-pnl-down">
            Could not load the Action Board: {apiError(error)}{" "}
            <Button size="sm" variant="ghost" onClick={() => refetch()}>Retry</Button>
          </div>
        )}
        {data && (
          <>
            <BoardStatusLine data={data} />
            <TrackRecordStrip record={data.track_record[data.strategy.name]} />
            {data.cards.length === 0 ? (
              <BoardEmpty data={data} />
            ) : (
              <div className="grid gap-3 xl:grid-cols-2" data-testid="action-cards">
                {data.cards.map((c) => (
                  <ActionCardView key={c.id} card={c} canDecide={data.can_decide}
                                  record={data.track_record[c.strategy]} isToday={c.date === data.date} />
                ))}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function ReceiptLink({ id, children }: { id: string; children?: React.ReactNode }) {
  return <Link to={`/runs/${id}`} className="text-accent hover:underline">{children ?? "receipt"}</Link>;
}

function BoardStatusLine({ data }: { data: TodayResponse }) {
  const r = data.board.receipt;
  if (!r) return null;
  const when = istTime(r.as_of ?? r.started_at);
  const tone = r.status === "success" ? "success" : r.status === "partial" ? "warning"
    : r.status === "skipped" ? "neutral" : r.status === "running" ? "info" : "danger";
  return (
    <div className="flex flex-wrap items-center gap-2 text-body-sm text-fg-muted" data-testid="board-status">
      <Badge tone={tone}>board {r.status}</Badge>
      <span>
        built {when}{r.replay ? " (replay)" : ""} · {r.candidates ?? 0} candidates · {r.planner_calls ?? 0} planned by
        Claude · {r.planned ?? 0} plan(s) · {r.passed_risk ?? 0} passed risk
      </span>
      <ReceiptLink id={r.id} />
    </div>
  );
}

function BoardEmpty({ data }: { data: TodayResponse }) {
  const r = data.board.receipt;
  let title = "No actions today";
  let description: string;
  if (!data.board.is_trading_day) {
    title = "Market closed today";
    description = "Not an NSE trading day — the board runs on trading days only.";
  } else if (!r) {
    title = "Board not built yet";
    description = data.board.due
      ? `It was due at ${data.board.schedule_label} and has not run — check Runs.`
      : `It runs at ${data.board.schedule_label}, after the first minutes of 1-minute data.`;
  } else if (r.status === "failed") {
    title = "Board failed — no cards";
    description = r.error || "The build failed its checks.";
  } else if (r.status === "skipped") {
    description = `Board skipped: ${r.skipped ?? "not applicable"}.`;
  } else {
    description =
      `No actions today — board built ${istTime(r.as_of ?? r.started_at)}, ${r.candidates ?? 0} candidates, ` +
      `${r.planned ?? 0} planned, ${r.passed_risk ?? 0} passed risk.`;
  }
  return (
    <EmptyState
      icon={<ClipboardList className="h-6 w-6" aria-hidden />}
      title={title}
      description={description}
      action={r ? <ReceiptLink id={r.id}>View the board receipt</ReceiptLink> : <Link to="/runs" className="text-accent hover:underline">Runs</Link>}
    />
  );
}

function TrackRecordStrip({ record }: { record?: TrackRecord }) {
  if (!record) return null;
  const p = record.paper;
  const m = record.missed;
  const winRate = p.win_rate == null ? "—" : `${Math.round(p.win_rate * 100)}%`;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-sm border border-border bg-surface-2/50 px-3 py-2 text-caption"
         data-testid="track-record">
      <span className="font-medium text-fg">Track record · {record.strategy}</span>
      <span>paper: <b>{p.label}</b> · win {winRate} · avg R {p.avg_r == null ? "—" : fmtNum(p.avg_r, 2)} ·
        net {p.net_pnl == null ? "—" : <span className={clsPnl(p.net_pnl)}>{fmtInr(p.net_pnl)}</span>}</span>
      <span>missed (skipped/expired, replayed): <b>{m.label}</b> · avg R {m.avg_r == null ? "—" : fmtNum(m.avg_r, 2)}</span>
      {!record.proven && <Badge tone="warning">unproven</Badge>}
    </div>
  );
}

function ActionCardView({ card, canDecide, record, isToday }: {
  card: TodayActionCard; canDecide: boolean; record?: TrackRecord; isToday: boolean;
}) {
  const decide = useDecideAction();
  const [mode, setMode] = React.useState<"idle" | "approve" | "skip">("idle");
  const [reason, setReason] = React.useState("");
  const open = (card.status === "proposed" || card.status === "snoozed") && !card.shadow;
  const rc = card.risk_check;
  const crit = rc.failed_criterion ? (CRITERION_LABEL[rc.failed_criterion] ?? rc.failed_criterion) : "";
  const reduced = card.risk_multiplier < 1 && !card.shadow;

  return (
    <div className={cn("rounded-md border border-border bg-surface p-3 space-y-2", card.shadow && "opacity-80")}
         data-testid={`action-card-${card.symbol}`}>
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-body font-semibold">{card.symbol}</span>
        <Badge tone={card.side === "BUY" ? "success" : "danger"}>{card.side}</Badge>
        <Badge tone={STATUS_TONE[card.status]}>{card.status}</Badge>
        {card.shadow && <Badge tone="warning">shadow · blocked by cooldown</Badge>}
        <Badge tone="neutral">{card.horizon}</Badge>
        <span className="ml-auto text-caption text-fg-subtle">
          {card.strategy} v{card.strategy_version}
          {card.confidence != null && <> · conf {fmtNum(card.confidence, 2)}</>}
        </span>
      </div>

      {card.reasons.length > 0 && (
        <ul className="list-disc space-y-0.5 pl-4 text-body-sm text-fg-muted">
          {card.reasons.slice(0, 3).map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      )}

      <dl className="grid grid-cols-3 gap-x-3 gap-y-1 text-body-sm sm:grid-cols-6">
        <Fact label="Entry" value={px(card.entry)} />
        <Fact label="Stop" value={px(card.stoploss)} />
        <Fact label="Target" value={px(card.target)} />
        <Fact label="Qty" value={fmtNum(card.suggested_qty)} />
        <Fact label="Capital" value={fmtInr(card.capital_required)} />
        <Fact label="₹ at risk" value={fmtInr(card.risk_at_stake) + (reduced ? ` (×${card.risk_multiplier})` : "")} />
      </dl>

      <div className="text-body-sm" data-testid="risk-precheck">
        {rc.approved ? (
          <Badge tone="success">risk pre-check: would pass</Badge>
        ) : (
          <span className="flex flex-wrap items-center gap-1.5">
            <Badge tone="danger">risk pre-check: would fail — {crit || "unknown"}</Badge>
            <span className="text-caption text-fg-muted">{rc.reason}</span>
          </span>
        )}
      </div>

      {record && (
        <div className="text-caption text-fg-subtle">
          {card.strategy} track record: paper {record.paper.label}
          {record.paper.n > 0 && record.paper.win_rate != null && <> · win {Math.round(record.paper.win_rate * 100)}%</>}
          {" "}· missed {record.missed.label}
        </div>
      )}

      {(card.decision_note || card.order || card.outcome) && (
        <div className="space-y-0.5 text-caption text-fg-muted">
          {card.decision_note && <div className="whitespace-pre-line">{card.decision_note}</div>}
          {card.order && <div>Paper order {card.order.status} · {card.order.qty} @ {px(card.order.price)}{card.order.error && ` · ${card.order.error}`}</div>}
          {card.outcome && <OutcomeLine o={card.outcome} />}
        </div>
      )}

      {decide.isError && <div className="text-caption text-pnl-down" role="alert">{apiError(decide.error)}</div>}

      {open && canDecide && isToday && (
        <div className="flex flex-wrap items-center gap-2 pt-1">
          {mode === "idle" && (
            <>
              <Button size="sm" onClick={() => setMode("approve")} disabled={decide.isPending}>Approve</Button>
              <Button size="sm" variant="secondary" onClick={() => setMode("skip")} disabled={decide.isPending}>Skip</Button>
              <span className="text-caption text-fg-subtle">Snooze</span>
              {[15, 30, 60].map((m) => (
                <Button key={m} size="sm" variant="ghost" disabled={decide.isPending}
                        onClick={() => decide.mutate({ id: card.id, decision: "snooze", minutes: m })}>
                  {m}m
                </Button>
              ))}
              <span className="ml-auto text-caption text-fg-subtle">
                {card.status === "snoozed" ? `snoozed until ${istTime(card.snoozed_until)} · ` : ""}expires {istTime(card.expires_at)}
              </span>
            </>
          )}
          {mode === "approve" && (
            <>
              <span className="text-body-sm">
                Place PAPER order: {card.side} {card.suggested_qty} {card.symbol} LIMIT {px(card.entry)}?
              </span>
              <Button size="sm" disabled={decide.isPending}
                      onClick={() => decide.mutate({ id: card.id, decision: "approve" }, { onSettled: () => setMode("idle") })}>
                Confirm paper order
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setMode("idle")}>Cancel</Button>
            </>
          )}
          {mode === "skip" && (
            <>
              <input
                aria-label="Skip reason"
                className="h-8 min-w-0 flex-1 rounded-sm border border-border bg-surface-2 px-2 text-body-sm"
                placeholder="Reason (optional)"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
              />
              <Button size="sm" variant="secondary" disabled={decide.isPending}
                      onClick={() => decide.mutate({ id: card.id, decision: "skip", reason }, { onSettled: () => setMode("idle") })}>
                Confirm skip
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setMode("idle")}>Cancel</Button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-caption text-fg-subtle">{label}</dt>
      <dd className="tabular-nums">{value}</dd>
    </div>
  );
}

function OutcomeLine({ o }: { o: NonNullable<TodayActionCard["outcome"]> }) {
  const text: Record<string, string> = {
    no_data: "no 1m candles stored for the replay",
    no_entry: "entry price never traded",
    stop: "would have hit the stop",
    target: "would have hit the target",
    eod: "would have exited at 15:14",
  };
  return (
    <div>
      What the plan would have done: {text[o.status] ?? o.status}
      {o.r != null && <> · <span className={clsPnl(o.r)}>{fmtNum(o.r, 2)}R</span></>}
      {o.pnl != null && o.entry_hit && <> · <span className={clsPnl(o.pnl)}>{fmtInr(o.pnl)}</span></>}
    </div>
  );
}

// ─── Portfolio snapshot ──────────────────────────────────────────────────

function PortfolioSnapshot() {
  const { data, isLoading, isError } = usePortfolioOverview();
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Wallet className="h-4 w-4" aria-hidden /> Portfolio</CardTitle>
        <CardDescription><Link to="/portfolio" className="text-accent hover:underline">Open Portfolio →</Link></CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading && <Skeleton className="h-16 w-full" />}
        {isError && <div className="text-body-sm text-fg-muted">Portfolio overview unavailable.</div>}
        {data && (
          <dl className="grid grid-cols-2 gap-3 text-body-sm" data-testid="portfolio-snapshot">
            <Fact label="Total value" value={fmtInr(data.total.value)} />
            <div>
              <dt className="text-caption text-fg-subtle">Day P&amp;L</dt>
              <dd className={cn("tabular-nums", clsPnl(data.pnl.day.total))}>{fmtInr(data.pnl.day.total)}</dd>
            </div>
            <div>
              <dt className="text-caption text-fg-subtle">Daily loss budget used</dt>
              <dd className="tabular-nums">
                {data.risk.daily_loss.pct_used == null ? "—" : `${fmtNum(data.risk.daily_loss.pct_used, 1)}%`}
                <span className="text-fg-subtle"> of {fmtInr(data.risk.daily_loss.limit)}</span>
              </dd>
            </div>
            <Fact label="Open positions"
                  value={fmtNum(data.components.equity.open_count + data.components.options.open_count)} />
          </dl>
        )}
      </CardContent>
    </Card>
  );
}

// ─── Needs attention ─────────────────────────────────────────────────────

function NeedsAttention({ board }: { board?: TodayResponse }) {
  const status = useRunsStatus();
  const cooldown = useCooldown();
  const items: { key: string; text: string; to: string }[] = [];

  for (const p of status.data?.jobs.problems ?? []) {
    items.push({ key: `job-${p.key}-${p.due_at}`, text: `${p.label}: ${p.state}`, to: p.receipt_id ? `/runs/${p.receipt_id}` : "/runs" });
  }
  for (const b of status.data?.data.brokers ?? []) {
    if (b.status !== "active") {
      items.push({ key: `broker-${b.broker}`, text: `${b.broker} link ${b.status} — re-login needed`, to: "/brokers" });
    }
  }
  for (const c of cooldown.data?.results ?? []) {
    if (c.status !== "normal") {
      items.push({
        key: `cooldown-${c.portfolio_id}`,
        text: c.status === "cooldown"
          ? `${c.portfolio}: losing-streak cooldown until ${c.cooldown_until ?? "?"} — automated entries blocked`
          : `${c.portfolio}: reduced risk ×${c.risk_multiplier} until the first profitable day`,
        to: "/nightly-scan",
      });
    }
  }
  const r = board?.board.receipt;
  if (r && r.status === "failed" && !items.some((i) => i.to === `/runs/${r.id}`)) {
    items.push({ key: "board", text: `Action Board build failed: ${r.error || "see receipt"}`, to: `/runs/${r.id}` });
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><AlertTriangle className="h-4 w-4" aria-hidden /> Needs attention</CardTitle>
        <CardDescription>Failed or missed jobs, broker links, cooldown.</CardDescription>
      </CardHeader>
      <CardContent>
        {status.isLoading ? <Skeleton className="h-12 w-full" /> : items.length === 0 ? (
          <div className="text-body-sm text-fg-muted" data-testid="attention-empty">Nothing needs you right now.</div>
        ) : (
          <ul className="space-y-1 text-body-sm" data-testid="attention-list">
            {items.map((i) => (
              <li key={i.key}><Link to={i.to} className="hover:underline">{i.text}</Link></li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

// ─── Market context ──────────────────────────────────────────────────────

function MarketContext() {
  const [open, setOpen] = React.useState(false);
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <button type="button" className="flex flex-1 items-center gap-2 text-left" aria-expanded={open}
                  onClick={() => setOpen((v) => !v)}>
            {open ? <ChevronDown className="h-4 w-4" aria-hidden /> : <ChevronRight className="h-4 w-4" aria-hidden />}
            <CardTitle>Market context</CardTitle>
            <span className="text-caption text-fg-subtle">regime, indices, sector heatmap</span>
          </button>
          <Link to="/pulse" className="text-caption text-accent hover:underline">Open Market Pulse</Link>
        </div>
      </CardHeader>
      {open && <CardContent data-testid="market-context"><MarketPulsePage /></CardContent>}
    </Card>
  );
}
