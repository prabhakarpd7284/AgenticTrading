import * as React from "react";
import {
  Activity, AlertTriangle, RefreshCcw, ShieldAlert, ShieldCheck, Zap,
} from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip as ReTooltip,
  XAxis, YAxis,
} from "recharts";

import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import { KPI } from "@/components/ui/KPI";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { DataTable, type Column } from "@/components/ui/DataTable";
import { Skeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/ErrorState";
import { FreshnessIndicator } from "@/components/ui/FreshnessIndicator";
import { cn, fmtRel } from "@/lib/utils";

import {
  QUEUE_LABEL, QUEUE_WARN_THRESHOLD,
  epochToClock, fmtCountdown, linkStatusTone, useBrokerMonitor,
  type BrokerLink, type CallRatePoint, type QueueDepths,
} from "./useBrokerMonitor";

export function BrokerMonitorPage() {
  const {
    data, isLoading, isError, refetch, isFetching, dataUpdatedAt,
  } = useBrokerMonitor();

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      {/* Header */}
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <p className="text-caption uppercase tracking-wider text-fg-subtle">
            Broker telemetry
          </p>
          <h1 className="text-h1 text-fg">
            Broker <span className="text-accent">Monitor</span>
          </h1>
          <p className="text-body-sm text-fg-muted mt-1 flex items-center gap-2 flex-wrap">
            Live SmartAPI rate-limit, call volume &amp; queue health
            <span aria-hidden>·</span>
            <FreshnessIndicator
              variant="muted"
              label="Updated"
              timestamp={dataUpdatedAt}
              freshMs={8_000}
              staleMs={20_000}
            />
          </p>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => refetch()}
          disabled={isFetching}
          aria-label="Refresh broker telemetry"
          title="Refetch broker telemetry"
        >
          <RefreshCcw className={cn("h-4 w-4", isFetching && "animate-spin")} />
        </Button>
      </header>

      {isError ? (
        <ErrorState
          title="Couldn't load broker telemetry"
          description="The monitor endpoint failed. Retry, or check that the backend is reachable."
          onRetry={() => refetch()}
        />
      ) : (
        <>
          {/* Row 1 — breaker + queue depths */}
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_2fr] gap-6">
            <BreakerCard
              open={data?.breaker.open}
              cooldownS={data?.breaker.cooldown_remaining_s}
              trips={data?.breaker.trips}
              loading={isLoading}
            />
            <QueueDepthsCard queues={data?.queues} loading={isLoading} />
          </div>

          {/* Row 2 — call volume chart */}
          <CallVolumeCard series={data?.call_rate} loading={isLoading} />

          {/* Row 3 — broker links */}
          <BrokerLinksCard links={data?.links} loading={isLoading} />
        </>
      )}
    </div>
  );
}

/* ================================================================== */
/* 1. Rate-limit breaker                                               */
/* ================================================================== */
function BreakerCard({
  open, cooldownS, trips, loading,
}: {
  open?: boolean;
  cooldownS?: number;
  trips?: number;
  loading?: boolean;
}) {
  // Local countdown ticker — the payload's cooldown is a snapshot, so we
  // decrement it once a second between 5s polls for a smooth read-out.
  const [remaining, setRemaining] = React.useState(cooldownS ?? 0);
  React.useEffect(() => {
    setRemaining(cooldownS ?? 0);
  }, [cooldownS]);
  React.useEffect(() => {
    if (!open || remaining <= 0) return;
    const id = window.setInterval(() => {
      setRemaining((r) => Math.max(0, r - 1));
    }, 1000);
    return () => window.clearInterval(id);
  }, [open, remaining]);

  return (
    <Card
      className={cn(
        open === true && "border-danger/40",
        open === false && "border-pnl-up/30",
      )}
    >
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle>Rate-limit breaker</CardTitle>
            <CardDescription>
              Trips when the SmartAPI rate limit is exceeded.
            </CardDescription>
          </div>
          {open === true ? (
            <ShieldAlert className="h-5 w-5 text-danger shrink-0" aria-hidden />
          ) : (
            <ShieldCheck className="h-5 w-5 text-pnl-up shrink-0" aria-hidden />
          )}
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-16 w-full" />
        ) : (
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div>
              <div
                className={cn(
                  "text-num-lg font-mono font-semibold",
                  open ? "text-danger" : "text-pnl-up",
                )}
                aria-live="polite"
              >
                {open ? "OPEN" : "CLOSED"}
              </div>
              <p className="text-caption text-fg-subtle mt-0.5">
                {open
                  ? "Calls paused — cooling down"
                  : "Healthy — calls flowing"}
              </p>
            </div>

            {open && (
              <div className="text-right">
                <div className="text-num-lg font-mono tabular text-fg">
                  {fmtCountdown(remaining)}
                </div>
                <p className="text-caption text-fg-subtle mt-0.5">cooldown left</p>
              </div>
            )}

            <Badge tone={(trips ?? 0) > 0 ? "warning" : "neutral"}>
              {trips ?? 0} trip{(trips ?? 0) === 1 ? "" : "s"}
            </Badge>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/* ================================================================== */
/* 2. SmartAPI call volume                                            */
/* ================================================================== */
function CallVolumeCard({
  series, loading,
}: {
  series?: CallRatePoint[];
  loading?: boolean;
}) {
  const chartData = React.useMemo(
    () =>
      (series ?? []).map((p) => ({
        clock: epochToClock(p.minute),
        calls: p.calls,
      })),
    [series],
  );
  const total = React.useMemo(
    () => (series ?? []).reduce((acc, p) => acc + p.calls, 0),
    [series],
  );

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle>SmartAPI call volume</CardTitle>
            <CardDescription>
              Calls per minute — last {(series ?? []).length || 30} minutes.
            </CardDescription>
          </div>
          <Badge tone="info" dot>
            <Zap className="h-3 w-3" aria-hidden /> {total} calls
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="h-64">
        {loading ? (
          <div className="h-full flex items-center justify-center">
            <Skeleton className="h-48 w-full" />
          </div>
        ) : chartData.length === 0 ? (
          <div className="h-full flex items-center justify-center text-body-sm text-fg-subtle">
            No call activity in the last 30 minutes.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="rgb(var(--border))" vertical={false} strokeDasharray="3 3" />
              <XAxis
                dataKey="clock"
                stroke="rgb(var(--fg-subtle))"
                fontSize={11}
                interval="preserveStartEnd"
                minTickGap={24}
                tickLine={false}
              />
              <YAxis
                stroke="rgb(var(--fg-subtle))"
                fontSize={11}
                width={36}
                allowDecimals={false}
              />
              <ReTooltip
                contentStyle={{
                  background: "rgb(var(--surface))",
                  border: "1px solid rgb(var(--border-strong))",
                  borderRadius: 6,
                  color: "rgb(var(--fg))",
                  fontSize: 12,
                }}
                cursor={{ fill: "rgb(var(--surface-2))" }}
                formatter={(v: number) => [`${v} calls`, "Calls"]}
                labelFormatter={(l) => `at ${l}`}
              />
              <Bar
                dataKey="calls"
                fill="rgb(var(--accent))"
                radius={[2, 2, 0, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}

/* ================================================================== */
/* 3. Celery queue depths                                             */
/* ================================================================== */
function QueueDepthsCard({
  queues, loading,
}: {
  queues?: QueueDepths;
  loading?: boolean;
}) {
  const keys: (keyof QueueDepths)[] = ["celery", "orders", "agents", "backtests"];
  const anyHot =
    !!queues && keys.some((k) => queues[k] > QUEUE_WARN_THRESHOLD);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle>Celery queue depths</CardTitle>
            <CardDescription>
              A growing backlog is the early-warning signal of a stuck worker.
            </CardDescription>
          </div>
          {anyHot && (
            <Badge tone="danger">
              <AlertTriangle className="h-3 w-3" aria-hidden /> backlog
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {keys.map((k) => {
            const depth = queues?.[k];
            const hot = (depth ?? 0) > QUEUE_WARN_THRESHOLD;
            return (
              <KPI
                key={k}
                label={QUEUE_LABEL[k]}
                value={depth}
                valueFormat="num"
                loading={loading}
                hint={hot ? `> ${QUEUE_WARN_THRESHOLD} — backlog` : "queued tasks"}
                className={cn(hot && "border-danger/40 bg-pnl-down/5")}
              />
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

/* ================================================================== */
/* 4. Broker links                                                    */
/* ================================================================== */
function BrokerLinksCard({
  links, loading,
}: {
  links?: BrokerLink[];
  loading?: boolean;
}) {
  const columns: Column<BrokerLink>[] = [
    {
      key: "display_name",
      header: "Account",
      render: (l) => (
        <div className="min-w-0">
          <div className="text-body-sm text-fg font-medium truncate">
            {l.display_name}
          </div>
          <div className="text-caption text-fg-subtle uppercase tracking-wider">
            {l.broker}
          </div>
        </div>
      ),
    },
    {
      key: "status",
      header: "Status",
      kind: "status",
      render: (l) => <Badge tone={linkStatusTone(l.status)} dot>{l.status}</Badge>,
    },
    {
      key: "last_refreshed_at",
      header: "Refreshed",
      align: "right",
      render: (l) => (
        <span className="font-mono tabular text-fg-muted" title={l.last_refreshed_at ?? ""}>
          {l.last_refreshed_at ? fmtRel(l.last_refreshed_at) : "—"}
        </span>
      ),
    },
    {
      key: "last_error",
      header: "Last error",
      render: (l) =>
        l.last_error ? (
          <span className="text-body-sm text-danger" title={l.last_error}>
            {l.last_error}
          </span>
        ) : (
          <span className="text-fg-subtle">—</span>
        ),
    },
  ];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-fg-muted" aria-hidden />
          <div>
            <CardTitle>Broker links</CardTitle>
            <CardDescription>
              Health of each linked broker account.
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <DataTable
          columns={columns}
          rows={links ?? []}
          loading={loading}
          rowKey={(l) => l.id}
          emptyTitle="No broker links"
          emptyDescription="Link a broker account to start monitoring its health."
        />
      </CardContent>
    </Card>
  );
}
