import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  ArrowRight, CheckCircle2, ExternalLink, Eye, EyeOff, Lock, RotateCw,
  ShieldAlert, ShieldCheck, Star, Trash2,
} from "lucide-react";

import { api } from "@/lib/api";
import { cn, fmtInr } from "@/lib/utils";

import {
  Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle,
} from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/Tooltip";
import { DataTable, type Column } from "@/components/ui/DataTable";
import {
  Dialog, DialogContent, DialogClose,
} from "@/components/ui/Dialog";
import { Input } from "@/components/ui/Input";
import { FreshnessIndicator } from "@/components/ui/FreshnessIndicator";

import { BROKER_CATALOG, getBrokerSpec, type BrokerSpec } from "./brokerCatalog";
import { TradingViewSection } from "./TradingViewSection";

// ─── API types ────────────────────────────────────────────────────────

interface BrokerLink {
  id: string;
  broker_name: string;
  display_name: string;
  owner: string;
  status: "active" | "expired" | "disabled" | "errored";
  last_refreshed_at: string | null;
  last_error: string;
  credential_meta: Record<string, string>;
  is_default: boolean;
  last_snapshot_at: string | null;
  last_snapshot_ok: boolean | null;
}

interface CombinedBroker {
  link_id: string;
  broker_name: string;
  display_name: string;
  status: BrokerLink["status"];
  is_default: boolean;
  fetched_at: string | null;
  age_seconds: number | null;
  ok: boolean | null;
  error: string;
  positions: Array<{ symbol: string; quantity: number; pnl: number; mtm: number }>;
  holdings: Array<{ symbol: string; quantity: number; pnl: number }>;
  margin: { available_cash?: number; used?: number; total?: number };
}

interface CombinedResponse {
  fetched_at: string | null;
  stalest_age_seconds: number | null;
  brokers: CombinedBroker[];
  totals: {
    positions_count: number;
    holdings_count: number;
    open_pnl: number;
    available_cash: number;
    used_margin: number;
  };
}

// ─── Hooks ────────────────────────────────────────────────────────────

function useAvailableBrokers() {
  return useQuery({
    queryKey: ["broker-available"],
    queryFn: () => api.get<{ brokers: string[] }>("/brokers/available/").then((r) => r.data.brokers),
    staleTime: 5 * 60_000,
  });
}

function useLinks() {
  return useQuery({
    queryKey: ["broker-links"],
    queryFn: () => api.get<BrokerLink[]>("/brokers/").then((r) => r.data),
  });
}

function useCombined() {
  return useQuery({
    queryKey: ["broker-combined"],
    queryFn: () => api.get<CombinedResponse>("/brokers/combined/positions/").then((r) => r.data),
    refetchInterval: 15_000,
  });
}

// ─── Page ─────────────────────────────────────────────────────────────

export function BrokerLinkPage() {
  const qc = useQueryClient();
  const { data: available = [] } = useAvailableBrokers();
  const { data: links = [], isLoading: linksLoading } = useLinks();
  const { data: combined, dataUpdatedAt: combinedUpdatedAt } = useCombined();

  const [connectBroker, setConnectBroker] = useState<BrokerSpec | null>(null);

  // One-shot OAuth callback feedback. The backend bounces the browser to
  // /broker?connected=fyers or /broker?error=fyers&reason=... after the
  // redirect-based connect flow. Surface as a toast and clear the URL.
  useEffect(() => {
    const url = new URL(window.location.href);
    const connected = url.searchParams.get("connected");
    const error = url.searchParams.get("error");
    if (connected) {
      toast.success(`${BROKER_CATALOG[connected]?.label ?? connected} connected`);
      qc.invalidateQueries({ queryKey: ["broker-links"] });
      qc.invalidateQueries({ queryKey: ["broker-combined"] });
    } else if (error) {
      const reason = url.searchParams.get("reason") || "Connect failed";
      toast.error(`${BROKER_CATALOG[error]?.label ?? error}: ${reason}`);
    }
    if (connected || error) {
      url.searchParams.delete("connected");
      url.searchParams.delete("error");
      url.searchParams.delete("reason");
      window.history.replaceState({}, "", url.toString());
    }
  }, [qc]);

  const refreshOne = useMutation({
    mutationFn: (id: string) => api.post(`/brokers/${id}/refresh/`),
    onSuccess: () => {
      toast.success("Refreshed");
      qc.invalidateQueries({ queryKey: ["broker-links"] });
      qc.invalidateQueries({ queryKey: ["broker-combined"] });
    },
    onError: (e: any) =>
      toast.error(e?.response?.data?.detail ?? "Refresh failed"),
  });

  const refreshAll = useMutation({
    mutationFn: () =>
      api.get<CombinedResponse>("/brokers/combined/positions/?refresh=1").then((r) => r.data),
    onSuccess: () => {
      toast.success("All brokers refreshed");
      qc.invalidateQueries({ queryKey: ["broker-links"] });
      qc.invalidateQueries({ queryKey: ["broker-combined"] });
    },
    onError: () => toast.error("Refresh failed"),
  });

  const setDefault = useMutation({
    mutationFn: (id: string) => api.post(`/brokers/${id}/set-default/`),
    onSuccess: () => {
      toast.success("Default account updated");
      qc.invalidateQueries({ queryKey: ["broker-links"] });
    },
  });

  const unlink = useMutation({
    mutationFn: (id: string) => api.delete(`/brokers/${id}/`),
    onSuccess: () => {
      toast.success("Broker unlinked");
      qc.invalidateQueries({ queryKey: ["broker-links"] });
      qc.invalidateQueries({ queryKey: ["broker-combined"] });
    },
  });

  const linkColumns = useMemo<Column<BrokerLink>[]>(() => [
    {
      key: "broker_name", header: "Broker",
      render: (l) => (
        <div className="flex items-center gap-2">
          <span className="font-medium text-fg">{BROKER_CATALOG[l.broker_name]?.label ?? l.broker_name}</span>
          {l.is_default && (
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="inline-flex"><Badge tone="brand" dot>Default</Badge></span>
              </TooltipTrigger>
              <TooltipContent>New orders route to this account.</TooltipContent>
            </Tooltip>
          )}
        </div>
      ),
    },
    {
      key: "display_name", header: "Alias",
      render: (l) => (
        <span className="text-body-sm text-fg-muted">
          {l.display_name || l.credential_meta?.account_alias || "—"}
        </span>
      ),
    },
    {
      key: "status", header: "Status",
      render: (l) => <StatusPill link={l} />,
    },
    {
      key: "last_snapshot_at", header: "Last snapshot",
      render: (l) => (
        <FreshnessIndicator
          timestamp={l.last_snapshot_at}
          freshMs={60_000}
          staleMs={5 * 60_000}
          label="Snapshot"
          variant="muted"
        />
      ),
    },
    {
      key: "actions", header: "", align: "right",
      render: (l) => (
        <div className="flex items-center justify-end gap-1">
          <Button
            variant="ghost" size="sm"
            onClick={() => refreshOne.mutate(l.id)}
            loading={refreshOne.isPending && refreshOne.variables === l.id}
            leading={<RotateCw className="h-4 w-4" />}
          >Refresh</Button>
          {!l.is_default && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost" size="sm"
                  onClick={() => setDefault.mutate(l.id)}
                  loading={setDefault.isPending && setDefault.variables === l.id}
                  leading={<Star className="h-4 w-4" />}
                >Default</Button>
              </TooltipTrigger>
              <TooltipContent>Route new orders to this account.</TooltipContent>
            </Tooltip>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost" size="sm"
                onClick={() => {
                  if (window.confirm(`Unlink ${l.display_name || l.broker_name}?`)) {
                    unlink.mutate(l.id);
                  }
                }}
                loading={unlink.isPending && unlink.variables === l.id}
                aria-label="Unlink broker"
              ><Trash2 className="h-4 w-4 text-danger" /></Button>
            </TooltipTrigger>
            <TooltipContent>Remove the link (revokes encrypted credentials).</TooltipContent>
          </Tooltip>
        </div>
      ),
    },
  ], [refreshOne, setDefault, unlink]);

  // Catalog cards — show every backend-available broker, dimming those
  // already linked so the operator knows where the credential blob lives.
  const catalog = useMemo(() => {
    const linkedNames = new Set(links.map((l) => l.broker_name));
    return available
      .map((name) => ({ spec: getBrokerSpec(name), linked: linkedNames.has(name) }))
      .sort((a, b) => Number(b.spec.recommended) - Number(a.spec.recommended));
  }, [available, links]);

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      <header>
        <p className="text-caption uppercase tracking-wider text-fg-subtle">Broker</p>
        <h1 className="text-h1 text-fg">Connected accounts</h1>
        <p className="text-body-sm text-fg-muted mt-1 max-w-2xl">
          Link as many broker accounts as you want — Angel One, Zerodha, Fyers.
          Positions, holdings, and margin from every linked account are aggregated
          below and refreshed every 30 seconds during market hours.
        </p>
      </header>

      {/* ── Catalog ─────────────────────────────────────────────────── */}
      <section aria-labelledby="catalog-h">
        <h2 id="catalog-h" className="sr-only">Available brokers</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {catalog.map(({ spec, linked }) => (
            <CatalogCard
              key={spec.name}
              spec={spec}
              linked={linked}
              onConnect={() => setConnectBroker(spec)}
            />
          ))}
        </div>
      </section>

      {/* ── Combined positions / totals ─────────────────────────────── */}
      <section aria-labelledby="combined-h" className="space-y-3">
        <div className="flex items-end justify-between gap-3 flex-wrap">
          <div>
            <h2 id="combined-h" className="text-h2 text-fg">Combined positions</h2>
            <p className="text-body-sm text-fg-muted">
              Aggregated across every linked account. Auto-refreshes every 15s; click below to force a live broker call.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <FreshnessIndicator
              timestamp={combined?.fetched_at ?? combinedUpdatedAt}
              freshMs={30_000}
              staleMs={120_000}
              label="Updated"
            />
            <Button
              size="sm"
              variant="primary"
              onClick={() => refreshAll.mutate()}
              loading={refreshAll.isPending}
              leading={<RotateCw className="h-4 w-4" />}
            >Refresh all</Button>
          </div>
        </div>

        <PortfolioOverviewCard data={combined} />

        <UnifiedHoldingsTable data={combined} />
        <UnifiedPositionsTable data={combined} />

        <details className="group">
          <summary className="cursor-pointer text-body-sm text-fg-muted hover:text-fg inline-flex items-center gap-1">
            <span className="group-open:hidden">Show per-broker breakdown</span>
            <span className="hidden group-open:inline">Hide per-broker breakdown</span>
          </summary>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mt-3">
            {(combined?.brokers ?? []).map((b) => (
              <BrokerSnapshotCard key={b.link_id} broker={b} />
            ))}
            {combined && combined.brokers.length === 0 && (
              <Card className="md:col-span-3">
                <CardContent className="py-10 text-center text-body-sm text-fg-muted">
                  Link a broker above to see combined positions here.
                </CardContent>
              </Card>
            )}
          </div>
        </details>
      </section>

      {/* ── Linked accounts table ───────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>Linked accounts</CardTitle>
          <CardDescription>Active broker sessions for this tenant.</CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <DataTable<BrokerLink>
            columns={linkColumns}
            rows={links}
            loading={linksLoading}
            rowKey={(l) => l.id}
            emptyTitle="No broker linked yet"
            emptyDescription="Pick a broker above and connect to start aggregating positions."
          />
        </CardContent>
      </Card>

      {/* ── TradingView signals ─────────────────────────────────────── */}
      <TradingViewSection />

      {/* ── Connect dialog ──────────────────────────────────────────── */}
      <Dialog open={!!connectBroker} onOpenChange={(open) => !open && setConnectBroker(null)}>
        <DialogContent>
          {connectBroker && (
            <ConnectForm
              spec={connectBroker}
              onClose={() => setConnectBroker(null)}
              onConnected={() => {
                setConnectBroker(null);
                qc.invalidateQueries({ queryKey: ["broker-links"] });
                qc.invalidateQueries({ queryKey: ["broker-combined"] });
              }}
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

// ─── Catalog card ─────────────────────────────────────────────────────

function CatalogCard({
  spec, linked, onConnect,
}: { spec: BrokerSpec; linked: boolean; onConnect: () => void }) {
  return (
    <Card className={cn("h-full flex flex-col", linked && "opacity-95")}>
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2">
              {spec.label}
              {linked && <Badge tone="success" dot>Linked</Badge>}
            </CardTitle>
            <CardDescription>{spec.blurb}</CardDescription>
          </div>
          {spec.recommended && <Badge tone="brand">Recommended</Badge>}
        </div>
      </CardHeader>
      <CardContent className="flex-1">
        <ul className="text-body-sm text-fg-muted space-y-1.5">
          <li className="flex gap-2">
            <ShieldCheck className="h-4 w-4 text-accent shrink-0 mt-0.5" aria-hidden />
            Pre-flight auth before persisting
          </li>
          <li className="flex gap-2">
            <Lock className="h-4 w-4 text-accent shrink-0 mt-0.5" aria-hidden />
            Fernet-encrypted at rest
          </li>
          <li className="flex gap-2">
            <CheckCircle2 className="h-4 w-4 text-accent shrink-0 mt-0.5" aria-hidden />
            {spec.dailyTokenRefresh ? "Daily token re-login required" : "Long-lived session (24h)"}
          </li>
        </ul>
      </CardContent>
      <CardFooter className="flex items-center justify-between">
        {spec.setupUrl ? (
          <a
            href={spec.setupUrl}
            target="_blank" rel="noreferrer"
            className="text-caption text-fg-subtle hover:text-fg inline-flex items-center gap-1"
          >
            Get keys <ExternalLink className="h-3 w-3" aria-hidden />
          </a>
        ) : <span />}
        <Button
          variant={spec.recommended ? "primary" : "secondary"}
          size="sm"
          onClick={onConnect}
          trailing={<ArrowRight className="h-3.5 w-3.5" />}
        >
          {linked ? "Add another" : "Connect"}
        </Button>
      </CardFooter>
    </Card>
  );
}

// ─── Status pill ──────────────────────────────────────────────────────

function StatusPill({ link }: { link: BrokerLink }) {
  if (link.status === "active") return <Badge tone="success" dot>Active</Badge>;
  if (link.status === "expired") return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex"><Badge tone="warning" dot>Expired</Badge></span>
      </TooltipTrigger>
      <TooltipContent>{link.last_error || "Session expired — click Refresh to renew."}</TooltipContent>
    </Tooltip>
  );
  if (link.status === "errored") return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex"><Badge tone="danger" dot><ShieldAlert className="h-3 w-3 mr-0.5" aria-hidden />Errored</Badge></span>
      </TooltipTrigger>
      <TooltipContent>{link.last_error || "Last broker call failed."}</TooltipContent>
    </Tooltip>
  );
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex"><Badge tone="danger" dot>Disabled</Badge></span>
      </TooltipTrigger>
      <TooltipContent>Disabled by admin. New orders will be rejected.</TooltipContent>
    </Tooltip>
  );
}

// ─── Unified positions / holdings tables (across every linked broker) ──

interface FlatRow {
  broker_name: string;
  broker_label: string;
  display_name: string;
  symbol: string;
  exchange?: string;
  product?: string;
  quantity: number;
  avg_price: number;
  last_price: number;
  pnl: number;
}

function flattenPositions(data: CombinedResponse | undefined): FlatRow[] {
  if (!data) return [];
  const out: FlatRow[] = [];
  for (const b of data.brokers) {
    const label = BROKER_CATALOG[b.broker_name]?.label ?? b.broker_name;
    for (const p of b.positions as any[]) {
      out.push({
        broker_name: b.broker_name,
        broker_label: label,
        display_name: b.display_name || b.broker_name,
        symbol: p.symbol,
        exchange: p.exchange,
        product: p.product,
        quantity: p.quantity,
        avg_price: p.avg_price,
        last_price: p.last_price,
        pnl: (p.pnl ?? 0) + (p.mtm ?? 0),
      });
    }
  }
  return out.sort((a, b) => Math.abs(b.pnl) - Math.abs(a.pnl));
}

function flattenHoldings(data: CombinedResponse | undefined): FlatRow[] {
  if (!data) return [];
  const out: FlatRow[] = [];
  for (const b of data.brokers) {
    const label = BROKER_CATALOG[b.broker_name]?.label ?? b.broker_name;
    for (const h of b.holdings as any[]) {
      out.push({
        broker_name: b.broker_name,
        broker_label: label,
        display_name: b.display_name || b.broker_name,
        symbol: h.symbol,
        exchange: h.exchange,
        quantity: h.quantity,
        avg_price: h.avg_price,
        last_price: h.last_price,
        pnl: h.pnl ?? 0,
      });
    }
  }
  return out.sort((a, b) => b.quantity * b.last_price - a.quantity * a.last_price);
}

function UnifiedPositionsTable({ data }: { data: CombinedResponse | undefined }) {
  const rows = useMemo(() => flattenPositions(data), [data]);
  if (rows.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-h3">Open positions</CardTitle>
          <CardDescription>Open intraday + carryforward across every linked account.</CardDescription>
        </CardHeader>
        <CardContent className="py-6 text-center text-body-sm text-fg-muted">
          No open positions at any linked broker right now.
        </CardContent>
      </Card>
    );
  }
  const columns: Column<FlatRow>[] = [
    {
      key: "broker", header: "Account",
      render: (r) => (
        <span className="text-caption text-fg-muted">{r.broker_label} · {r.display_name}</span>
      ),
    },
    {
      key: "symbol", header: "Symbol",
      render: (r) => (
        <div className="font-medium text-fg">
          {r.symbol}
          {r.exchange && <span className="ml-1.5 text-caption text-fg-subtle">{r.exchange}</span>}
        </div>
      ),
    },
    {
      key: "product", header: "Type",
      render: (r) => <span className="text-caption text-fg-muted">{r.product ?? "—"}</span>,
    },
    {
      key: "quantity", header: "Qty", align: "right",
      render: (r) => (
        <span className={cn("tabular-nums", r.quantity < 0 && "text-warning")}>{r.quantity}</span>
      ),
    },
    {
      key: "avg_price", header: "Avg", align: "right",
      render: (r) => <span className="tabular-nums text-fg-muted">{fmtInr(r.avg_price)}</span>,
    },
    {
      key: "last_price", header: "LTP", align: "right",
      render: (r) => <span className="tabular-nums">{fmtInr(r.last_price)}</span>,
    },
    {
      key: "pnl", header: "P&L", align: "right",
      render: (r) => (
        <span className={cn("tabular-nums font-medium", r.pnl >= 0 ? "text-success" : "text-danger")}>
          {fmtInr(r.pnl)}
        </span>
      ),
    },
  ];
  return (
    <Card>
      <CardHeader className="flex-row items-end justify-between gap-2 space-y-0">
        <div>
          <CardTitle className="text-h3">Open positions</CardTitle>
          <CardDescription>{rows.length} open across {new Set(rows.map((r) => r.broker_name)).size} accounts.</CardDescription>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <DataTable<FlatRow> columns={columns} rows={rows} rowKey={(r, i) => `${r.broker_name}:${r.symbol}:${i}`} />
      </CardContent>
    </Card>
  );
}

function UnifiedHoldingsTable({ data }: { data: CombinedResponse | undefined }) {
  const rows = useMemo(() => flattenHoldings(data), [data]);
  if (rows.length === 0) return null;
  const columns: Column<FlatRow>[] = [
    {
      key: "broker", header: "Account",
      render: (r) => (
        <span className="text-caption text-fg-muted">{r.broker_label} · {r.display_name}</span>
      ),
    },
    {
      key: "symbol", header: "Symbol",
      render: (r) => (
        <div className="font-medium text-fg">
          {r.symbol}
          {r.exchange && <span className="ml-1.5 text-caption text-fg-subtle">{r.exchange}</span>}
        </div>
      ),
    },
    {
      key: "quantity", header: "Qty", align: "right",
      render: (r) => <span className="tabular-nums">{r.quantity}</span>,
    },
    {
      key: "avg_price", header: "Avg cost", align: "right",
      render: (r) => <span className="tabular-nums text-fg-muted">{fmtInr(r.avg_price)}</span>,
    },
    {
      key: "last_price", header: "LTP", align: "right",
      render: (r) => <span className="tabular-nums">{fmtInr(r.last_price)}</span>,
    },
    {
      key: "value", header: "Value", align: "right",
      render: (r) => <span className="tabular-nums">{fmtInr(r.quantity * r.last_price)}</span>,
    },
    {
      key: "pnl", header: "P&L", align: "right",
      render: (r) => (
        <span className={cn("tabular-nums font-medium", r.pnl >= 0 ? "text-success" : "text-danger")}>
          {fmtInr(r.pnl)}
        </span>
      ),
    },
  ];
  return (
    <Card>
      <CardHeader className="flex-row items-end justify-between gap-2 space-y-0">
        <div>
          <CardTitle className="text-h3">Holdings</CardTitle>
          <CardDescription>{rows.length} settled holdings across {new Set(rows.map((r) => r.broker_name)).size} demat accounts.</CardDescription>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <DataTable<FlatRow> columns={columns} rows={rows} rowKey={(r, i) => `${r.broker_name}:${r.symbol}:${i}`} />
      </CardContent>
    </Card>
  );
}

// ─── Portfolio overview (aggregated stats) ────────────────────────────

interface PortfolioStats {
  totalValue: number;             // cash + holdings_value + positions_mtm
  cash: number;
  used: number;
  holdingsValue: number;
  holdingsCost: number;
  holdingsPnl: number;
  holdingsPnlPct: number;
  positionsValue: number;
  positionsPnl: number;
  exposurePct: number;            // used / (cash + used)
  brokerCount: number;
  positionsCount: number;
  holdingsCount: number;
  perBroker: Array<{
    label: string; displayName: string; value: number; share: number; pnl: number;
  }>;
}

function computeStats(data: CombinedResponse | undefined): PortfolioStats {
  if (!data) return EMPTY_STATS;

  let cash = 0, used = 0;
  let hValue = 0, hCost = 0, hPnl = 0;
  let pValue = 0, pPnl = 0;
  let positionsCount = 0, holdingsCount = 0;
  const perBrokerRaw: Array<{ label: string; displayName: string; value: number; pnl: number }> = [];

  for (const b of data.brokers) {
    if (!b.ok) continue;
    const bLabel = BROKER_CATALOG[b.broker_name]?.label ?? b.broker_name;
    const bCash = b.margin?.available_cash ?? 0;
    const bUsed = b.margin?.used ?? 0;
    let bHoldingsVal = 0;
    let bPnl = 0;
    for (const h of b.holdings as any[]) {
      const v = (h.quantity ?? 0) * (h.last_price ?? 0);
      bHoldingsVal += v;
      hCost += (h.quantity ?? 0) * (h.avg_price ?? 0);
      hPnl += h.pnl ?? 0;
      bPnl += h.pnl ?? 0;
      holdingsCount += 1;
    }
    for (const p of b.positions as any[]) {
      const v = Math.abs((p.quantity ?? 0) * (p.last_price ?? 0));
      pValue += v;
      const ppnl = (p.pnl ?? 0) + (p.mtm ?? 0);
      pPnl += ppnl;
      bPnl += ppnl;
      positionsCount += 1;
    }
    hValue += bHoldingsVal;
    cash += bCash;
    used += bUsed;
    perBrokerRaw.push({
      label: bLabel,
      displayName: b.display_name || b.broker_name,
      value: bCash + bUsed + bHoldingsVal,
      pnl: bPnl,
    });
  }

  const totalValue = cash + used + hValue;
  const perBroker = perBrokerRaw
    .map((b) => ({ ...b, share: totalValue > 0 ? b.value / totalValue : 0 }))
    .sort((a, b) => b.value - a.value);

  return {
    totalValue,
    cash,
    used,
    holdingsValue: hValue,
    holdingsCost: hCost,
    holdingsPnl: hPnl,
    holdingsPnlPct: hCost > 0 ? (hPnl / hCost) * 100 : 0,
    positionsValue: pValue,
    positionsPnl: pPnl,
    exposurePct: cash + used > 0 ? (used / (cash + used)) * 100 : 0,
    brokerCount: perBroker.length,
    positionsCount,
    holdingsCount,
    perBroker,
  };
}

const EMPTY_STATS: PortfolioStats = {
  totalValue: 0, cash: 0, used: 0,
  holdingsValue: 0, holdingsCost: 0, holdingsPnl: 0, holdingsPnlPct: 0,
  positionsValue: 0, positionsPnl: 0,
  exposurePct: 0, brokerCount: 0, positionsCount: 0, holdingsCount: 0,
  perBroker: [],
};

function PortfolioOverviewCard({ data }: { data: CombinedResponse | undefined }) {
  const s = useMemo(() => computeStats(data), [data]);
  return (
    <Card>
      <CardHeader>
        <div className="flex items-end justify-between gap-3 flex-wrap">
          <div>
            <CardTitle className="text-h3">Portfolio overview</CardTitle>
            <CardDescription>
              Aggregated across {s.brokerCount} {s.brokerCount === 1 ? "account" : "accounts"} ·
              {" "}{s.positionsCount} open positions · {s.holdingsCount} holdings
            </CardDescription>
          </div>
          <div className="text-right">
            <div className="text-caption uppercase tracking-wider text-fg-subtle">Total portfolio value</div>
            <div className="text-h1 tabular-nums text-fg">{fmtInr(s.totalValue)}</div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Primary KPIs */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-border rounded-sm overflow-hidden">
          <StatTile
            label="Holdings value"
            value={fmtInr(s.holdingsValue)}
            sub={s.holdingsCount > 0 ? `${s.holdingsCount} symbols` : undefined}
          />
          <StatTile
            label="Holdings P&L"
            value={fmtInr(s.holdingsPnl)}
            valueTone={s.holdingsPnl >= 0 ? "success" : "danger"}
            sub={s.holdingsCost > 0 ? `${s.holdingsPnl >= 0 ? "+" : ""}${s.holdingsPnlPct.toFixed(2)}% on ${fmtInr(s.holdingsCost, { compact: true })} cost` : undefined}
          />
          <StatTile
            label="Open P&L"
            value={fmtInr(s.positionsPnl)}
            valueTone={s.positionsPnl >= 0 ? "success" : "danger"}
            sub={s.positionsCount > 0 ? `${s.positionsCount} live positions` : "no live positions"}
          />
          <StatTile
            label="Available cash"
            value={fmtInr(s.cash)}
            sub={s.used > 0 ? `${fmtInr(s.used, { compact: true })} used` : undefined}
          />
        </div>

        {/* Secondary KPIs */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-body-sm">
          <SecondaryStat label="Used margin" value={fmtInr(s.used)} />
          <SecondaryStat label="Exposure" value={`${s.exposurePct.toFixed(1)}%`} />
          <SecondaryStat label="Position notional" value={fmtInr(s.positionsValue)} />
          <SecondaryStat label="Cost basis (holdings)" value={fmtInr(s.holdingsCost)} />
        </div>

        {/* Per-broker contribution */}
        {s.perBroker.length > 1 && (
          <div className="space-y-1.5 pt-2 border-t border-border">
            <div className="text-caption uppercase tracking-wider text-fg-subtle">
              By account
            </div>
            {s.perBroker.map((b) => (
              <div key={b.label + b.displayName} className="text-body-sm">
                <div className="flex justify-between gap-2 mb-0.5">
                  <span className="text-fg-muted truncate">{b.label} · {b.displayName}</span>
                  <span className="tabular-nums text-fg">
                    {fmtInr(b.value)}{" "}
                    <span className="text-fg-subtle">({(b.share * 100).toFixed(1)}%)</span>
                  </span>
                </div>
                <div className="h-1.5 bg-surface-2 rounded-xs overflow-hidden">
                  <div
                    className={cn("h-full", b.pnl >= 0 ? "bg-success" : "bg-danger")}
                    style={{ width: `${Math.max(2, b.share * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function StatTile({
  label, value, sub, valueTone,
}: {
  label: string; value: string; sub?: string;
  valueTone?: "success" | "danger";
}) {
  return (
    <div className="bg-surface p-3">
      <div className="text-caption uppercase tracking-wider text-fg-subtle">{label}</div>
      <div className={cn(
        "text-h2 mt-1 tabular-nums",
        valueTone === "success" && "text-success",
        valueTone === "danger" && "text-danger",
        !valueTone && "text-fg",
      )}>{value}</div>
      {sub && <div className="mt-0.5 text-caption text-fg-muted">{sub}</div>}
    </div>
  );
}

function SecondaryStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-fg-muted">{label}</span>
      <span className="tabular-nums text-fg">{value}</span>
    </div>
  );
}

// ─── Per-broker snapshot card ─────────────────────────────────────────

function BrokerSnapshotCard({ broker }: { broker: CombinedBroker }) {
  const spec = BROKER_CATALOG[broker.broker_name];
  const cash = broker.margin?.available_cash ?? 0;
  const used = broker.margin?.used ?? 0;
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div>
            <CardTitle className="text-h3 flex items-center gap-2">
              {spec?.label ?? broker.broker_name}
              {broker.is_default && <Badge tone="brand" dot>Default</Badge>}
            </CardTitle>
            <CardDescription>{broker.display_name || broker.broker_name}</CardDescription>
          </div>
          {broker.ok === false ? (
            <Badge tone="danger" dot>Stale</Badge>
          ) : (
            <FreshnessIndicator
              timestamp={broker.fetched_at}
              freshMs={30_000}
              staleMs={120_000}
              variant="badge"
              label=""
            />
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        {broker.error && (
          <div className="text-caption text-danger">{broker.error}</div>
        )}
        <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-body-sm">
          <Row label="Open positions" value={broker.positions.length.toString()} />
          <Row label="Holdings" value={broker.holdings.length.toString()} />
          <Row label="Cash" value={fmtInr(cash)} />
          <Row label="Used" value={fmtInr(used)} />
        </div>
        {broker.positions.length > 0 && (
          <details className="mt-2">
            <summary className="cursor-pointer text-caption text-fg-muted hover:text-fg">
              Show {broker.positions.length} positions
            </summary>
            <ul className="mt-2 space-y-1 text-caption">
              {broker.positions.slice(0, 8).map((p, i) => (
                <li key={i} className="flex justify-between gap-2">
                  <span className="font-mono">{p.symbol}</span>
                  <span className="tabular-nums text-fg-muted">qty {p.quantity}</span>
                  <span className={cn(
                    "tabular-nums",
                    p.pnl >= 0 ? "text-success" : "text-danger",
                  )}>{fmtInr(p.pnl)}</span>
                </li>
              ))}
              {broker.positions.length > 8 && (
                <li className="text-fg-subtle">+ {broker.positions.length - 8} more</li>
              )}
            </ul>
          </details>
        )}
      </CardContent>
    </Card>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <>
      <span className="text-fg-muted">{label}</span>
      <span className="text-right tabular-nums text-fg">{value}</span>
    </>
  );
}

// ─── Connect form (dialog body) ───────────────────────────────────────

function ConnectForm({
  spec, onClose, onConnected,
}: { spec: BrokerSpec; onClose: () => void; onConnected: () => void }) {
  const isOAuth = !!spec.oauth;
  const fieldList = isOAuth ? spec.oauth!.handshakeFields : spec.fields;

  const [values, setValues] = useState<Record<string, string>>(() => {
    const d: Record<string, string> = {};
    if (spec.name === "paper") d.mode = "paper";
    return d;
  });
  const [meta, setMeta] = useState<Record<string, string>>({});
  const [reveal, setReveal] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);

  // Paste-the-token path (Angel One, Paper, future static brokers)
  const directConnect = useMutation({
    mutationFn: () => api.post(`/brokers/${spec.name}/connect/`, {
      credentials: values,
      meta,
      display_name: meta.account_alias || spec.label,
    }),
    onSuccess: () => {
      toast.success(`${spec.label} connected`);
      onConnected();
    },
    onError: (e: any) => {
      const detail = e?.response?.data?.detail ?? e?.response?.data ?? "Could not connect";
      setError(typeof detail === "string" ? detail : JSON.stringify(detail));
    },
  });

  // OAuth-redirect path (Zerodha, Fyers)
  const oauthStart = useMutation({
    mutationFn: () => api.post<{ login_url: string }>(`/brokers/${spec.name}/oauth/start/`, {
      handshake: values,
      meta,
      display_name: meta.account_alias || spec.label,
    }).then((r) => r.data),
    onSuccess: (data) => {
      // Hand off the browser. The broker will redirect back to
      // /api/v1/brokers/{name}/oauth/callback/ which bounces to /broker.
      window.location.assign(data.login_url);
    },
    onError: (e: any) => {
      const detail = e?.response?.data?.detail ?? "Could not start OAuth flow";
      setError(typeof detail === "string" ? detail : JSON.stringify(detail));
    },
  });

  const valid = fieldList.every((f) => !f.required || (values[f.key]?.trim().length ?? 0) > 0);
  const submitting = directConnect.isPending || oauthStart.isPending;

  return (
    <form
      className="space-y-4"
      onSubmit={(e) => {
        e.preventDefault();
        setError(null);
        (isOAuth ? oauthStart : directConnect).mutate();
      }}
    >
      <header>
        <h2 className="text-h2 text-fg">Connect {spec.label}</h2>
        <p className="text-body-sm text-fg-muted mt-1">{spec.blurb}</p>
        {spec.setupUrl && (
          <a
            href={spec.setupUrl} target="_blank" rel="noreferrer"
            className="text-caption text-accent inline-flex items-center gap-1 mt-1"
          >Where to get keys <ExternalLink className="h-3 w-3" aria-hidden /></a>
        )}
      </header>

      {isOAuth && (
        <div className="rounded-sm border border-accent/40 bg-accent/5 px-3 py-2 text-caption text-fg">
          <div className="font-medium text-accent mb-1">Before you continue</div>
          <p className="text-fg-muted">{spec.oauth!.redirectHelp}</p>
          <p className="text-fg-muted mt-1">
            We&rsquo;ll send you to {spec.label} to log in. After you authorise,
            you&rsquo;ll be bounced back here automatically.
          </p>
        </div>
      )}

      {spec.dailyTokenRefresh && (
        <div className="rounded-sm border border-warning/40 bg-warning/5 px-3 py-2 text-caption text-warning">
          Daily access-token re-login required (expires ~6 AM IST). You&rsquo;ll
          click &ldquo;Re-login&rdquo; on the brokers page each trading morning.
        </div>
      )}

      <div className="space-y-3">
        {fieldList.map((f) => {
          const isSecret = f.kind === "secret" || f.kind === "password";
          const inputType = isSecret && !reveal[f.key] ? "password" : "text";
          return (
            <Input
              key={f.key}
              label={f.label}
              type={inputType}
              required={f.required}
              placeholder={f.placeholder}
              hint={f.hint}
              value={values[f.key] ?? ""}
              onChange={(e) => setValues((v) => ({ ...v, [f.key]: e.target.value }))}
              trailing={isSecret ? (
                <button
                  type="button"
                  aria-label={reveal[f.key] ? "Hide" : "Reveal"}
                  onClick={() => setReveal((r) => ({ ...r, [f.key]: !r[f.key] }))}
                  className="text-fg-muted hover:text-fg"
                >{reveal[f.key] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}</button>
              ) : undefined}
            />
          );
        })}

        {(spec.metaFields ?? []).map((m) => (
          <Input
            key={m.key}
            label={m.label}
            optional
            placeholder={m.placeholder}
            hint={m.hint}
            value={meta[m.key] ?? ""}
            onChange={(e) => setMeta((v) => ({ ...v, [m.key]: e.target.value }))}
          />
        ))}
      </div>

      {error && (
        <div role="alert" className="rounded-sm border border-danger/40 bg-danger/5 px-3 py-2 text-caption text-danger">
          {error}
        </div>
      )}

      <footer className="flex items-center justify-end gap-2 pt-2 border-t border-border">
        <DialogClose asChild>
          <Button variant="secondary" size="sm" type="button" onClick={onClose}>Cancel</Button>
        </DialogClose>
        <Button
          type="submit" size="sm" variant="primary"
          loading={submitting} disabled={!valid}
          trailing={isOAuth ? <ArrowRight className="h-3.5 w-3.5" /> : undefined}
        >
          {isOAuth ? `Continue to ${spec.label}` : "Test & save"}
        </Button>
      </footer>
    </form>
  );
}
