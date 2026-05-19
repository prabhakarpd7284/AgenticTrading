/**
 * TradingViewSection — sits alongside the broker catalog on BrokerLinkPage.
 *
 * Renders the operator's configured TradingView webhook URLs and lets them
 * connect new ones, rotate the URL secret, configure auto-fire, and inspect
 * recent received alerts. The underlying contract is:
 *
 *   POST /api/v1/webhooks/tradingview/<secret>/   ← TradingView posts here
 *
 *   /api/v1/notifications/tradingview/             ← CRUD + recent feed
 *
 * The webhook URL is read-once-then-display: rotating regenerates the
 * secret and invalidates the old URL.
 */
import * as React from "react";
import { toast } from "sonner";
import {
  AlertCircle, Copy, ExternalLink, RefreshCw, Settings, Trash2, Zap,
} from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/Card";
import {
  Dialog, DialogContent, DialogDescription, DialogTitle,
} from "@/components/ui/Dialog";
import { EmptyState } from "@/components/ui/EmptyState";
import { FreshnessIndicator } from "@/components/ui/FreshnessIndicator";
import { Input } from "@/components/ui/Input";
import { Skeleton } from "@/components/ui/Skeleton";
import {
  Sheet, SheetBody, SheetContent, SheetFooter, SheetHeader, SheetTitle,
} from "@/components/ui/Sheet";
import {
  type TradingViewLink, type TradingViewSignalRow,
  useCreateTradingViewLink, useDeleteTradingViewLink,
  useRotateTradingViewSecret, useTradingViewLinks, useTradingViewRecent,
  useWatchlists, useUpdateTradingViewLink,
} from "@/lib/v2";
import { cn, fmtRel } from "@/lib/utils";


export function TradingViewSection() {
  const { data: links = [], isLoading } = useTradingViewLinks();
  const [connectOpen, setConnectOpen] = React.useState(false);
  const [editingId, setEditingId] = React.useState<string | undefined>();

  const editing = links.find((l) => l.id === editingId);

  return (
    <section className="space-y-4">
      <Card>
        <CardHeader className="flex flex-row items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              TradingView signals
              <Badge tone="brand">Webhook</Badge>
            </CardTitle>
            <CardDescription>
              Turn TradingView alerts into AlphaDesk signals. Optionally auto-fire
              a workflow (RiskGuard always runs).
            </CardDescription>
          </div>
          <Button onClick={() => setConnectOpen(true)} size="sm">
            Connect TradingView
          </Button>
        </CardHeader>

        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              <Skeleton className="h-20 w-full" />
              <Skeleton className="h-20 w-full" />
            </div>
          ) : links.length === 0 ? (
            <EmptyState
              icon={<Zap />}
              title="No TradingView links yet"
              description="Connect a webhook URL and paste it into TradingView's alert dialog. Alerts will appear on the Now feed."
            />
          ) : (
            <ul className="space-y-3" role="list">
              {links.map((link) => (
                <LinkRow key={link.id} link={link} onEdit={() => setEditingId(link.id)} />
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <ConnectDialog open={connectOpen} onClose={() => setConnectOpen(false)} />
      <EditSheet
        link={editing}
        onClose={() => setEditingId(undefined)}
      />
    </section>
  );
}

/* =================================================================== */
/* Per-link row                                                         */
/* =================================================================== */

function LinkRow({ link, onEdit }: { link: TradingViewLink; onEdit: () => void }) {
  const rotate = useRotateTradingViewSecret();
  const remove = useDeleteTradingViewLink();
  const [revealed, setRevealed] = React.useState(false);

  const onCopy = async () => {
    await navigator.clipboard.writeText(link.webhook_url);
    toast.success("Webhook URL copied");
  };

  const onRotate = async () => {
    if (!window.confirm("Rotate the webhook secret? The old URL stops working immediately.")) return;
    await rotate.mutateAsync(link.id);
    setRevealed(true);
    toast.success("Webhook URL rotated — paste the new one into TradingView.");
  };

  const onDelete = async () => {
    if (!window.confirm(`Delete the "${link.display_name || "unnamed"}" link? This cannot be undone.`)) return;
    await remove.mutateAsync(link.id);
    toast.success("Link removed");
  };

  const masked = revealed
    ? link.webhook_url
    : link.webhook_url.replace(/tradingview\/[^/]+\//, "tradingview/••••••••/");

  return (
    <li className="rounded-md border border-border bg-surface-2 p-4 space-y-3">
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-body-sm font-semibold text-fg">
              {link.display_name || "Unnamed link"}
            </span>
            {!link.is_active && <Badge tone="neutral">Inactive</Badge>}
            {link.autofire_enabled && (
              <Badge tone="warning" title={`Fires ${link.default_strategy_name || "?"}`}>
                <Zap className="h-3 w-3 mr-1" aria-hidden />
                Auto-fire → {link.default_strategy_name || "(not set)"}
              </Badge>
            )}
          </div>
          <div className="text-caption text-fg-subtle mt-1">
            Received {link.receive_count} alert{link.receive_count === 1 ? "" : "s"}
            {link.last_received_at && (
              <> · last {fmtRel(link.last_received_at)} ago</>
            )}
          </div>
        </div>
        <FreshnessIndicator
          label="Last alert"
          timestamp={link.last_received_at}
          freshMs={60_000}
          staleMs={24 * 3_600_000}
        />
      </div>

      {link.last_error && (
        <div role="alert" className="flex items-start gap-2 text-caption text-warn">
          <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" aria-hidden />
          <span className="break-all">{link.last_error}</span>
        </div>
      )}

      <div className="flex items-center gap-2 rounded-sm bg-surface px-2 py-1.5 border border-border">
        <code className="flex-1 text-caption font-mono text-fg-muted truncate" title={link.webhook_url}>
          {masked}
        </code>
        <Button
          size="sm" variant="secondary"
          onClick={() => setRevealed((r) => !r)}
          aria-label={revealed ? "Hide URL" : "Reveal URL"}
        >
          {revealed ? "Hide" : "Reveal"}
        </Button>
        <Button size="sm" variant="secondary" onClick={onCopy} leading={<Copy className="h-3.5 w-3.5" />}>
          Copy
        </Button>
      </div>

      <div className="flex items-center justify-end gap-2">
        <Button
          size="sm" variant="secondary"
          onClick={onEdit}
          leading={<Settings className="h-3.5 w-3.5" />}
        >
          Settings
        </Button>
        <Button
          size="sm" variant="secondary"
          onClick={onRotate} loading={rotate.isPending}
          leading={<RefreshCw className="h-3.5 w-3.5" />}
        >
          Rotate secret
        </Button>
        <Button
          size="sm" variant="secondary"
          onClick={onDelete} loading={remove.isPending}
          leading={<Trash2 className="h-3.5 w-3.5" />}
        >
          Delete
        </Button>
      </div>
    </li>
  );
}

/* =================================================================== */
/* Connect dialog                                                       */
/* =================================================================== */

function ConnectDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [name, setName] = React.useState("");
  const create = useCreateTradingViewLink();
  const [created, setCreated] = React.useState<TradingViewLink | null>(null);

  React.useEffect(() => {
    if (!open) {
      // Reset after the close animation so the user doesn't see the
      // form contents flicker back on reopen.
      const t = setTimeout(() => { setName(""); setCreated(null); }, 250);
      return () => clearTimeout(t);
    }
  }, [open]);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const link = await create.mutateAsync({ display_name: name.trim() || "TradingView alert" });
    setCreated(link);
  };

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="w-[min(92vw,560px)] max-h-[85vh] overflow-auto">
        <DialogTitle>Connect TradingView</DialogTitle>
        <DialogDescription>
          Generates a private webhook URL. Paste it into TradingView's alert dialog under
          Notifications → Webhook URL.
        </DialogDescription>

        {!created ? (
          <form onSubmit={onSubmit} className="mt-4 space-y-3">
            <Input
              label="Name"
              hint="Operator-facing label, e.g. 'VWAP breakout'."
              placeholder="My TradingView strategy"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
            />
            <div className="flex items-center justify-end gap-2 pt-2">
              <Button type="button" variant="secondary" onClick={onClose}>Cancel</Button>
              <Button type="submit" loading={create.isPending}>Create webhook URL</Button>
            </div>
          </form>
        ) : (
          <CreatedView link={created} onDone={onClose} />
        )}
      </DialogContent>
    </Dialog>
  );
}

function CreatedView({ link, onDone }: { link: TradingViewLink; onDone: () => void }) {
  const [copied, setCopied] = React.useState(false);

  const onCopy = async () => {
    await navigator.clipboard.writeText(link.webhook_url);
    setCopied(true);
    toast.success("Webhook URL copied");
  };

  return (
    <div className="mt-4 space-y-4">
      <div className="rounded-md border border-pnl-up/30 bg-pnl-up/5 p-3 text-body-sm text-fg">
        Webhook created. <strong>Copy the URL now</strong> and paste it into TradingView —
        you can rotate it later, but the current value is your only proof of identity for incoming
        alerts.
      </div>

      <div>
        <div className="text-caption uppercase tracking-wider text-fg-subtle mb-1">Webhook URL</div>
        <div className="flex items-center gap-2 rounded-sm border border-border bg-surface-2 px-2 py-2">
          <code className="flex-1 text-caption font-mono text-fg break-all">{link.webhook_url}</code>
          <Button size="sm" onClick={onCopy} leading={<Copy className="h-3.5 w-3.5" />}>
            {copied ? "Copied" : "Copy"}
          </Button>
        </div>
      </div>

      <details className="text-caption text-fg-muted">
        <summary className="cursor-pointer text-fg-subtle hover:text-fg select-none">
          Sample TradingView alert message
        </summary>
        <pre className="mt-2 rounded-sm bg-surface-2 border border-border p-3 whitespace-pre-wrap break-all font-mono">
{`{
  "symbol": "{{ticker}}",
  "action": "{{strategy.order.action}}",
  "price": {{close}},
  "comment": "{{strategy.order.comment}}"
}`}
        </pre>
        <p className="mt-2">
          Plain text like <code>BUY HDFCBANK @ 1500</code> is also accepted as a fallback.
        </p>
      </details>

      <div className="flex items-center justify-end gap-2">
        <Button onClick={onDone} leading={<ExternalLink className="h-3.5 w-3.5" />}>Done</Button>
      </div>
    </div>
  );
}

/* =================================================================== */
/* Edit sheet (settings + recent alerts)                                */
/* =================================================================== */

function EditSheet({
  link, onClose,
}: { link: TradingViewLink | undefined; onClose: () => void }) {
  const update = useUpdateTradingViewLink();
  const { data: recent = [] } = useTradingViewRecent(link?.id);

  // Local mirror of the edited fields so toggling switches feels instant —
  // commit on every change via mutateAsync.
  const [draft, setDraft] = React.useState<TradingViewLink | undefined>(link);
  React.useEffect(() => setDraft(link), [link?.id, link?.updated_at]);

  if (!draft) {
    return (
      <Sheet open={false} onOpenChange={(o) => { if (!o) onClose(); }}>
        <SheetContent><div /></SheetContent>
      </Sheet>
    );
  }

  const patch = async (changes: Partial<TradingViewLink>) => {
    const next = { ...draft, ...changes };
    setDraft(next);
    await update.mutateAsync({ id: next.id, ...changes });
  };

  return (
    <Sheet open={!!link} onOpenChange={(o) => { if (!o) onClose(); }}>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>{draft.display_name || "TradingView link"}</SheetTitle>
        </SheetHeader>

        <SheetBody>
          <div className="space-y-5">
            {/* ── Identity ── */}
            <Input
              label="Name"
              value={draft.display_name}
              onChange={(e) => setDraft({ ...draft, display_name: e.target.value })}
              onBlur={() => patch({ display_name: draft.display_name })}
            />

            <ToggleRow
              label="Active"
              hint="When off, the webhook URL returns 404. Use this to pause without losing the URL."
              checked={draft.is_active}
              onChange={(v) => patch({ is_active: v })}
            />

            {/* ── Auto-fire ── */}
            <div className="space-y-3 rounded-md border border-border bg-surface-2 p-3">
              <ToggleRow
                label="Auto-fire workflow"
                hint="Spawn an AgentRun on every parsed alert. RiskGuard still gates execution."
                checked={draft.autofire_enabled}
                onChange={(v) => patch({ autofire_enabled: v })}
              />
              {draft.autofire_enabled && (
                <>
                  <Input
                    label="Strategy name"
                    hint="One of: directional, short_straddle, pyramid, intraday_screener, ..."
                    value={draft.default_strategy_name}
                    onChange={(e) => setDraft({ ...draft, default_strategy_name: e.target.value })}
                    onBlur={() => patch({ default_strategy_name: draft.default_strategy_name })}
                  />
                  <Input
                    label="Allowed actions (comma-separated)"
                    hint="Leave blank to allow all. Example: BUY,SELL"
                    value={(draft.allowed_actions || []).join(",")}
                    onChange={(e) => setDraft({
                      ...draft,
                      allowed_actions: e.target.value
                        .split(",")
                        .map((s) => s.trim().toUpperCase())
                        .filter(Boolean),
                    })}
                    onBlur={() => patch({ allowed_actions: draft.allowed_actions })}
                  />
                  <WatchlistSelect
                    value={draft.watchlist}
                    onChange={(id) => patch({ watchlist: id })}
                  />
                </>
              )}
            </div>

            {/* ── Recent alerts ── */}
            <div>
              <h3 className="text-body-sm font-semibold text-fg mb-2">Recent alerts</h3>
              {recent.length === 0 ? (
                <p className="text-caption text-fg-subtle">No alerts received yet.</p>
              ) : (
                <ul className="space-y-2" role="list">
                  {recent.map((r) => <RecentAlertRow key={r.id} row={r} />)}
                </ul>
              )}
            </div>
          </div>
        </SheetBody>

        <SheetFooter>
          <Button variant="secondary" onClick={onClose}>Close</Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  );
}

function RecentAlertRow({ row }: { row: TradingViewSignalRow }) {
  const parsed = row.parsed as { symbol?: string; action?: string; price?: number };
  return (
    <li className="rounded-sm border border-border bg-surface px-3 py-2">
      <div className="flex items-center gap-2 text-body-sm">
        <span className="text-fg-subtle font-mono">{fmtRel(row.received_at)} ago</span>
        {row.parse_error ? (
          <Badge tone="danger">parse failed</Badge>
        ) : (
          <>
            <Badge tone={parsed.action === "SELL" ? "warning" : "success"}>
              {parsed.action || "?"}
            </Badge>
            <span className="text-fg">{parsed.symbol || "?"}</span>
            {parsed.price != null && (
              <span className="text-fg-subtle font-mono">@ {parsed.price}</span>
            )}
          </>
        )}
        {row.workflow_run && <Badge tone="brand">Run fired</Badge>}
      </div>
      {row.parse_error && (
        <div className="text-caption text-warn mt-1 break-all">{row.parse_error}</div>
      )}
    </li>
  );
}

function ToggleRow({
  label, hint, checked, onChange,
}: {
  label: string; hint?: string; checked: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-3 cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className={cn(
          "mt-1 h-4 w-4 rounded-xs border-border bg-surface",
          "checked:bg-accent checked:border-accent",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
        )}
      />
      <div className="flex-1">
        <div className="text-body-sm text-fg">{label}</div>
        {hint && <div className="text-caption text-fg-subtle mt-0.5">{hint}</div>}
      </div>
    </label>
  );
}

/** Watchlist binding for autofire. Empty selection = no symbol gate; pick
 *  a watchlist to restrict autofire to symbols in that list (auto-resolved
 *  or operator-typed). Orthogonal to the action allowlist (BUY/SELL).
 *
 *  Lives next to the strategy + action inputs so the operator sees the
 *  full gate picture in one place: "fire `directional`, but only for BUYs
 *  on symbols in my Signal-Rank top-20". */
function WatchlistSelect({
  value, onChange,
}: { value: string | null; onChange: (id: string | null) => void }) {
  const { data: watchlists = [] } = useWatchlists();
  return (
    <div>
      <div className="text-body-sm text-fg mb-1">Restrict to watchlist (optional)</div>
      <select
        value={value || ""}
        onChange={(e) => onChange(e.target.value || null)}
        className="h-9 w-full rounded-xs bg-surface border border-border px-2 text-body-sm"
      >
        <option value="">No symbol gate (autofire any symbol)</option>
        {watchlists.map((w) => (
          <option key={w.id} value={w.id}>
            {w.name} · {w.symbol_count} symbol{w.symbol_count === 1 ? "" : "s"}
            {w.is_auto ? " (auto)" : ""}
          </option>
        ))}
      </select>
      <p className="text-caption text-fg-subtle mt-1">
        Alerts for symbols outside the watchlist are still persisted, but no AgentRun fires.
      </p>
    </div>
  );
}
