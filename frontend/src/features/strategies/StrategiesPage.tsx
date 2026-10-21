/**
 * Strategies (UX redesign §6 / Strategy Studio R4): every strategy with its
 * stage, version, last backtest result + provenance and the next step.
 * New = pick a template (rule spec or built-in plugin, with an honest
 * "backtestable?" label) → name + real parameter form → the Studio page.
 */
import * as React from "react";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { AlertTriangle, Archive, ArchiveRestore, BookOpen, Copy, FlaskConical, Plus, Trash2 } from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { DataTable, type Column } from "@/components/ui/DataTable";
import { Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger } from "@/components/ui/Dialog";
import { Input } from "@/components/ui/Input";
import { fmtRel } from "@/lib/utils";
import { useKnowledgeDocs } from "@/lib/v2";

import { BacktestLine } from "./BacktestReport";
import { SchemaForm, schemaErrors } from "./SchemaForm";
import {
  BACKTEST_LABEL, STAGE_TONE, apiErrors, useCreateStrategy, useStrategies, useStrategyAction, useTemplates,
  type StrategyRow, type Template,
} from "./useStrategies";

const BT_TONE = { backtest: "success", forward_paper_only: "warning", unavailable: "neutral" } as const;

export function StrategiesPage() {
  const [showArchived, setShowArchived] = React.useState(false);
  const { data: rows = [], isLoading } = useStrategies(showArchived);
  const { data: templates = [] } = useTemplates();
  const act = useStrategyAction();
  const nav = useNavigate();
  const [open, setOpen] = React.useState(false);
  const [confirmDelete, setConfirmDelete] = React.useState<StrategyRow | null>(null);

  const tplTitle = (r: StrategyRow) =>
    templates.find((t) => t.key === r.template_key)?.title ?? (r.kind === "spec" ? "Rule spec" : r.strategy_name);

  const doAction = (r: StrategyRow, action: "duplicate" | "archive" | "unarchive" | "delete") =>
    act.mutate({ id: r.id, action }, {
      onSuccess: (d) => {
        toast.success(action === "duplicate" ? "Duplicated" : action === "delete" ? "Deleted" : `${action}d`);
        if (action === "duplicate" && d) nav(`/strategies/${d.id}`);
        setConfirmDelete(null);
      },
      onError: (e) => toast.error(apiErrors(e).detail),
    });

  const columns: Column<StrategyRow>[] = [
    { key: "name", header: "Strategy", render: (r) => (
      <div className="min-w-0 py-1">
        <Link to={`/strategies/${r.id}`} className="font-medium text-fg hover:underline">{r.name}</Link>
        <div className="text-caption text-fg-subtle">{tplTitle(r)} · v{r.current_version} · updated {fmtRel(r.updated_at)} ago</div>
      </div>
    ) },
    { key: "stage", header: "Stage", render: (r) => (
      <div className="flex flex-col gap-1 items-start">
        <Badge tone={STAGE_TONE[r.stage]} dot>{r.stage}</Badge>
        {r.backtest !== "backtest" && <Badge tone={BT_TONE[r.backtest]}>{BACKTEST_LABEL[r.backtest]}</Badge>}
        {r.archived && <Badge tone="neutral">archived</Badge>}
      </div>
    ) },
    { key: "last_result", header: "Last backtest", render: (r) =>
      r.last_result ? <BacktestLine b={r.last_result} />
        : r.last_backtest ? <BacktestLine b={r.last_backtest} />
        : <span className="text-caption text-fg-subtle">none yet</span> },
    { key: "next_step", header: "What's next", render: (r) => (
      <div className="max-w-xs">
        <div className="text-body-sm text-fg">{r.next_step.title}</div>
        <div className="text-caption text-fg-subtle line-clamp-2" title={r.next_step.detail}>{r.next_step.detail}</div>
      </div>
    ) },
    { key: "actions", header: "", align: "right", render: (r) => (
      <div className="flex justify-end gap-1">
        {r.backtest === "backtest" && !r.archived && (
          <Button size="sm" variant="secondary" leading={<FlaskConical className="h-3.5 w-3.5" />}
                  onClick={() => nav(`/strategies/${r.id}?tab=test`)}>Backtest</Button>
        )}
        <Button size="icon" variant="ghost" aria-label={`Duplicate ${r.name}`} onClick={() => doAction(r, "duplicate")}>
          <Copy className="h-3.5 w-3.5" />
        </Button>
        <Button size="icon" variant="ghost" aria-label={`${r.archived ? "Unarchive" : "Archive"} ${r.name}`}
                onClick={() => doAction(r, r.archived ? "unarchive" : "archive")}>
          {r.archived ? <ArchiveRestore className="h-3.5 w-3.5" /> : <Archive className="h-3.5 w-3.5" />}
        </Button>
        <Button size="icon" variant="ghost" aria-label={`Delete ${r.name}`} onClick={() => setConfirmDelete(r)}>
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
      </div>
    ) },
  ];

  return (
    <div className="p-6 pt-2 space-y-6 max-w-[1400px]">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <p className="text-caption uppercase tracking-wider text-fg-subtle">Strategies</p>
          <h1 className="text-h1 text-fg">Your strategies</h1>
          <p className="text-body-sm text-fg-muted mt-1 max-w-2xl">
            Define a strategy from a template, backtest it on real candles (with costs and an out-of-sample hold-out),
            then paper-trade it. Edits create a new version; results always say which version and data they came from.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" onClick={() => setShowArchived((v) => !v)}>
            {showArchived ? "Show active" : "Show archived"}
          </Button>
          <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
              <Button leading={<Plus className="h-4 w-4" />}>New strategy</Button>
            </DialogTrigger>
            <NewStrategyDialog templates={templates} onCreated={(id) => { setOpen(false); nav(`/strategies/${id}`); }} />
          </Dialog>
        </div>
      </header>

      <Card>
        <CardContent className="p-0">
          <DataTable<StrategyRow>
            columns={columns} rows={rows} loading={isLoading} rowKey={(r) => r.id}
            emptyTitle={showArchived ? "No archived strategies" : "No strategies yet"}
            emptyDescription="Create one from a template — the parameters and rules are editable."
          />
        </CardContent>
      </Card>

      <Dialog open={confirmDelete !== null} onOpenChange={(o) => !o && setConfirmDelete(null)}>
        {confirmDelete && (
          <DialogContent>
            <DialogTitle>Delete {confirmDelete.name}?</DialogTitle>
            {confirmDelete.can_delete ? (
              <DialogDescription>
                This removes the strategy, all {confirmDelete.current_version} version(s) and its backtests. Archive it instead
                to keep the record.
              </DialogDescription>
            ) : (
              <DialogDescription>
                <span className="inline-flex gap-1.5 text-warn"><AlertTriangle className="h-4 w-4 shrink-0" />
                  Can't delete: {confirmDelete.delete_blocker}</span>
              </DialogDescription>
            )}
            <div className="flex justify-end gap-2 mt-4">
              <Button variant="ghost" onClick={() => setConfirmDelete(null)}>Keep</Button>
              {confirmDelete.can_delete ? (
                <Button variant="destructive" loading={act.isPending} onClick={() => doAction(confirmDelete, "delete")}>Delete</Button>
              ) : (
                <Button variant="secondary" onClick={() => doAction(confirmDelete, "archive")}>Archive instead</Button>
              )}
            </div>
          </DialogContent>
        )}
      </Dialog>

      <LegacyPlaybook />
    </div>
  );
}

/* ------------------------------------------------------------------ */

function NewStrategyDialog({ templates, onCreated }: { templates: Template[]; onCreated: (id: string) => void }) {
  const [tpl, setTpl] = React.useState<Template | null>(null);
  const [name, setName] = React.useState("");
  const [params, setParams] = React.useState<Record<string, unknown>>({});
  const create = useCreateStrategy();
  const clientErrs = tpl?.kind === "plugin" ? schemaErrors(tpl.schema, params) : [];
  const server = create.isError ? apiErrors(create.error) : null;

  const pick = (t: Template) => { setTpl(t); setName(t.title); setParams({ ...(t.defaults ?? {}) }); create.reset(); };
  const groups: [string, Template[]][] = [
    ["Rule templates (editable rules)", templates.filter((t) => t.kind === "spec")],
    ["Built-in strategies", templates.filter((t) => t.kind === "plugin")],
  ];

  return (
    <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
      <DialogTitle>New strategy</DialogTitle>
      {!tpl ? (
        <>
          <DialogDescription>Start from a template. Rule templates open in the rule builder; built-in strategies get their real parameter form.</DialogDescription>
          <div className="mt-4 space-y-4">
            {groups.map(([title, list]) => (
              <section key={title}>
                <h3 className="text-caption uppercase tracking-wider text-fg-subtle mb-2">{title}</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {list.map((t) => (
                    <button key={t.key} type="button" onClick={() => pick(t)}
                            className="text-left rounded-md border border-border bg-surface p-3 hover:border-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60">
                      <div className="flex items-start justify-between gap-2">
                        <span className="text-body-sm font-medium text-fg">{t.title}</span>
                        <Badge tone={BT_TONE[t.backtest]}>{BACKTEST_LABEL[t.backtest]}</Badge>
                      </div>
                      {t.summary && <p className="text-caption text-fg-muted mt-1">{t.summary}</p>}
                      {t.reason && <p className="text-caption text-fg-subtle mt-1">{t.reason}</p>}
                      {Object.keys(t.broken ?? {}).length > 0 && (
                        <p className="text-caption text-warn mt-1">Broken: {Object.keys(t.broken).join(", ")} never fire.</p>
                      )}
                      {t.live_gate && t.live_gate.passes === false && (
                        <p className="text-caption text-warn mt-1">{t.live_gate.message}</p>
                      )}
                    </button>
                  ))}
                </div>
              </section>
            ))}
          </div>
        </>
      ) : (
        <form className="mt-3 space-y-4" onSubmit={(e) => {
          e.preventDefault();
          if (clientErrs.length) return;
          create.mutate(
            tpl.kind === "spec" ? { name, template_key: tpl.key } : { name, template_key: tpl.key, params },
            { onSuccess: (d) => { toast.success("Strategy created"); onCreated(d.id); } },
          );
        }}>
          <DialogDescription>
            {tpl.title} — <Badge tone={BT_TONE[tpl.backtest]}>{BACKTEST_LABEL[tpl.backtest]}</Badge>
          </DialogDescription>
          <Input label="Name" required value={name} onChange={(e) => setName(e.target.value)} />
          {tpl.kind === "plugin" ? (
            <div>
              <h3 className="text-caption uppercase tracking-wider text-fg-subtle mb-2">Parameters</h3>
              <SchemaForm schema={tpl.schema} value={params} onChange={setParams}
                          errors={[...clientErrs, ...(server?.errors ?? [])]} broken={tpl.broken} />
              {tpl.notes.length > 0 && (
                <ul className="mt-3 space-y-1 text-caption text-fg-muted list-disc pl-4">
                  {tpl.notes.map((n) => <li key={n}>{n}</li>)}
                </ul>
              )}
            </div>
          ) : (
            <p className="text-body-sm text-fg-muted">
              The rules ({tpl.timeframe}) open in the builder after you create it — change anything, every save is a new version.
            </p>
          )}
          {server && !server.errors.length && <p role="alert" className="text-body-sm text-pnl-down">{server.detail}</p>}
          <div className="flex justify-between gap-2">
            <Button type="button" variant="ghost" onClick={() => setTpl(null)}>Back</Button>
            <Button type="submit" loading={create.isPending} disabled={!name.trim() || clientErrs.length > 0}>Create</Button>
          </div>
        </form>
      )}
    </DialogContent>
  );
}

function LegacyPlaybook() {
  const { data: docs = [] } = useKnowledgeDocs();
  const [open, setOpen] = React.useState(false);
  if (!docs.length) return null;
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2"><BookOpen className="h-4 w-4" />Legacy playbook notes</CardTitle>
            <CardDescription>{docs.length} note(s) from the pre-v2 StrategyDoc table — text only, not runnable.</CardDescription>
          </div>
          <Button size="sm" variant="ghost" onClick={() => setOpen((o) => !o)}>{open ? "Hide" : "Show"}</Button>
        </div>
      </CardHeader>
      {open && (
        <CardContent className="p-0">
          <ul className="divide-y divide-border">
            {docs.map((s) => (
              <li key={s.id} className="px-5 py-3">
                <div className="text-body-sm text-fg font-medium">{s.title}</div>
                {s.content && <p className="text-body-sm text-fg-muted mt-1 line-clamp-2">{s.content}</p>}
              </li>
            ))}
          </ul>
        </CardContent>
      )}
    </Card>
  );
}
