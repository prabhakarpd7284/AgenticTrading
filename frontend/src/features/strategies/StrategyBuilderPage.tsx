import * as React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { BookOpen, Plus, Sparkles } from "lucide-react";

import { api } from "@/lib/api";
import type { StrategySchema } from "@/types";
import { fmtRel } from "@/lib/utils";
import { useLegacyStrategies } from "@/lib/legacy";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import {
  Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger,
} from "@/components/ui/Dialog";
import { DataTable, type Column } from "@/components/ui/DataTable";
import { EmptyState } from "@/components/ui/EmptyState";

interface StrategyInstance {
  id: string;
  strategy_name: string;
  name: string;
  is_enabled: boolean;
  params: Record<string, unknown>;
  updated_at: string;
}

export function StrategyBuilderPage() {
  const qc = useQueryClient();
  const { data: catalog = [] } = useQuery({
    queryKey: ["strategy-catalog"],
    queryFn: () => api.get<StrategySchema[]>("/agents/catalog/").then((r) => r.data),
  });
  const { data: instances = [], isLoading } = useQuery({
    queryKey: ["strategy-instances"],
    queryFn: () => api.get<StrategyInstance[]>("/strategies/instances/").then((r) => r.data),
  });
  const { data: legacyStrategies = [] } = useLegacyStrategies();

  const [open, setOpen] = React.useState(false);

  const columns: Column<StrategyInstance>[] = [
    { key: "name", header: "Name", sortable: true,
      render: (s) => <span className="font-medium text-fg">{s.name}</span> },
    { key: "strategy_name", header: "Strategy",
      render: (s) => <Badge tone="brand">{s.strategy_name}</Badge> },
    { key: "is_enabled", header: "Status",
      render: (s) => s.is_enabled
        ? <Badge tone="success" dot>Enabled</Badge>
        : <Badge tone="neutral" dot>Paused</Badge> },
    { key: "updated_at", header: "Updated", align: "right",
      render: (s) => <span className="text-body-sm text-fg-muted">{fmtRel(s.updated_at)} ago</span> },
  ];

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      <header className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <p className="text-caption uppercase tracking-wider text-fg-subtle">Strategies</p>
          <h1 className="text-h1 text-fg">Your playbook</h1>
          <p className="text-body-sm text-fg-muted mt-1">
            Compose strategies from the catalog, tune parameters, and enable them for live desks.
          </p>
        </div>
        <Dialog open={open} onOpenChange={setOpen}>
          <DialogTrigger asChild>
            <Button leading={<Plus className="h-4 w-4" />}>New strategy</Button>
          </DialogTrigger>
          <NewStrategyDialog catalog={catalog} onDone={() => {
            qc.invalidateQueries({ queryKey: ["strategy-instances"] });
            setOpen(false);
          }} />
        </Dialog>
      </header>

      {/* Catalog preview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {catalog.length === 0 ? (
          <EmptyState
            className="lg:col-span-3"
            icon={<BookOpen />}
            title="Catalog unavailable"
            description="Register strategies via the entry-point plugin system. See docs/ARCHITECTURE.md."
          />
        ) : (
          catalog.map((s) => (
            <Card key={s.name} className="h-full">
              <CardHeader>
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <CardTitle>{s.name}</CardTitle>
                    <CardDescription>v{s.version} · {s.asset_class}</CardDescription>
                  </div>
                  <Sparkles className="h-4 w-4 text-accent shrink-0" aria-hidden />
                </div>
              </CardHeader>
              <CardContent>
                <ul className="text-body-sm text-fg-muted space-y-1">
                  {s.required_retrievers?.length ? (
                    <li>Retrievers: <span className="font-mono text-fg">{s.required_retrievers.join(", ")}</span></li>
                  ) : null}
                  <li>Params: <span className="text-fg-subtle">{Object.keys(s.params ?? {}).length} fields</span></li>
                </ul>
              </CardContent>
            </Card>
          ))
        )}
      </div>

      {/* Your strategies */}
      <Card>
        <CardHeader>
          <CardTitle>Your strategies</CardTitle>
          <CardDescription>Instances you've configured for this tenant.</CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <DataTable<StrategyInstance>
            columns={columns}
            rows={instances}
            loading={isLoading}
            rowKey={(s) => s.id}
            emptyTitle="No strategies yet"
            emptyDescription="Create one from a catalog template to start configuring parameters."
          />
        </CardContent>
      </Card>

      {/* Legacy playbook — StrategyDoc rows from the legacy trading app */}
      {legacyStrategies.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Legacy playbook</CardTitle>
            <CardDescription>
              {legacyStrategies.length} strategy note{legacyStrategies.length > 1 ? "s" : ""} imported
              from the pre-v2 `trading.StrategyDoc` table. Migrate into v2 strategies when ready.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <ul className="divide-y divide-border">
              {legacyStrategies.map((s) => (
                <li key={s.id} className="px-5 py-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-body-sm text-fg font-medium">{s.name}</div>
                      <div className="text-caption text-fg-subtle">
                        {s.category}{s.created_at && ` · added ${fmtRel(s.created_at)} ago`}
                      </div>
                      {s.description && (
                        <p className="text-body-sm text-fg-muted mt-1 line-clamp-2">{s.description}</p>
                      )}
                    </div>
                    <Badge tone="neutral" className="shrink-0">{s.category || "general"}</Badge>
                  </div>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* New strategy dialog                                                 */
/* ------------------------------------------------------------------ */
function NewStrategyDialog({
  catalog, onDone,
}: { catalog: StrategySchema[]; onDone: () => void }) {
  const [name, setName] = React.useState("");
  const [strategy, setStrategy] = React.useState(catalog[0]?.name ?? "");
  React.useEffect(() => { if (catalog.length && !strategy) setStrategy(catalog[0].name); }, [catalog, strategy]);

  const mut = useMutation({
    mutationFn: (body: unknown) => api.post("/strategies/instances/", body),
    onSuccess: () => { toast.success("Strategy created"); onDone(); setName(""); },
    onError: () => toast.error("Failed to create strategy"),
  });

  return (
    <DialogContent>
      <DialogTitle>New strategy</DialogTitle>
      <DialogDescription>
        Parameters can be tuned after creation from the instance view.
      </DialogDescription>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          mut.mutate({ strategy_name: strategy, strategy_version: "1.0.0", name, params: {} });
        }}
        className="mt-4 space-y-3"
      >
        <Input
          label="Name"
          required
          placeholder="e.g. NIFTY breakout — conservative"
          value={name}
          onChange={(e) => setName(e.target.value)}
          hint="Give it a name your team will recognise."
        />
        <div>
          <label className="text-body-sm text-fg mb-1.5 inline-block">Template</label>
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
            className="h-9 w-full rounded-sm border border-border bg-surface px-3 text-body-sm text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
          >
            {catalog.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name} · {s.asset_class}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center justify-end gap-2 pt-2">
          <Button type="submit" loading={mut.isPending}>Create</Button>
        </div>
      </form>
    </DialogContent>
  );
}
