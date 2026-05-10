import * as React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { ArrowRight, CheckCircle2, Lock, RotateCw, ShieldAlert, ShieldCheck } from "lucide-react";

import { api } from "@/lib/api";
import { fmtRel } from "@/lib/utils";

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/Tooltip";
import { DataTable, type Column } from "@/components/ui/DataTable";

interface BrokerLink {
  id: string;
  broker_name: string;
  status: "active" | "expired" | "disabled";
  last_refreshed_at: string | null;
  account_id: string | null;
}

const brokers = [
  {
    name: "angel_one",
    label: "Angel One SmartAPI",
    blurb: "Certified integration. NSE cash + NFO options.",
    recommended: true,
  },
  {
    name: "zerodha",
    label: "Zerodha Kite",
    blurb: "Coming soon — join the waitlist.",
    disabled: true,
  },
  {
    name: "paper",
    label: "Paper broker",
    blurb: "Virtual fills against live market prices.",
  },
];

export function BrokerLinkPage() {
  const qc = useQueryClient();
  const { data: links = [], isLoading } = useQuery({
    queryKey: ["broker-links"],
    queryFn: () => api.get<BrokerLink[]>("/brokers/").then((r) => r.data),
  });

  const connectMut = useMutation({
    mutationFn: (name: string) => api.post(`/brokers/${name}/connect/`),
    onSuccess: () => { toast.success("Broker linked"); qc.invalidateQueries({ queryKey: ["broker-links"] }); },
    onError: () => toast.error("Could not link broker"),
  });

  const refreshMut = useMutation({
    mutationFn: (id: string) => api.post(`/brokers/${id}/refresh/`),
    onSuccess: () => { toast.success("Token refreshed"); qc.invalidateQueries({ queryKey: ["broker-links"] }); },
  });

  const columns: Column<BrokerLink>[] = [
    { key: "broker_name", header: "Broker",
      render: (l) => <span className="font-medium text-fg capitalize">{l.broker_name.replace("_", " ")}</span> },
    { key: "account_id", header: "Account",
      render: (l) => <span className="font-mono text-body-sm text-fg-muted">{l.account_id ?? "—"}</span> },
    { key: "status", header: "Status",
      render: (l) => <BrokerStatus status={l.status} /> },
    { key: "last_refreshed_at", header: "Refreshed",
      render: (l) => <span className="text-body-sm text-fg-muted">{l.last_refreshed_at ? `${fmtRel(l.last_refreshed_at)} ago` : "never"}</span> },
    { key: "action", header: "", align: "right",
      render: (l) => (
        <Button
          variant="ghost" size="sm"
          onClick={() => refreshMut.mutate(l.id)}
          loading={refreshMut.isPending && refreshMut.variables === l.id}
          leading={<RotateCw className="h-4 w-4" />}
        >
          Refresh
        </Button>
      ) },
  ];

  return (
    <div className="px-6 py-6 space-y-6 max-w-[1440px] mx-auto">
      <header>
        <p className="text-caption uppercase tracking-wider text-fg-subtle">Broker</p>
        <h1 className="text-h1 text-fg">Route orders to your broker</h1>
        <p className="text-body-sm text-fg-muted mt-1 max-w-2xl">
          Credentials are stored in AWS Secrets Manager, rotated daily, and never logged.
          You can revoke access at any time.
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {brokers.map((b) => (
          <Card key={b.name} className="h-full flex flex-col">
            <CardHeader>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <CardTitle>{b.label}</CardTitle>
                  <CardDescription>{b.blurb}</CardDescription>
                </div>
                {b.recommended && <Badge tone="brand">Recommended</Badge>}
              </div>
            </CardHeader>
            <CardContent className="flex-1">
              <ul className="text-body-sm text-fg-muted space-y-1.5">
                <li className="flex gap-2"><ShieldCheck className="h-4 w-4 text-accent shrink-0 mt-0.5" aria-hidden /> OAuth-based — no password sharing</li>
                <li className="flex gap-2"><Lock       className="h-4 w-4 text-accent shrink-0 mt-0.5" aria-hidden /> Secrets Manager with auto-rotation</li>
                <li className="flex gap-2"><CheckCircle2 className="h-4 w-4 text-accent shrink-0 mt-0.5" aria-hidden /> Revocable from Settings at any time</li>
              </ul>
            </CardContent>
            <CardFooter>
              <Button
                variant={b.recommended ? "primary" : "secondary"}
                size="sm"
                disabled={b.disabled}
                loading={connectMut.isPending && connectMut.variables === b.name}
                onClick={() => connectMut.mutate(b.name)}
                trailing={<ArrowRight className="h-3.5 w-3.5" />}
              >
                {b.disabled ? "Waitlist" : "Connect"}
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Linked accounts</CardTitle>
          <CardDescription>Active broker sessions for this tenant.</CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <DataTable<BrokerLink>
            columns={columns}
            rows={links}
            loading={isLoading}
            rowKey={(l) => l.id}
            emptyTitle="No broker linked yet"
            emptyDescription="Link Angel One to route live orders — or stay in paper mode while you calibrate."
          />
        </CardContent>
      </Card>
    </div>
  );
}

function BrokerStatus({ status }: { status: BrokerLink["status"] }) {
  if (status === "active") return <Badge tone="success" dot>Active</Badge>;
  if (status === "expired") return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex"><Badge tone="warning" dot>Expired</Badge></span>
      </TooltipTrigger>
      <TooltipContent>Session expired — click Refresh to renew the token.</TooltipContent>
    </Tooltip>
  );
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex"><Badge tone="danger" dot><ShieldAlert className="h-3 w-3 mr-0.5" aria-hidden />Disabled</Badge></span>
      </TooltipTrigger>
      <TooltipContent>Disabled by admin. New orders will be rejected.</TooltipContent>
    </Tooltip>
  );
}
