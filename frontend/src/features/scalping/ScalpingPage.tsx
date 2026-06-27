import * as React from "react";
import { CandlestickChart, ChevronLeft, ChevronRight, Pause, Play, SkipForward, Square } from "lucide-react";

import {
  Badge, Button, Card, CardBody, CardHeader, CardTitle, Input, KPI,
  Tabs, TabsContent, TabsList, TabsTrigger,
} from "@/components/ui";
import { LiveCandleChart } from "@/components/charts/LiveCandleChart";
import { INDEX_LIST, getLotSize } from "@/lib/market-config";
import { cn, fmtInr } from "@/lib/utils";
import { useCreateScalpRun, useScalpDefaults } from "./scalping.api";
import { useScalpSession } from "./hooks/useScalpSession";
import type { ScalpConfig } from "./scalping.types";

const SPEEDS = [1, 2, 4, 8, 16, 64];
const RESOLUTIONS = ["5S", "10S", "15S", "30S"] as const;

const seg = (active: boolean) =>
  cn("px-3 py-1.5 text-sm rounded-md border transition-colors",
    active ? "bg-accent/15 text-accent border-accent/40" : "border-border text-fg-muted hover:text-fg");

const stepBtn = "grid h-9 w-7 place-items-center rounded-sm border border-border text-fg-muted hover:text-fg disabled:opacity-40 disabled:cursor-not-allowed";
const fieldCls = "h-9 rounded-sm border border-border bg-surface px-2 text-sm text-fg outline-none focus:border-border-strong";

function StrikeStepper({ value, step, onChange }: { value: number; step: number; onChange: (v: number) => void }) {
  return (
    <div>
      <div className="mb-1 text-xs text-fg-muted">Strike</div>
      <div className="flex items-center gap-1">
        <button type="button" className={stepBtn} onClick={() => onChange(Math.max(0, value - step))}
          aria-label="Lower strike">
          <ChevronLeft className="h-4 w-4" />
        </button>
        <input type="number" step={step} value={value || ""} onChange={(e) => onChange(Number(e.target.value))}
          className={cn(fieldCls, "w-24 text-center")} />
        <button type="button" className={stepBtn} onClick={() => onChange(value + step)} aria-label="Higher strike">
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

function DateStepper({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  // IST calendar date — a plain UTC date blocks selecting the current IST day
  // between 00:00 and 05:30 IST (when UTC is still on the previous date).
  const today = new Date(Date.now() + 5.5 * 3600 * 1000).toISOString().slice(0, 10);
  const step = (dir: number) => {
    if (!value) return;
    const d = new Date(`${value}T00:00:00`);
    do { d.setDate(d.getDate() + dir); } while (d.getDay() === 0 || d.getDay() === 6);  // skip weekends
    const iso = d.toISOString().slice(0, 10);
    onChange(iso > today ? today : iso);
  };
  return (
    <div>
      <div className="mb-1 text-xs text-fg-muted">Date</div>
      <div className="flex items-center gap-1">
        <button type="button" className={stepBtn} onClick={() => step(-1)} aria-label="Previous trading day">
          <ChevronLeft className="h-4 w-4" />
        </button>
        <input type="date" value={value} max={today} onChange={(e) => onChange(e.target.value)}
          className={cn(fieldCls, "w-36 [color-scheme:dark]")} />
        <button type="button" className={stepBtn} onClick={() => step(1)} disabled={!value || value >= today}
          aria-label="Next trading day">
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

function ExpiryStepper({ value, options, onChange }: { value: string; options: string[]; onChange: (v: string) => void }) {
  const opts = value && !options.includes(value) ? [value, ...options] : options;
  const i = opts.indexOf(value);
  const step = (dir: number) => {
    const ni = Math.min(Math.max(i + dir, 0), opts.length - 1);
    if (opts[ni]) onChange(opts[ni]);
  };
  return (
    <div>
      <div className="mb-1 text-xs text-fg-muted">Expiry</div>
      <div className="flex items-center gap-1">
        <button type="button" className={stepBtn} onClick={() => step(-1)} disabled={i <= 0}
          aria-label="Earlier expiry">
          <ChevronLeft className="h-4 w-4" />
        </button>
        <select value={value} onChange={(e) => onChange(e.target.value)} className={cn(fieldCls, "w-32")}>
          {!opts.length && <option value="">—</option>}
          {opts.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
        <button type="button" className={stepBtn} onClick={() => step(1)} disabled={i < 0 || i >= opts.length - 1}
          aria-label="Later expiry">
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

export default function ScalpingPage() {
  const [cfg, setCfg] = React.useState<Partial<ScalpConfig>>({
    underlying: "NIFTY", strike: 23800, type: "CE", expiry: "", date: "",
    resolution: "5S", bin_width: 20, capital: 100_000, risk_pct: 1, max_pyramids: 4,
    lot_size: getLotSize("NIFTY"), require_bias_alignment: true, allow_reverse: true,
    speed: 4, dry_run: true, mode: "sim", place_orders: false,
    pullback_min_bins: 0.6, reversal_bins: 0.35,
  });
  const live = cfg.mode === "live";
  const set = <K extends keyof ScalpConfig>(k: K, v: ScalpConfig[K]) => setCfg((c) => ({ ...c, [k]: v }));

  const [runId, setRunId] = React.useState<string | null>(null);
  const createRun = useCreateScalpRun();
  const session = useScalpSession();

  // Auto-fill the resolved default date + expiry. Expiry follows the underlying;
  // date fills only when blank so a manual date survives an underlying switch.
  const defaults = useScalpDefaults(cfg.underlying ?? "NIFTY");
  React.useEffect(() => {
    if (!defaults.data) return;
    setCfg((c) => ({
      ...c,
      date: c.date || defaults.data!.date,
      expiry: defaults.data!.expiry,
      strike: defaults.data!.strike ?? c.strike,   // ATM for the underlying (keep current if unavailable)
    }));
  }, [defaults.data]);

  const simulate = () => {
    createRun.mutate(cfg, {
      onSuccess: ({ run_id }) => { setRunId(run_id); session.controls.start(run_id); },
    });
  };

  const running = session.status === "running";
  const paused = session.status === "paused";

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-lg bg-accent/10 text-accent">
            <CandlestickChart className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-lg font-semibold">Scalping Simulator</h1>
            <p className="text-sm text-fg-muted">
              Bin-pressure scalping on seconds candles · both directions · live tick replay
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {session.source && (
            <Badge tone="neutral">
              {session.source}
              {session.info.date ? ` · ${session.info.date}` : ""}
              {session.info.expiry ? ` · exp ${session.info.expiry}` : ""}
            </Badge>
          )}
          <ConnPill state={session.connState} status={session.status} />
        </div>
      </div>

      {/* Config */}
      <Card>
        <CardHeader><CardTitle>Setup</CardTitle></CardHeader>
        <CardBody className="space-y-3">
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex gap-1">
              {INDEX_LIST.map((u) => (
                <button key={u} className={seg(cfg.underlying === u)}
                  onClick={() => { set("underlying", u); set("lot_size", getLotSize(u)); }}>{u}</button>
              ))}
            </div>
            <div className="flex gap-1">
              {(["CE", "PE"] as const).map((t) => (
                <button key={t} className={seg(cfg.type === t)} onClick={() => set("type", t)}>{t}</button>
              ))}
            </div>
            <StrikeStepper value={cfg.strike ?? 0} step={defaults.data?.strike_step ?? 50}
              onChange={(v) => set("strike", v)} />
            <ExpiryStepper value={cfg.expiry ?? ""} options={defaults.data?.expiries ?? []}
              onChange={(v) => set("expiry", v)} />
            <DateStepper value={cfg.date ?? ""} onChange={(v) => set("date", v)} />
            <div>
              <div className="mb-1 text-xs text-fg-muted">Resolution</div>
              <div className="flex gap-1">
                {RESOLUTIONS.map((r) => (
                  <button key={r} className={seg(cfg.resolution === r)} onClick={() => set("resolution", r)}>{r}</button>
                ))}
              </div>
            </div>
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <Input label="Bin width" type="number" value={cfg.bin_width ?? ""} containerClassName="w-28"
              onChange={(e) => set("bin_width", Number(e.target.value))} />
            <Input label="Capital" type="number" value={cfg.capital ?? ""} containerClassName="w-32"
              onChange={(e) => set("capital", Number(e.target.value))} />
            <Input label="Risk %" type="number" value={cfg.risk_pct ?? ""} containerClassName="w-24"
              onChange={(e) => set("risk_pct", Number(e.target.value))} />
            <Input label="Pullback (bins)" type="number" step="0.1" value={cfg.pullback_min_bins ?? ""}
              hint="dip before reversal" containerClassName="w-32"
              onChange={(e) => set("pullback_min_bins", Number(e.target.value))} />
            <Input label="Reversal (bins)" type="number" step="0.1" value={cfg.reversal_bins ?? ""}
              hint="bounce to confirm" containerClassName="w-32"
              onChange={(e) => set("reversal_bins", Number(e.target.value))} />
            <div>
              <div className="mb-1 text-xs text-fg-muted">Mode</div>
              <div className="flex gap-1">
                <button className={seg(!live)} onClick={() => set("mode", "sim")}>Sim</button>
                <button className={seg(live)} onClick={() => set("mode", "live")}>Live</button>
              </div>
            </div>
            {!live && (
              <label className="flex items-center gap-2 text-sm text-fg-muted">
                <input type="checkbox" className="accent-accent" checked={cfg.dry_run ?? true}
                  onChange={(e) => set("dry_run", e.target.checked)} />
                Sample data
              </label>
            )}
            {live && (
              <label className="flex items-center gap-2 text-sm text-fg-muted" title="Paper orders only">
                <input type="checkbox" className="accent-accent" checked={cfg.place_orders ?? false}
                  onChange={(e) => set("place_orders", e.target.checked)} />
                Place paper orders
              </label>
            )}
            <label className="flex items-center gap-2 text-sm text-fg-muted">
              <input type="checkbox" className="accent-accent" checked={cfg.require_bias_alignment ?? true}
                onChange={(e) => set("require_bias_alignment", e.target.checked)} />
              Bias gate
            </label>
            <Button leading={<Play className="h-4 w-4" />} loading={createRun.isPending} onClick={simulate}>
              {live ? "Go Live (Fyers)" : "Simulate"}
            </Button>
          </div>
          {createRun.isError && (
            <p className="text-sm text-pnl-down">Could not start: {String((createRun.error as Error)?.message)}</p>
          )}
        </CardBody>
      </Card>

      {runId && (
        <>
          {/* Control bar */}
          <Card>
            <CardBody className="flex flex-wrap items-center gap-2">
              {paused
                ? <Button size="sm" variant="secondary" leading={<Play className="h-4 w-4" />} onClick={session.controls.resume}>Resume</Button>
                : <Button size="sm" variant="secondary" leading={<Pause className="h-4 w-4" />} onClick={session.controls.pause} disabled={!running}>Pause</Button>}
              <Button size="sm" variant="ghost" leading={<SkipForward className="h-4 w-4" />} onClick={session.controls.step} disabled={!paused}>Step</Button>
              <div className="flex items-center gap-1 pl-2">
                <span className="text-xs text-fg-muted">Speed</span>
                {SPEEDS.map((s) => (
                  <button key={s} className={seg(session.speed === s)} onClick={() => session.controls.setSpeed(s)}>{s}×</button>
                ))}
              </div>
              <div className="ml-auto flex items-center gap-1">
                <button className={seg(false) + " !text-pnl-up"} onClick={() => session.controls.manualOrder("buy")}>Buy</button>
                <button className={seg(false) + " !text-pnl-down"} onClick={() => session.controls.manualOrder("short")}>Short</button>
                <button className={seg(false)} onClick={() => session.controls.manualOrder("add")}>Add</button>
                <button className={seg(false) + " !text-warn"} onClick={() => session.controls.manualOrder("exit")}>Exit</button>
                <Button size="sm" variant="destructive" leading={<Square className="h-3.5 w-3.5" />} onClick={session.controls.stop}>Stop</Button>
              </div>
            </CardBody>
          </Card>

          {/* KPIs */}
          <KpiStrip session={session} />

          {/* Chart + pressure */}
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-4">
            <Card>
              <CardBody>
                <LiveCandleChart ref={session.chartRef} className="h-[420px] w-full" />
              </CardBody>
            </Card>
            <PressureHistogram session={session} />
          </div>

          {/* Stats + decision log */}
          <Card>
            <CardBody>
              <Tabs defaultValue="stats">
                <TabsList>
                  <TabsTrigger value="stats">Trade Stats</TabsTrigger>
                  <TabsTrigger value="log">Decision Log</TabsTrigger>
                </TabsList>
                <TabsContent value="stats" className="pt-3">
                  <TradeStats session={session} lotSize={cfg.lot_size ?? 65} />
                </TabsContent>
                <TabsContent value="log" className="pt-3">
                  <LogView session={session} />
                </TabsContent>
              </Tabs>
            </CardBody>
          </Card>
        </>
      )}
    </div>
  );
}

function ConnPill({ state, status }: { state: string; status: string }) {
  const tone = status === "error" ? "danger" : state === "live" ? "success" : state === "closed_auth" ? "danger" : "warning";
  const label = status === "done" ? "done" : state === "live" ? status : state;
  return <Badge tone={tone as never} dot>{label}</Badge>;
}

type S = ReturnType<typeof useScalpSession>;

function KpiStrip({ session }: { session: S }) {
  const pos = session.position;
  const k = session.kpis;
  const side = pos?.side ?? "FLAT";
  const sideTone = side === "LONG" ? "success" : side === "SHORT" ? "danger" : "neutral";
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
      <Card><CardBody className="space-y-1">
        <div className="text-xs text-fg-muted">Side</div>
        <Badge tone={sideTone as never} dot>{side}{pos && side !== "FLAT" ? ` ${pos.lots}` : ""}</Badge>
      </CardBody></Card>
      <KPI label="Unrealized" value={pos?.unrealized_inr ?? 0} valueFormat="inr" live />
      <KPI label="Avg entry" value={pos?.avg ?? 0} valueFormat="num" />
      <KPI label="SL" value={pos?.sl ?? 0} valueFormat="num" />
      <Card><CardBody className="space-y-1">
        <div className="text-xs text-fg-muted">Pressure</div>
        <div className={cn("font-mono text-lg font-semibold", (session.pressure?.pressure ?? 0) >= 0 ? "text-pnl-up" : "text-pnl-down")}>
          {session.pressure ? session.pressure.pressure.toFixed(2) : "—"}
        </div>
        <div className="text-[10px] text-fg-muted">{session.pressure?.bias ?? ""}</div>
      </CardBody></Card>
      <KPI label="Realized" value={k?.realized_pnl_inr ?? 0} valueFormat="inr" />
      <KPI label="Peak lots" value={k?.peak_lots ?? pos?.lots ?? 0} valueFormat="num" />
    </div>
  );
}

function PressureHistogram({ session }: { session: S }) {
  const p = session.pressure;
  const prof = session.profile;
  const bins = p?.bins ?? [];
  const max = Math.max(1, ...bins.map((b) => b.weight));
  const smax = Math.max(1, ...(prof?.bins ?? []).map((b) => b.weight));
  const inVA = (low: number) => p && low >= p.val && low < p.vah;
  return (
    <Card>
      <CardHeader><CardTitle>Pressure profile</CardTitle></CardHeader>
      <CardBody className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-fg-muted">Pressure</span>
          <span className={cn("font-mono font-semibold", p && p.pressure >= 0 ? "text-pnl-up" : "text-pnl-down")}>
            {p ? p.pressure.toFixed(3) : "—"}
          </span>
        </div>
        <div className="h-2 rounded bg-border overflow-hidden">
          <div className={cn("h-full", p && p.pressure >= 0 ? "bg-pnl-up" : "bg-pnl-down")}
            style={{ width: `${Math.abs(p?.pressure ?? 0) * 100}%`, marginLeft: p && p.pressure < 0 ? "auto" : 0 }} />
        </div>
        <div className="space-y-0.5 max-h-[150px] overflow-y-auto pt-1">
          {[...bins].reverse().map((b) => (
            <div key={b.low} className="flex items-center gap-2 text-[11px] font-mono">
              <span className="w-12 text-right text-fg-muted">{b.low}</span>
              <div className="flex-1 h-3 rounded-sm bg-border/40">
                <div className={cn("h-full rounded-sm", inVA(b.low) ? "bg-accent" : "bg-accent/40")}
                  style={{ width: `${(b.weight / max) * 100}%` }} />
              </div>
            </div>
          ))}
          {!bins.length && <p className="text-xs text-fg-muted">waiting for ticks…</p>}
        </div>
        {p && <p className="text-[11px] text-fg-muted">window VA {p.val}–{p.vah} · POC {p.poc}</p>}

        {prof && prof.bins.length > 0 && (
          <div className="pt-2 mt-1 border-t border-border/60">
            <div className="flex items-center justify-between text-[11px] mb-1">
              <span className="text-fg-muted">Visited bins (session)</span>
              <span className="text-fg-muted">POC {prof.poc}</span>
            </div>
            <div className="grid grid-cols-[2.7rem_1fr_2rem_2.2rem_1.8rem] gap-1 text-[10px] text-fg-muted px-0.5 pb-0.5">
              <span>Price</span><span>Vol</span><span className="text-right">%</span>
              <span className="text-right">Ticks</span><span className="text-right">Rev</span>
            </div>
            <div className="space-y-0.5 max-h-[220px] overflow-y-auto">
              {[...prof.bins].reverse().map((b) => (
                <div key={b.low}
                  className={cn("grid grid-cols-[2.7rem_1fr_2rem_2.2rem_1.8rem] gap-1 items-center text-[10px] font-mono",
                    b.poc ? "text-accent" : "text-fg")}>
                  <span className="text-right">{b.low}{b.poc ? " ◀" : ""}</span>
                  <div className="h-2.5 rounded-sm bg-border/40">
                    <div className={cn("h-full rounded-sm", b.poc ? "bg-accent" : "bg-accent/50")}
                      style={{ width: `${(b.weight / smax) * 100}%` }} />
                  </div>
                  <span className="text-right text-fg-muted">{b.pct}</span>
                  <span className="text-right text-fg-muted">{b.ticks}</span>
                  <span className="text-right text-fg-muted">{b.visits}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardBody>
    </Card>
  );
}

function LogView({ session }: { session: S }) {
  const ref = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => { ref.current?.scrollTo({ top: ref.current.scrollHeight }); }, [session.log.length]);
  const tone = (l: string) =>
    l.includes("ENTER") ? "text-pnl-up" : l.includes("EXIT") ? "text-warn"
      : l.includes("ADD") ? "text-accent" : l.startsWith("⚠") || l.startsWith("✗") ? "text-pnl-down" : "text-fg-muted";
  return (
    <div ref={ref} className="max-h-[260px] overflow-y-auto space-y-0.5 font-mono text-xs">
      {session.log.map((l, i) => <div key={i} className={tone(l)}>{l}</div>)}
      {!session.log.length && <p className="text-fg-muted">No decisions yet.</p>}
    </div>
  );
}

function TradeStats({ session, lotSize }: { session: S; lotSize: number }) {
  const s = React.useMemo(() => {
    const exits = session.decisions.filter((d) => d.action === "exit");
    const pnl = (e: (typeof exits)[number]) => e.pnl_pts ?? 0;
    const wins = exits.filter((e) => pnl(e) > 0);
    const losses = exits.filter((e) => pnl(e) < 0);
    const grossWin = wins.reduce((a, e) => a + pnl(e), 0);
    const grossLoss = losses.reduce((a, e) => a + pnl(e), 0);
    return {
      trades: exits.length, wins: wins.length, losses: losses.length,
      winRate: exits.length ? (wins.length / exits.length) * 100 : 0,
      net: grossWin + grossLoss, grossWin, grossLoss,
      avgWin: wins.length ? grossWin / wins.length : 0,
      avgLoss: losses.length ? grossLoss / losses.length : 0,
      pf: grossLoss ? grossWin / Math.abs(grossLoss) : (grossWin > 0 ? Infinity : 0),
      maxWin: wins.length ? Math.max(...wins.map(pnl)) : 0,
      maxLoss: losses.length ? Math.min(...losses.map(pnl)) : 0,
      longs: exits.filter((e) => e.side === "LONG").length,
      shorts: exits.filter((e) => e.side === "SHORT").length,
    };
  }, [session.decisions]);
  const inr = (pts: number) => fmtInr(pts * lotSize);
  const cell = (label: string, value: React.ReactNode, cls = "") => (
    <div className="rounded-md border border-border/60 px-3 py-2">
      <div className="text-[11px] text-fg-muted">{label}</div>
      <div className={cn("font-mono text-sm font-semibold", cls)}>{value}</div>
    </div>
  );
  const up = "text-pnl-up", down = "text-pnl-down";
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
      {cell("Trades", s.trades)}
      {cell("Win rate", `${s.winRate.toFixed(0)}%`)}
      {cell("Wins / Losses", `${s.wins} / ${s.losses}`)}
      {cell("Net", inr(s.net), s.net >= 0 ? up : down)}
      {cell("Profit factor", s.pf === Infinity ? "∞" : s.pf.toFixed(2))}
      {cell("Long / Short", `${s.longs} / ${s.shorts}`)}
      {cell("Gross win", inr(s.grossWin), up)}
      {cell("Gross loss", inr(s.grossLoss), down)}
      {cell("Avg win", inr(s.avgWin), up)}
      {cell("Avg loss", inr(s.avgLoss), down)}
      {cell("Best", inr(s.maxWin), up)}
      {cell("Worst", inr(s.maxLoss), down)}
    </div>
  );
}
