/**
 * SpecBuilder — the no-code rule builder for a StrategySpec.
 *
 * Entry rules are condition rows (left operand · operator · right operand)
 * joined by ALL / ANY, per side. Deeper trees (NOT, nested groups) stay
 * editable as JSON. Validation and the plain-English summary come from the
 * server (`/strategies/spec/validate/`), so the builder never disagrees with
 * the engine.
 */
import * as React from "react";
import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";
import type { Condition, FieldError, Operand, PriceField, Spec } from "./useStrategies";

const FIELD =
  "h-8 rounded-sm border border-border bg-surface px-2 text-body-sm text-fg " +
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60";

const PRICES: PriceField[] = ["open", "high", "low", "close", "volume"];
const OPS: [string, string][] = [
  [">", ">"], [">=", "≥"], ["<", "<"], ["<=", "≤"],
  ["crosses_above", "crosses above"], ["crosses_below", "crosses below"],
  ["rising", "is rising over"], ["falling", "is falling over"],
];

/** kind → params it takes (mirror of spec.py INDICATORS) and whether intraday-only. */
export const INDICATOR_KINDS: Record<string, { label: string; params: Partial<Operand>; intraday?: boolean }> = {
  ema: { label: "EMA", params: { source: "close", period: 20 } },
  sma: { label: "SMA", params: { source: "close", period: 20 } },
  rsi: { label: "RSI", params: { source: "close", period: 14 } },
  atr: { label: "ATR", params: { period: 14 } },
  bb_upper: { label: "Bollinger upper", params: { period: 20, std: 2 } },
  bb_middle: { label: "Bollinger middle", params: { period: 20, std: 2 } },
  bb_lower: { label: "Bollinger lower", params: { period: 20, std: 2 } },
  highest: { label: "Highest of prev N", params: { source: "high", period: 20 } },
  lowest: { label: "Lowest of prev N", params: { source: "low", period: 20 } },
  avg: { label: "Average of prev N", params: { source: "volume", period: 20 } },
  prev_day_high: { label: "Prev day high", params: {} },
  prev_day_low: { label: "Prev day low", params: {} },
  prev_day_close: { label: "Prev day close", params: {} },
  vwap: { label: "VWAP", params: {}, intraday: true },
  day_open: { label: "Today's open", params: {}, intraday: true },
  gap_pct: { label: "Gap % vs prev close", params: {}, intraday: true },
  orb_high: { label: "Opening-range high", params: { minutes: 15 }, intraday: true },
  orb_low: { label: "Opening-range low", params: { minutes: 15 }, intraday: true },
};

type Join = "all" | "any";
interface Flat { join: Join; rows: Condition[] }

export function flatten(c: Condition | null | undefined): Flat | null {
  if (!c) return { join: "all", rows: [] };
  if (c.op) return { join: "all", rows: [c] };
  const kids = c.all ?? c.any;
  if (kids && kids.every((k) => !!k?.op)) return { join: c.all ? "all" : "any", rows: kids };
  return null;
}

export function unflatten(f: Flat): Condition | null {
  if (f.rows.length === 0) return null;
  if (f.rows.length === 1) return f.rows[0];
  return { [f.join]: f.rows } as Condition;
}

const emptyRow = (): Condition => ({ left: { price: "close" }, op: ">", right: { ind: "ema", source: "close", period: 20 } });

function errsFor(errors: FieldError[], prefix: string) {
  return errors.filter((e) => e.path === prefix || e.path.startsWith(`${prefix}.`) || e.path.startsWith(`${prefix}[`));
}

function ErrList({ errors }: { errors: FieldError[] }) {
  if (!errors.length) return null;
  return (
    <ul className="mt-1 space-y-0.5">
      {errors.map((e) => <li key={e.path + e.message} role="alert" className="text-caption text-pnl-down">{e.path}: {e.message}</li>)}
    </ul>
  );
}

export function SpecBuilder({
  spec, onChange, errors = [], disabled,
}: { spec: Spec; onChange: (s: Spec) => void; errors?: FieldError[]; disabled?: boolean }) {
  const daily = spec.timeframe === "1d";
  const set = (patch: Partial<Spec>) => onChange({ ...spec, ...patch });
  const setIn = <K extends keyof Spec>(k: K, patch: Partial<Spec[K]>) =>
    onChange({ ...spec, [k]: { ...(spec[k] as object), ...patch } });
  const [symText, setSymText] = React.useState(spec.universe.symbols.join(", "));
  React.useEffect(() => setSymText(spec.universe.symbols.join(", ")), [spec.universe.symbols]);
  const [json, setJson] = React.useState<string | null>(null);

  return (
    <fieldset disabled={disabled} className="space-y-5" data-testid="spec-builder">
      <Section title="Market">
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
          <Labeled label="Timeframe">
            <select aria-label="Timeframe" className={cn(FIELD, "w-full")} value={spec.timeframe}
                    onChange={(e) => set({ timeframe: e.target.value as Spec["timeframe"] })}>
              {["1m", "5m", "15m", "60m", "1d"].map((t) => <option key={t} value={t}>{t === "1d" ? "1d (daily, delivery)" : `${t} (intraday)`}</option>)}
            </select>
          </Labeled>
          <div className="sm:col-span-3">
            <Labeled label="Stocks (NSE symbols)">
              <input aria-label="Stocks" className={cn(FIELD, "w-full font-mono")} value={symText}
                     onChange={(e) => setSymText(e.target.value)}
                     onBlur={() => setIn("universe", { symbols: symText.split(/[\s,]+/).map((s) => s.trim().toUpperCase()).filter(Boolean) })} />
            </Labeled>
          </div>
        </div>
        <ErrList errors={[...errsFor(errors, "timeframe"), ...errsFor(errors, "universe")]} />
      </Section>

      <Section title="Entry">
        {(["long", "short"] as const).map((side) => (
          <SideRules key={side} side={side} cond={spec.entry[side]} daily={daily}
                     onChange={(c) => setIn("entry", { [side]: c } as Partial<Spec["entry"]>)}
                     errors={errsFor(errors, `entry.${side}`)} />
        ))}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {!daily && (
            <>
              <Labeled label="Entries from (IST)">
                <input aria-label="Entries from" type="time" className={cn(FIELD, "w-full")} value={spec.entry.time_from}
                       onChange={(e) => setIn("entry", { time_from: e.target.value })} />
              </Labeled>
              <Labeled label="Entries until (IST)">
                <input aria-label="Entries until" type="time" className={cn(FIELD, "w-full")} value={spec.entry.time_to}
                       onChange={(e) => setIn("entry", { time_to: e.target.value })} />
              </Labeled>
            </>
          )}
          <Labeled label="Cooldown (bars after an exit)">
            <Num value={spec.entry.cooldown_bars} onChange={(v) => setIn("entry", { cooldown_bars: v ?? 0 })} step={1} />
          </Labeled>
        </div>
        <ErrList errors={errsFor(errors, "entry").filter((e) => !e.path.startsWith("entry.long") && !e.path.startsWith("entry.short"))} />
      </Section>

      <Section title="Exits">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Labeled label="Stop">
            <div className="flex gap-1">
              <select aria-label="Stop kind" className={FIELD} value={spec.exits.stop.kind}
                      onChange={(e) => setIn("exits", { stop: { ...spec.exits.stop, kind: e.target.value as "atr" | "pct" } })}>
                <option value="atr">× ATR</option><option value="pct">% of price</option>
              </select>
              <Num aria="Stop value" value={spec.exits.stop.value} onChange={(v) => setIn("exits", { stop: { ...spec.exits.stop, value: v ?? 0 } })} />
            </div>
          </Labeled>
          <Labeled label="Target">
            <div className="flex gap-1">
              <select aria-label="Target kind" className={FIELD} value={spec.exits.target.kind}
                      onChange={(e) => setIn("exits", { target: { ...spec.exits.target, kind: e.target.value as "r" | "pct" | "none" } })}>
                <option value="r">R multiple</option><option value="pct">% of price</option><option value="none">none</option>
              </select>
              {spec.exits.target.kind !== "none" && (
                <Num aria="Target value" value={spec.exits.target.value ?? 2} onChange={(v) => setIn("exits", { target: { ...spec.exits.target, value: v ?? 0 } })} />
              )}
            </div>
          </Labeled>
          <Labeled label="Trail">
            <select aria-label="Trail" className={cn(FIELD, "w-full")} value={spec.exits.trail.kind}
                    onChange={(e) => setIn("exits", { trail: e.target.value === "atr"
                      ? { kind: "atr", trigger_r: 1, atr_mult: 1.5 } : { kind: e.target.value as "owner" | "none" } })}>
              <option value="owner">Owner rule (BE at 1R, then 1.5×ATR)</option>
              <option value="atr">Custom ATR trail</option>
              <option value="none">No trail</option>
            </select>
          </Labeled>
          <Labeled label="Breakeven at R (optional)">
            <Num value={spec.exits.breakeven_r} onChange={(v) => setIn("exits", { breakeven_r: v })} />
          </Labeled>
          {spec.exits.trail.kind === "atr" && (
            <>
              <Labeled label="Trail starts at R">
                <Num value={spec.exits.trail.trigger_r ?? 1} onChange={(v) => setIn("exits", { trail: { ...spec.exits.trail, trigger_r: v ?? 1 } })} />
              </Labeled>
              <Labeled label="Trail × ATR">
                <Num value={spec.exits.trail.atr_mult ?? 1.5} onChange={(v) => setIn("exits", { trail: { ...spec.exits.trail, atr_mult: v ?? 1.5 } })} />
              </Labeled>
            </>
          )}
          {!daily && (
            <Labeled label="Flat by (IST)">
              <input aria-label="Flat by" type="time" className={cn(FIELD, "w-full")} value={spec.exits.eod_time ?? "15:14"}
                     onChange={(e) => setIn("exits", { eod_time: e.target.value })} />
            </Labeled>
          )}
          <Labeled label="Max hold (bars, optional)">
            <Num value={spec.exits.max_hold_bars} step={1} onChange={(v) => setIn("exits", { max_hold_bars: v })} />
          </Labeled>
        </div>
        <ErrList errors={errsFor(errors, "exits")} />
      </Section>

      <Section title="Sizing, limits and costs">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Labeled label="Risk % per trade"><Num value={spec.sizing.risk_pct} onChange={(v) => setIn("sizing", { risk_pct: v ?? 0 })} /></Labeled>
          <Labeled label="Max position % of capital"><Num value={spec.sizing.max_position_pct} onChange={(v) => setIn("sizing", { max_position_pct: v ?? 0 })} /></Labeled>
          <Labeled label="Max open positions"><Num value={spec.sizing.max_positions} step={1} onChange={(v) => setIn("sizing", { max_positions: v ?? 1 })} /></Labeled>
          <Labeled label="Daily loss stop %"><Num value={spec.limits.daily_loss_pct} onChange={(v) => setIn("limits", { daily_loss_pct: v ?? 0 })} /></Labeled>
          <Labeled label="Max trades / day (0 = no cap)"><Num value={spec.limits.max_trades_per_day} step={1} onChange={(v) => setIn("limits", { max_trades_per_day: v ?? 0 })} /></Labeled>
          <Labeled label="Slippage % per side"><Num value={spec.costs.slippage_pct} onChange={(v) => setIn("costs", { slippage_pct: v ?? 0 })} /></Labeled>
          <label className="flex items-center gap-2 text-body-sm text-fg sm:col-span-2 pt-5">
            <input type="checkbox" checked={spec.limits.owner_cooldown}
                   onChange={(e) => setIn("limits", { owner_cooldown: e.target.checked })} />
            Owner cooldown (3 losing days → 2 days off → half risk)
          </label>
        </div>
        <p className="text-caption text-fg-subtle">Sizing can't exceed the live RiskEngine limits; costs use the NSE {daily ? "delivery" : "intraday (MIS)"} schedule.</p>
        <ErrList errors={[...errsFor(errors, "sizing"), ...errsFor(errors, "limits"), ...errsFor(errors, "costs")]} />
      </Section>

      <div>
        <Button type="button" size="sm" variant="ghost" onClick={() => setJson(json === null ? JSON.stringify(spec, null, 2) : null)}>
          {json === null ? "Edit as JSON" : "Close JSON editor"}
        </Button>
        {json !== null && (
          <div className="mt-2 space-y-1">
            <textarea aria-label="Spec JSON" rows={14} value={json} onChange={(e) => setJson(e.target.value)}
                      className="w-full rounded-sm border border-border bg-surface p-2 font-mono text-caption text-fg" />
            <Button type="button" size="sm" variant="secondary" onClick={() => {
              try { onChange(JSON.parse(json)); setJson(null); } catch { /* invalid JSON: keep editing */ }
            }}>Apply JSON</Button>
          </div>
        )}
      </div>
    </fieldset>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2">
      <h3 className="text-caption uppercase tracking-wider text-fg-subtle">{title}</h3>
      {children}
    </section>
  );
}

function Labeled({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-caption text-fg-muted mb-0.5">{label}</div>
      {children}
    </div>
  );
}

function Num({ value, onChange, step = 0.1, aria }: {
  value: number | null | undefined; onChange: (v: number | null) => void; step?: number; aria?: string;
}) {
  return (
    <input type="number" aria-label={aria} step={step} className={cn(FIELD, "w-full min-w-[4rem]")}
           value={value === null || value === undefined ? "" : value}
           onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))} />
  );
}

function SideRules({ side, cond, onChange, errors, daily }: {
  side: "long" | "short"; cond: Condition | null; onChange: (c: Condition | null) => void;
  errors: FieldError[]; daily: boolean;
}) {
  const flat = flatten(cond);
  const [raw, setRaw] = React.useState<string>(() => JSON.stringify(cond, null, 2));
  const title = side === "long" ? "Buy (long) when" : "Sell short when";
  if (side === "short" && daily) {
    return <p className="text-caption text-fg-subtle">Daily strategies are long-only (cash-delivery shorts can't be held overnight).</p>;
  }
  if (flat === null) {
    return (
      <div className="rounded-sm border border-border p-3 space-y-1">
        <div className="text-body-sm text-fg">{title} — advanced condition (JSON)</div>
        <textarea aria-label={`${side} condition JSON`} rows={6} value={raw} onChange={(e) => setRaw(e.target.value)}
                  onBlur={() => { try { onChange(JSON.parse(raw)); } catch { /* keep */ } }}
                  className="w-full rounded-sm border border-border bg-surface p-2 font-mono text-caption text-fg" />
        <ErrList errors={errors} />
      </div>
    );
  }
  const update = (f: Flat) => onChange(unflatten(f));
  return (
    <div className="rounded-sm border border-border p-3 space-y-2" data-testid={`rules-${side}`}>
      <div className="flex flex-wrap items-center gap-2 text-body-sm text-fg">
        <span className="font-medium">{title}</span>
        {flat.rows.length > 1 && (
          <select aria-label={`${side} join`} className={FIELD} value={flat.join}
                  onChange={(e) => update({ ...flat, join: e.target.value as Join })}>
            <option value="all">ALL of these are true</option>
            <option value="any">ANY of these is true</option>
          </select>
        )}
        {flat.rows.length === 0 && <span className="text-fg-subtle">— no {side} entries</span>}
      </div>
      {flat.rows.map((row, i) => (
        <RuleRow key={i} row={row} daily={daily}
                 onChange={(r) => update({ ...flat, rows: flat.rows.map((x, j) => (j === i ? r : x)) })}
                 onRemove={() => update({ ...flat, rows: flat.rows.filter((_, j) => j !== i) })} />
      ))}
      <Button type="button" size="sm" variant="ghost" leading={<Plus className="h-3.5 w-3.5" />}
              onClick={() => update({ ...flat, rows: [...flat.rows, emptyRow()] })}>
        Add {side} condition
      </Button>
      <ErrList errors={errors} />
    </div>
  );
}

function RuleRow({ row, onChange, onRemove, daily }: {
  row: Condition; onChange: (c: Condition) => void; onRemove: () => void; daily: boolean;
}) {
  const trend = row.op === "rising" || row.op === "falling";
  return (
    <div className="flex flex-wrap items-center gap-1.5" data-testid="rule-row">
      <OperandEditor value={row.left ?? { price: "close" }} onChange={(left) => onChange({ ...row, left })} daily={daily} />
      <select aria-label="Operator" className={FIELD} value={row.op}
              onChange={(e) => {
                const op = e.target.value;
                const isTrend = op === "rising" || op === "falling";
                onChange(isTrend ? { left: row.left, op, bars: row.bars ?? 3 }
                                 : { left: row.left, op, right: row.right ?? { value: 0 } });
              }}>
        {OPS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
      {trend ? (
        <span className="inline-flex items-center gap-1 text-body-sm">
          <input type="number" aria-label="Bars" min={1} className={cn(FIELD, "w-16")} value={row.bars ?? 1}
                 onChange={(e) => onChange({ ...row, bars: Number(e.target.value) || 1 })} /> bars
        </span>
      ) : (
        <OperandEditor value={row.right ?? { value: 0 }} onChange={(right) => onChange({ ...row, right })} daily={daily} />
      )}
      <Button type="button" size="icon" variant="ghost" aria-label="Remove condition" onClick={onRemove}>
        <Trash2 className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}

type OpType = "price" | "value" | "ind" | "ref";

function OperandEditor({ value, onChange, daily }: { value: Operand; onChange: (o: Operand) => void; daily: boolean }) {
  const type: OpType = value.ref !== undefined ? "ref" : value.ind !== undefined ? "ind" : value.value !== undefined ? "value" : "price";
  const kinds = Object.entries(INDICATOR_KINDS).filter(([, k]) => !(daily && k.intraday));
  const extras = (o: Operand): Operand => ({ ...o, ...(value.bars_ago ? { bars_ago: value.bars_ago } : {}), ...(value.mult ? { mult: value.mult } : {}) });
  return (
    <span className="inline-flex flex-wrap items-center gap-1 rounded-sm bg-surface-2 px-1.5 py-1">
      <select aria-label="Operand type" className={FIELD} value={type}
              onChange={(e) => {
                const t = e.target.value as OpType;
                onChange(t === "price" ? { price: "close" } : t === "value" ? { value: 0 }
                  : { ind: "ema", ...INDICATOR_KINDS.ema.params });
              }}>
        <option value="price">Price</option>
        <option value="ind">Indicator</option>
        <option value="value">Number</option>
        {type === "ref" && <option value="ref">Named</option>}
      </select>
      {type === "price" && (
        <select aria-label="Price field" className={FIELD} value={value.price}
                onChange={(e) => onChange(extras({ price: e.target.value as PriceField }))}>
          {PRICES.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
      )}
      {type === "value" && (
        <input type="number" aria-label="Number" step="any" className={cn(FIELD, "w-20")} value={value.value ?? 0}
               onChange={(e) => onChange({ value: Number(e.target.value) })} />
      )}
      {type === "ref" && <span className="font-mono text-caption px-1">{value.ref}</span>}
      {type === "ind" && (
        <>
          <select aria-label="Indicator" className={FIELD} value={value.ind}
                  onChange={(e) => onChange(extras({ ind: e.target.value, ...INDICATOR_KINDS[e.target.value].params }))}>
            {kinds.map(([k, d]) => <option key={k} value={k}>{d.label}</option>)}
          </select>
          {"source" in (INDICATOR_KINDS[value.ind ?? ""]?.params ?? {}) && (
            <select aria-label="Source" className={FIELD} value={value.source}
                    onChange={(e) => onChange({ ...value, source: e.target.value as PriceField })}>
              {PRICES.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          )}
          {"period" in (INDICATOR_KINDS[value.ind ?? ""]?.params ?? {}) && (
            <input type="number" aria-label="Period" min={1} className={cn(FIELD, "w-16")} value={value.period ?? 14}
                   onChange={(e) => onChange({ ...value, period: Number(e.target.value) || 1 })} />
          )}
          {"std" in (INDICATOR_KINDS[value.ind ?? ""]?.params ?? {}) && (
            <input type="number" aria-label="Std" step="0.1" className={cn(FIELD, "w-16")} value={value.std ?? 2}
                   onChange={(e) => onChange({ ...value, std: Number(e.target.value) })} />
          )}
          {"minutes" in (INDICATOR_KINDS[value.ind ?? ""]?.params ?? {}) && (
            <input type="number" aria-label="Minutes" min={1} className={cn(FIELD, "w-16")} value={value.minutes ?? 15}
                   onChange={(e) => onChange({ ...value, minutes: Number(e.target.value) || 15 })} />
          )}
        </>
      )}
      {type !== "value" && (
        <>
          <input type="number" aria-label="Multiplier" step="0.1" min={0} title="multiplier (blank = 1)"
                 placeholder="×1" className={cn(FIELD, "w-14")} value={value.mult ?? ""}
                 onChange={(e) => onChange({ ...value, mult: e.target.value === "" ? undefined : Number(e.target.value) })} />
          <input type="number" aria-label="Bars ago" min={0} title="bars ago (blank = this bar)"
                 placeholder="now" className={cn(FIELD, "w-14")} value={value.bars_ago ?? ""}
                 onChange={(e) => onChange({ ...value, bars_ago: e.target.value === "" ? undefined : Number(e.target.value) })} />
        </>
      )}
    </span>
  );
}
