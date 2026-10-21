/**
 * SchemaForm — renders a plugin's JSON-Schema `schema()` params as a form.
 *
 * Supported (what the registered plugins use): string (text / enum select /
 * nullable), number + integer (with min/max), boolean, and arrays of strings
 * (enum → checkboxes, free → comma-separated). Unknown shapes fall back to a
 * JSON text field so nothing is silently dropped. Client-side checks mirror
 * the schema; the server re-validates every save.
 */
import * as React from "react";
import { AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { FieldError, JsonSchema } from "./useStrategies";

type Values = Record<string, unknown>;

const FIELD =
  "h-9 w-full rounded-sm border border-border bg-surface px-3 text-body-sm text-fg " +
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 disabled:opacity-60";

function types(s: JsonSchema): string[] {
  return Array.isArray(s.type) ? s.type : s.type ? [s.type] : [];
}

function label(key: string) {
  return key.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

/** Client-side errors for the subset of JSON-Schema we render. */
export function schemaErrors(schema: JsonSchema | null | undefined, values: Values): FieldError[] {
  const out: FieldError[] = [];
  if (!schema?.properties) return out;
  for (const key of schema.required ?? []) {
    const v = values[key];
    if (v === undefined || v === null || v === "") out.push({ path: key, message: "required" });
  }
  for (const [key, prop] of Object.entries(schema.properties)) {
    const v = values[key];
    if (v === undefined || v === null || v === "") continue;
    const t = types(prop);
    if ((t.includes("number") || t.includes("integer")) && typeof v === "number") {
      if (t.includes("integer") && !Number.isInteger(v)) out.push({ path: key, message: "must be a whole number" });
      if (prop.minimum !== undefined && v < prop.minimum) out.push({ path: key, message: `minimum ${prop.minimum}` });
      if (prop.maximum !== undefined && v > prop.maximum) out.push({ path: key, message: `maximum ${prop.maximum}` });
    }
    if (prop.enum && !Array.isArray(v) && !prop.enum.includes(v as string)) {
      out.push({ path: key, message: `one of ${prop.enum.join(", ")}` });
    }
  }
  return out;
}

export function SchemaForm({
  schema, value, onChange, errors = [], disabled, broken = {}, compact,
}: {
  schema: JsonSchema | null | undefined;
  value: Values;
  onChange: (v: Values) => void;
  errors?: FieldError[];
  disabled?: boolean;
  /** option value → reason it is broken (e.g. screener setups that never fire) */
  broken?: Record<string, string>;
  compact?: boolean;
}) {
  const props = schema?.properties ?? {};
  const keys = Object.keys(props);
  if (keys.length === 0) {
    return <p className="text-body-sm text-fg-muted">This strategy has no parameters.</p>;
  }
  const errFor = (k: string) => errors.filter((e) => e.path === k || e.path.startsWith(`${k}.`)).map((e) => e.message);
  const set = (k: string, v: unknown) => {
    const next = { ...value };
    if (v === undefined) delete next[k];
    else next[k] = v;
    onChange(next);
  };

  return (
    <div className={cn("grid gap-3", compact ? "grid-cols-1" : "grid-cols-1 sm:grid-cols-2")} data-testid="schema-form">
      {keys.map((k) => (
        <Field key={k} name={k} prop={props[k]} value={value[k]} required={(schema?.required ?? []).includes(k)}
               onChange={(v) => set(k, v)} errors={errFor(k)} disabled={disabled} broken={broken} />
      ))}
    </div>
  );
}

function Field({
  name, prop, value, onChange, errors, required, disabled, broken,
}: {
  name: string;
  prop: JsonSchema;
  value: unknown;
  onChange: (v: unknown) => void;
  errors: string[];
  required: boolean;
  disabled?: boolean;
  broken: Record<string, string>;
}) {
  const id = React.useId();
  const t = types(prop);
  const nullable = t.includes("null");
  const placeholder = prop.default !== undefined && prop.default !== null ? `default: ${String(prop.default)}` : undefined;
  let control: React.ReactNode;

  if (prop.enum && !t.includes("array")) {
    control = (
      <select id={id} className={FIELD} disabled={disabled} value={value === undefined || value === null ? "" : String(value)}
              onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)}>
        <option value="">{placeholder ?? "—"}</option>
        {prop.enum.map((o) => <option key={String(o)} value={String(o)}>{String(o)}</option>)}
      </select>
    );
  } else if (t.includes("boolean")) {
    control = (
      <label className="flex items-center gap-2 h-9 text-body-sm text-fg">
        <input id={id} type="checkbox" disabled={disabled}
               checked={value === undefined ? Boolean(prop.default) : Boolean(value)}
               onChange={(e) => onChange(e.target.checked)} />
        {value === undefined ? "(default)" : value ? "on" : "off"}
      </label>
    );
  } else if (t.includes("number") || t.includes("integer")) {
    control = (
      <input id={id} type="number" className={FIELD} disabled={disabled} placeholder={placeholder}
             min={prop.minimum} max={prop.maximum} step={t.includes("integer") ? 1 : "any"}
             value={typeof value === "number" ? value : ""}
             onChange={(e) => onChange(e.target.value === "" ? undefined : Number(e.target.value))} />
    );
  } else if (t.includes("array") && prop.items?.enum) {
    const cur = Array.isArray(value) ? (value as string[]) : [];
    control = (
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {prop.items.enum.map((o) => (
          <label key={String(o)} className="flex items-center gap-1.5 text-body-sm text-fg">
            <input type="checkbox" disabled={disabled} checked={cur.includes(String(o))}
                   onChange={(e) => onChange(e.target.checked ? [...cur, String(o)] : cur.filter((x) => x !== o))} />
            {String(o)}
          </label>
        ))}
      </div>
    );
  } else if (t.includes("array")) {
    const text = Array.isArray(value) ? (value as unknown[]).join(", ") : "";
    control = (
      <ListInput id={id} disabled={disabled} text={text}
                 onCommit={(list) => onChange(list.length ? list : undefined)} />
    );
  } else if (t.includes("string") || t.length === 0) {
    control = (
      <input id={id} type="text" className={FIELD} disabled={disabled} placeholder={placeholder ?? (nullable ? "empty = none" : "")}
             value={typeof value === "string" ? value : ""}
             onChange={(e) => onChange(e.target.value === "" ? (nullable ? null : undefined) : e.target.value)} />
    );
  } else {
    control = (
      <input id={id} type="text" className={FIELD} disabled={disabled} value={value === undefined ? "" : JSON.stringify(value)}
             onChange={(e) => { try { onChange(JSON.parse(e.target.value)); } catch { /* keep typing */ } }} />
    );
  }

  // Broken sub-strategies are flagged on the fields that select them.
  const brokenHits = name.includes("strateg") ? Object.entries(broken) : [];
  return (
    <div className="min-w-0">
      <label htmlFor={id} className="text-body-sm text-fg mb-1 inline-block">
        {label(name)}{required && <span className="text-pnl-down"> *</span>}
      </label>
      {control}
      {prop.description && <p className="text-caption text-fg-subtle mt-0.5">{prop.description}</p>}
      {brokenHits.length > 0 && (
        <ul className="mt-1 space-y-0.5">
          {brokenHits.map(([n, why]) => (
            <li key={n} className="flex gap-1 text-caption text-warn">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-px" aria-hidden />
              <span><b>{n}</b> is broken — {why}.</span>
            </li>
          ))}
        </ul>
      )}
      {errors.map((m) => <p key={m} role="alert" className="text-caption text-pnl-down mt-0.5">{m}</p>)}
    </div>
  );
}

function ListInput({ id, text, onCommit, disabled }: {
  id: string; text: string; onCommit: (v: string[]) => void; disabled?: boolean;
}) {
  const [draft, setDraft] = React.useState(text);
  React.useEffect(() => setDraft(text), [text]);
  return (
    <input id={id} type="text" className={FIELD} disabled={disabled} value={draft} placeholder="comma-separated"
           onChange={(e) => setDraft(e.target.value)}
           onBlur={() => onCommit(draft.split(",").map((s) => s.trim()).filter(Boolean))} />
  );
}
