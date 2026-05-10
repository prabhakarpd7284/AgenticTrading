import * as React from "react";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Skeleton } from "./Skeleton";
import { EmptyState } from "./EmptyState";

/**
 * DataTable — lightweight, accessible table for medium-sized result sets.
 * For ≥200 rows, wire a virtualiser (e.g. @tanstack/virtual) around `renderRow`.
 *
 * Column types (`kind`) pre-apply correct alignment + number formatting.
 */

export type ColumnKind = "text" | "num" | "money" | "delta" | "timestamp" | "status" | "action";

export interface Column<T> {
  key: keyof T | string;
  header: string;
  kind?: ColumnKind;
  width?: string;
  sortable?: boolean;
  render?: (row: T) => React.ReactNode;
  align?: "left" | "right" | "center";
  scope?: "col";
}

export interface DataTableProps<T> {
  columns: Column<T>[];
  rows: T[];
  loading?: boolean;
  emptyTitle?: string;
  emptyDescription?: string;
  dense?: boolean;
  rowKey: (row: T, i: number) => string;
  onRowClick?: (row: T) => void;
  sort?: { key: string; dir: "asc" | "desc" };
  onSort?: (key: string) => void;
  caption?: string;
}

export function DataTable<T>({
  columns, rows, loading, dense, rowKey, onRowClick,
  emptyTitle = "No results", emptyDescription,
  sort, onSort, caption,
}: DataTableProps<T>) {

  if (loading) return <TableSkeleton rows={6} cols={columns.length} dense={dense} />;
  // Defensive: a paginated envelope or a failed request can slip in a non-
  // array `rows`. Fall back to an empty state rather than crashing the page
  // on `rows.length` / `rows.map(...)`.
  const safeRows: T[] = Array.isArray(rows) ? rows : [];
  if (safeRows.length === 0) return <EmptyState title={emptyTitle} description={emptyDescription} />;

  return (
    <div className="overflow-x-auto rounded-md border border-border bg-surface">
      <table className="w-full border-collapse" role="table">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead className="bg-surface-2">
          <tr>
            {columns.map((c) => {
              const ariaSort =
                sort?.key === c.key ? (sort.dir === "asc" ? "ascending" : "descending") : c.sortable ? "none" : undefined;
              return (
                <th
                  key={String(c.key)}
                  scope="col"
                  aria-sort={ariaSort as any}
                  style={c.width ? { width: c.width } : undefined}
                  className={cn(
                    "sticky top-0 z-sticky border-b border-border px-3 py-2 text-caption uppercase tracking-wider text-fg-subtle",
                    c.align === "right" || c.kind === "num" || c.kind === "money" || c.kind === "delta" ? "text-right" : "text-left",
                    c.sortable && "cursor-pointer select-none hover:text-fg",
                  )}
                  onClick={c.sortable ? () => onSort?.(String(c.key)) : undefined}
                >
                  <span className="inline-flex items-center gap-1">
                    {c.header}
                    {c.sortable && <SortIcon active={sort?.key === c.key} dir={sort?.dir} />}
                  </span>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {safeRows.map((row, i) => (
            <tr
              key={rowKey(row, i)}
              tabIndex={onRowClick ? 0 : undefined}
              role={onRowClick ? "button" : undefined}
              onClick={() => onRowClick?.(row)}
              onKeyDown={(e) => onRowClick && (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onRowClick(row))}
              className={cn(
                "border-b border-border/60 last:border-none",
                onRowClick && "hover:bg-surface-2 focus:bg-surface-2 focus-visible:outline-none",
                dense ? "h-8" : "h-10",
              )}
            >
              {columns.map((c) => {
                const raw = (row as any)[c.key];
                const content = c.render
                  ? c.render(row)
                  : c.kind === "num" || c.kind === "money"
                  ? <span className="font-mono tabular">{raw}</span>
                  : raw;
                return (
                  <td
                    key={String(c.key)}
                    className={cn(
                      "px-3 text-body-sm text-fg align-middle",
                      c.align === "right" || c.kind === "num" || c.kind === "money" || c.kind === "delta" ? "text-right" : "text-left",
                    )}
                  >
                    {content}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SortIcon({ active, dir }: { active?: boolean; dir?: "asc" | "desc" }) {
  if (!active) return <ArrowUpDown className="h-3 w-3" aria-hidden />;
  return dir === "asc"
    ? <ArrowUp className="h-3 w-3" aria-hidden />
    : <ArrowDown className="h-3 w-3" aria-hidden />;
}

function TableSkeleton({ rows, cols, dense }: { rows: number; cols: number; dense?: boolean }) {
  return (
    <div className="rounded-md border border-border overflow-hidden">
      <div className="grid gap-2 p-3 bg-surface-2" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
        {Array.from({ length: cols }).map((_, i) => <Skeleton key={i} className="h-3 w-20" />)}
      </div>
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} className={cn("grid gap-2 p-3 border-t border-border/60", dense && "py-2")} style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
          {Array.from({ length: cols }).map((_, c) => <Skeleton key={c} className="h-3 w-full" />)}
        </div>
      ))}
    </div>
  );
}
