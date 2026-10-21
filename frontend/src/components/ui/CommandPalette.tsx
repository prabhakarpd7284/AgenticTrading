import * as React from "react";
import { Command } from "cmdk";
import * as DialogPrim from "@radix-ui/react-dialog";
import { useNavigate } from "react-router-dom";
import { BarChart3, History, Play, Search, TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";
import { DESTINATIONS } from "@/lib/navigation";

/**
 * CommandPalette — ⌘K launcher.
 * Action list is static for first cut; swap with a hook fed by tenant+role.
 */
export function CommandPalette() {
  const [open, setOpen] = React.useState(false);
  const navigate = useNavigate();

  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, []);

  const go = (path: string) => { navigate(path); setOpen(false); };

  return (
    <DialogPrim.Root open={open} onOpenChange={setOpen}>
      <DialogPrim.Portal>
        <DialogPrim.Overlay className="fixed inset-0 z-dialog bg-black/60 backdrop-blur-[2px] data-[state=open]:animate-fade-in" />
        <DialogPrim.Content
          className="fixed left-1/2 top-[18%] -translate-x-1/2 z-dialog w-[min(92vw,600px)] rounded-lg border border-border bg-surface shadow-lg data-[state=open]:animate-slide-up"
          aria-label="Command palette"
        >
          <Command label="Command palette" className="flex flex-col">
            <div className="flex items-center gap-2 px-3 h-11 border-b border-border">
              <Search className="h-4 w-4 text-fg-muted" aria-hidden />
              <Command.Input
                placeholder="Search commands, positions, journal entries…"
                className="flex-1 bg-transparent outline-none text-body text-fg placeholder:text-fg-subtle"
              />
              <kbd className="text-caption text-fg-subtle border border-border rounded-xs px-1.5 py-0.5 font-mono">esc</kbd>
            </div>

            <Command.List className="max-h-[420px] overflow-auto p-2">
              <Command.Empty className="py-8 text-center text-body-sm text-fg-muted">
                No matches. Try a different term.
              </Command.Empty>

              {/* Every destination and every page under it (plan §2: old pages stay searchable). */}
              {DESTINATIONS.map((d) => (
                <Group key={d.key} heading={d.label}>
                  <Item icon={<d.icon />} value={`${d.label} ${d.description}`} onSelect={() => go(d.to)}>
                    {d.label} · {d.description}
                  </Item>
                  {d.items
                    .filter((i) => !(i.to === d.to && i.label === d.label))
                    .map((i) => (
                      <Item key={i.to} icon={<i.icon />} value={`${d.label} ${i.label} ${i.keywords ?? ""} ${i.to}`}
                            onSelect={() => go(i.to)}>
                        {i.label}
                      </Item>
                    ))}
                </Group>
              ))}

              <Group heading="Actions">
                <Item icon={<Play/>}         shortcut="n" onSelect={() => go("/agents?new=1")}>New agent run</Item>
                <Item icon={<BarChart3/>}    shortcut="b" onSelect={() => go("/backtester?new=1")}>New backtest</Item>
                <Item icon={<TrendingUp/>}   shortcut="s" onSelect={() => go("/swing-scanner")}>Scan active phases</Item>
                <Item icon={<History/>}      onSelect={() => go("/runs")}>Did today's jobs run? · Run receipts</Item>
              </Group>
            </Command.List>
          </Command>
        </DialogPrim.Content>
      </DialogPrim.Portal>
    </DialogPrim.Root>
  );
}

function Group({ heading, children }: { heading: string; children: React.ReactNode }) {
  return (
    <Command.Group heading={heading} className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-caption [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-fg-subtle">
      {children}
    </Command.Group>
  );
}

function Item({ icon, shortcut, children, onSelect, value }: {
  icon: React.ReactNode; shortcut?: string; children: React.ReactNode; onSelect: () => void; value?: string;
}) {
  return (
    <Command.Item
      onSelect={onSelect}
      value={value}
      className={cn(
        "flex items-center gap-3 px-2 py-2 rounded-xs text-body text-fg cursor-pointer",
        "data-[selected=true]:bg-surface-2",
      )}
    >
      <span className="text-fg-muted [&>svg]:h-4 [&>svg]:w-4" aria-hidden>{icon}</span>
      <span className="flex-1">{children}</span>
      {shortcut && <kbd className="text-caption text-fg-subtle border border-border rounded-xs px-1.5 py-0.5 font-mono">{shortcut}</kbd>}
    </Command.Item>
  );
}
