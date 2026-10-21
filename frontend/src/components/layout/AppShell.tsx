import * as React from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { ChevronDown, LogOut, Moon, Search, Settings, Sun } from "lucide-react";
import { useAuthStore } from "@/stores/auth";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { CommandPalette } from "@/components/ui/CommandPalette";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/Tooltip";
import { StatusStrip } from "@/components/layout/StatusStrip";
import { connect } from "@/lib/ws";
import { DESTINATIONS, locate, type Destination } from "@/lib/navigation";

type LiveStatus = "connecting" | "live" | "offline";

export function AppShell() {
  const { userEmail, clear } = useAuthStore();
  const [theme, setTheme] = React.useState<"dark" | "light">(() => {
    const saved = (typeof window !== "undefined" && localStorage.getItem("theme")) as "dark" | "light" | null;
    return saved ?? "dark";
  });
  const [status, setStatus] = React.useState<LiveStatus>("connecting");

  // Apply theme to <html>
  React.useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  // Heartbeat WS — purely for the connection pill
  React.useEffect(() => {
    setStatus("connecting");
    const ws = connect("/ws/alerts/", () => {}, {
      onOpen: () => setStatus("live"),
      onClose: () => setStatus("offline"),
      onError: () => setStatus("offline"),
    });
    return () => ws.close();
  }, []);

  return (
    <div className="min-h-screen grid grid-cols-[240px_1fr] bg-bg text-fg">
      <Sidebar onSignOut={clear} userEmail={userEmail} />
      <div className="flex flex-col min-w-0">
        <TopBar status={status} theme={theme} onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")} />
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
      <CommandPalette />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Sidebar                                                             */
/* ------------------------------------------------------------------ */
function Sidebar({ userEmail, onSignOut }: { userEmail: string | null; onSignOut: () => void }) {
  return (
    <aside className="bg-surface border-r border-border flex flex-col min-h-screen sticky top-0">
      <div className="px-5 h-16 flex items-center gap-2 border-b border-border">
        <Logo />
        <div className="leading-tight">
          <div className="font-display text-body font-semibold tracking-tight text-fg">AlphaDesk</div>
          <div className="text-caption text-fg-subtle -mt-0.5">AI trading desk</div>
        </div>
      </div>

      <nav className="flex-1 p-3 space-y-0.5 overflow-y-auto" aria-label="Primary">
        {DESTINATIONS.map((d) => <DestinationGroup key={d.key} d={d} />)}
      </nav>

      <div className="p-3 border-t border-border space-y-2">
        <div className="flex items-center gap-2 text-body-sm text-fg-muted">
          <Settings className="h-4 w-4" aria-hidden />
          <span className="truncate" title={userEmail ?? ""}>{userEmail ?? "—"}</span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-start"
          leading={<LogOut className="h-4 w-4" />}
          onClick={onSignOut}
        >
          Sign out
        </Button>
      </div>
    </aside>
  );
}

/** One of the 6 destinations; its old pages are sub-items (expanded when active). */
function DestinationGroup({ d }: { d: Destination }) {
  const loc = useLocation();
  const active = locate(loc.pathname).destination?.key === d.key;
  const [open, setOpen] = React.useState(active);
  React.useEffect(() => { if (active) setOpen(true); }, [active]);
  const Icon = d.icon;
  const hasItems = d.items.length > 1 || (d.items.length === 1 && d.items[0].to !== d.to);

  return (
    <div>
      <div
        className={cn(
          "group flex items-center rounded-sm h-9 border",
          active ? "bg-accent/10 text-fg border-accent/30" : "text-fg-muted border-transparent hover:bg-surface-2 hover:text-fg",
        )}
      >
        <NavLink to={d.to} end={d.to === "/"} title={d.description}
                 className="flex flex-1 items-center gap-3 px-3 h-full text-body-sm font-medium">
          <Icon className="h-4 w-4" aria-hidden />
          <span className="flex-1">{d.label}</span>
        </NavLink>
        {hasItems && (
          <button type="button" aria-label={`${open ? "Collapse" : "Expand"} ${d.label}`}
                  aria-expanded={open} onClick={() => setOpen((o) => !o)}
                  className="h-full px-2 text-fg-subtle hover:text-fg">
            <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", !open && "-rotate-90")} aria-hidden />
          </button>
        )}
      </div>
      {hasItems && open && (
        <ul className="mt-0.5 mb-1 ml-5 border-l border-border pl-2 space-y-0.5" aria-label={`${d.label} pages`}>
          {d.items.map(({ to, label, icon: ItemIcon, hint }) => (
            <li key={to}>
              <NavLink
                to={to}
                className={({ isActive }) => cn(
                  "group flex items-center gap-2 rounded-sm px-2 h-8 text-body-sm transition-[background-color,color] duration-120",
                  isActive ? "text-fg bg-surface-2" : "text-fg-muted hover:bg-surface-2 hover:text-fg",
                )}
              >
                <ItemIcon className="h-3.5 w-3.5" aria-hidden />
                <span className="flex-1 truncate">{label}</span>
                {hint && (
                  <kbd className="text-caption text-fg-subtle font-mono opacity-0 group-hover:opacity-100">{hint}</kbd>
                )}
              </NavLink>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Logo() {
  return (
    <span
      aria-hidden
      className="inline-flex items-center justify-center h-8 w-8 rounded-md bg-gradient-to-br from-accent to-info text-accent-fg font-display font-bold"
    >
      α
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Top bar                                                             */
/* ------------------------------------------------------------------ */
function TopBar({
  status, theme, onToggleTheme,
}: { status: LiveStatus; theme: "dark" | "light"; onToggleTheme: () => void }) {
  const loc = useLocation();
  const { destination, item } = locate(loc.pathname);
  const title = destination
    ? (item && item.label !== destination.label ? `${destination.label} · ${item.label}` : destination.label)
    : "AlphaDesk";

  return (
    <header className="h-14 flex items-center gap-3 border-b border-border bg-bg/80 backdrop-blur-sm px-5 sticky top-0 z-sticky">
      <h1 className="text-body font-semibold tracking-tight text-fg">{title}</h1>

      <StatusStrip />

      <div className="flex-1 max-w-md mx-auto">
        <button
          type="button"
          onClick={() => {
            // Trigger the same shortcut the CommandPalette listens for
            const ev = new KeyboardEvent("keydown", { key: "k", metaKey: true, bubbles: true });
            document.dispatchEvent(ev);
          }}
          className={cn(
            "w-full inline-flex items-center gap-2 h-9 rounded-sm px-3",
            "border border-border bg-surface hover:bg-surface-2",
            "text-body-sm text-fg-subtle",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
          )}
        >
          <Search className="h-4 w-4" aria-hidden />
          <span className="flex-1 text-left">Search commands…</span>
          <kbd className="text-caption text-fg-subtle border border-border rounded-xs px-1.5 py-0.5 font-mono">
            ⌘K
          </kbd>
        </button>
      </div>

      <ConnectionPill status={status} />
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="ghost" size="icon" onClick={onToggleTheme} aria-label="Toggle theme">
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </TooltipTrigger>
        <TooltipContent>Toggle theme</TooltipContent>
      </Tooltip>
    </header>
  );
}

function ConnectionPill({ status }: { status: LiveStatus }) {
  const map = {
    live:       { tone: "success" as const, label: "Live" },
    connecting: { tone: "warning" as const, label: "Connecting" },
    offline:    { tone: "danger"  as const, label: "Offline" },
  }[status];
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex">
          <Badge tone={map.tone} dot aria-live="polite">
            {map.label}
          </Badge>
        </span>
      </TooltipTrigger>
      <TooltipContent>
        WebSocket {map.label.toLowerCase()} — live ticks &amp; agent events.
      </TooltipContent>
    </Tooltip>
  );
}
