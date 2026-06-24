import * as React from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  Activity, BarChart3, Bot, Briefcase, Calendar, Gauge, HeartPulse, Layers, Link2,
  LineChart, ListChecks, LogOut, Moon, Radio, Search, Settings, ShoppingBag, Sigma,
  Sun, Terminal, TrendingUp, Triangle, Workflow,
} from "lucide-react";
import { useAuthStore } from "@/stores/auth";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { CommandPalette } from "@/components/ui/CommandPalette";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/Tooltip";
import { connect } from "@/lib/ws";

const nav = [
  { to: "/pulse",      label: "Today",        icon: Gauge,      hint: "g t" },
  { to: "/rotation",   label: "Rotation",     icon: Layers,     hint: "g r" },
  { to: "/shortlist",  label: "Shortlist",    icon: ListChecks, hint: "g l" },
  { to: "/swing-scanner", label: "Swing Scanner", icon: TrendingUp, hint: "g w" },
  { to: "/basket",     label: "Basket",       icon: ShoppingBag, hint: "g x" },
  { to: "/dashboard",  label: "Desk",         icon: Activity,   hint: "g d" },
  { to: "/positions",  label: "Positions",    icon: Briefcase, hint: "g p" },
  { to: "/options",    label: "Options Desk", icon: Sigma,     hint: "g e" },
  { to: "/monthly",    label: "Monthly",      icon: Calendar,  hint: "g m" },
  { to: "/agents",     label: "Agent Console",icon: Bot,       hint: "g a" },
  { to: "/strategies", label: "Strategies",   icon: LineChart, hint: "g s" },
  { to: "/backtester", label: "Backtester",   icon: BarChart3, hint: "g b" },
  { to: "/pyramid",   label: "Pyramid",      icon: Triangle,   hint: "g y" },
  { to: "/brokers",    label: "Broker Link",  icon: Link2,     hint: "g k" },
  { to: "/broker-monitor", label: "Broker Monitor", icon: HeartPulse, hint: "g n" },
  { to: "/tradingview",label: "TradingView",  icon: Radio,     hint: "g v" },
  { to: "/ops",        label: "Ops Console",  icon: Terminal,  hint: "g o" },
  { to: "/pipeline",   label: "Daily Pipeline", icon: Workflow, hint: "g i" },
];

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

      <nav className="flex-1 p-3 space-y-0.5" aria-label="Primary">
        {nav.map(({ to, label, icon: Icon, hint }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "group flex items-center gap-3 rounded-sm px-3 h-9 text-body-sm",
                "transition-[background-color,color] duration-120",
                isActive
                  ? "bg-accent/10 text-fg border border-accent/30"
                  : "text-fg-muted hover:bg-surface-2 hover:text-fg border border-transparent",
              )
            }
          >
            <Icon className="h-4 w-4" aria-hidden />
            <span className="flex-1">{label}</span>
            <kbd className="text-caption text-fg-subtle font-mono opacity-0 group-hover:opacity-100">
              {hint}
            </kbd>
          </NavLink>
        ))}
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
  const path = loc.pathname === "/" ? "/pulse" : loc.pathname;
  const title = nav.find((n) => path.startsWith(n.to))?.label ?? "AlphaDesk";

  return (
    <header className="h-14 flex items-center gap-3 border-b border-border bg-bg/80 backdrop-blur-sm px-5 sticky top-0 z-sticky">
      <h1 className="text-body font-semibold tracking-tight text-fg">{title}</h1>

      <div className="flex-1 max-w-xl mx-auto">
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
