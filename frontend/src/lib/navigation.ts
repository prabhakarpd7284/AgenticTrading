/**
 * Information architecture (UX redesign R0, plan §2): 6 destinations.
 *
 * Every pre-redesign page is still here as a sub-item of its destination —
 * nothing is deleted and every old URL keeps working. The sidebar, the top
 * bar title and the command palette all read this one list.
 */
import type { LucideIcon } from "lucide-react";
import {
  Activity, BarChart3, Bot, Briefcase, Calendar, CandlestickChart, FlaskConical, Gauge,
  HeartPulse, Home, Landmark, Layers, LineChart, Link2, ListChecks, MoonStar, PieChart, Radio,
  Settings, ShoppingBag, Sigma, Terminal, TrendingUp, Triangle, Workflow, History,
} from "lucide-react";

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  /** Extra palette keywords. */
  keywords?: string;
  hint?: string;
}

export interface Destination {
  key: "today" | "portfolio" | "strategies" | "lab" | "runs" | "settings";
  label: string;
  /** Where clicking the destination itself goes. */
  to: string;
  icon: LucideIcon;
  description: string;
  items: NavItem[];
}

export const DESTINATIONS: Destination[] = [
  {
    key: "today", label: "Today", to: "/", icon: Home,
    description: "What to do now — market context and where today's ideas come from",
    items: [
      { to: "/pulse", label: "Market Pulse", icon: Gauge, hint: "g t", keywords: "today what's happening regime" },
      { to: "/rotation", label: "Rotation", icon: Layers, hint: "g r", keywords: "sector drill-in" },
      { to: "/shortlist", label: "Shortlist", icon: ListChecks, hint: "g l", keywords: "tradeable names" },
      { to: "/swing-scanner", label: "Swing Scanner", icon: TrendingUp, hint: "g w", keywords: "oliver kell ok cycles" },
      { to: "/basket", label: "Basket", icon: ShoppingBag, hint: "g x", keywords: "morning premarket" },
      { to: "/nightly-scan", label: "Nightly Scan", icon: MoonStar, hint: "g h", keywords: "news sentiment cooldown levels" },
    ],
  },
  {
    key: "portfolio", label: "Portfolio", to: "/dashboard", icon: PieChart,
    description: "Everything you hold — equity, options, mutual funds, performance",
    items: [
      { to: "/dashboard", label: "Desk", icon: Activity, hint: "g d", keywords: "overview capital" },
      { to: "/positions", label: "Positions", icon: Briefcase, hint: "g p" },
      { to: "/options", label: "Options Desk", icon: Sigma, hint: "g e", keywords: "chain straddle" },
      { to: "/monthly", label: "Performance (Monthly)", icon: Calendar, hint: "g m", keywords: "earning tracker monthly" },
      { to: "/mutual-funds", label: "Mutual Funds", icon: Landmark, hint: "g f", keywords: "mf nav cas screener" },
    ],
  },
  {
    key: "strategies", label: "Strategies", to: "/strategies", icon: LineChart,
    description: "Create, backtest and simulate strategies",
    items: [
      { to: "/strategies", label: "Strategies", icon: LineChart, hint: "g s", keywords: "playbook" },
      { to: "/backtester", label: "Backtester", icon: BarChart3, hint: "g b" },
      { to: "/pyramid", label: "Pyramid", icon: Triangle, hint: "g y" },
      { to: "/scalping", label: "Scalping", icon: CandlestickChart, hint: "g c", keywords: "simulator" },
    ],
  },
  {
    key: "lab", label: "Lab", to: "/lab", icon: FlaskConical,
    description: "Stock group × strategy × period analysis (coming in R5)",
    items: [],
  },
  {
    key: "runs", label: "Runs", to: "/runs", icon: History,
    description: "Every job and run, with receipts — did it run correctly?",
    items: [
      { to: "/runs", label: "Run receipts", icon: History, hint: "g u", keywords: "jobs schedule health" },
      { to: "/pipeline", label: "Daily Pipeline", icon: Workflow, hint: "g i" },
      { to: "/agents", label: "Agent Console", icon: Bot, hint: "g a" },
      { to: "/ops", label: "Ops Console", icon: Terminal, hint: "g o", keywords: "manage.py commands" },
    ],
  },
  {
    key: "settings", label: "Settings", to: "/brokers", icon: Settings,
    description: "Broker links, broker telemetry, TradingView webhooks",
    items: [
      { to: "/brokers", label: "Broker Link", icon: Link2, hint: "g k", keywords: "zerodha angel fyers login" },
      { to: "/broker-monitor", label: "Broker Monitor", icon: HeartPulse, hint: "g n", keywords: "rate limit" },
      { to: "/tradingview", label: "TradingView", icon: Radio, hint: "g v", keywords: "webhook pine" },
    ],
  },
];

function matches(path: string, to: string): boolean {
  if (to === "/") return path === "/";
  return path === to || path.startsWith(`${to}/`);
}

/** Destination + sub-item for a pathname (the index route belongs to Today). */
export function locate(pathname: string): { destination?: Destination; item?: NavItem } {
  const path = pathname === "/" ? "/pulse" : pathname;
  for (const d of DESTINATIONS) {
    const item = d.items.find((i) => matches(path, i.to));
    if (item) return { destination: d, item };
    if (matches(path, d.to)) return { destination: d };
  }
  // Pages reachable only by drill-in.
  if (path.startsWith("/setup/")) return { destination: DESTINATIONS[0] };
  return {};
}
