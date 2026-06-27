import { createBrowserRouter } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { LoginPage } from "@/features/auth/LoginPage";
import { SignupPage } from "@/features/auth/SignupPage";
import { MarketPulsePage } from "@/features/market-pulse/MarketPulsePage";
import { RotationPage } from "@/features/rotation/RotationPage";
import { ShortlistPage } from "@/features/shortlist/ShortlistPage";
import { SetupPage } from "@/features/setup/SetupPage";
import { DashboardPage } from "@/features/dashboard/DashboardPage";
import { PositionsPage } from "@/features/positions/PositionsPage";
import { MonthlyPage } from "@/features/monthly/MonthlyPage";
import { AgentConsolePage } from "@/features/agents/AgentConsolePage";
import { StrategyBuilderPage } from "@/features/strategies/StrategyBuilderPage";
import { BacktesterPage } from "@/features/backtester/BacktesterPage";
import { BrokerLinkPage } from "@/features/broker/BrokerLinkPage";
import { BrokerMonitorPage } from "@/features/broker-monitor/BrokerMonitorPage";
import { TradingViewPage } from "@/features/tradingview/TradingViewPage";
import { SwingScannerPage } from "@/features/swing-scanner/SwingScannerPage";
import { BasketPage } from "@/features/basket/BasketPage";
import PyramidPage from "@/features/pyramid/PyramidPage";
import ScalpingPage from "@/features/scalping/ScalpingPage";
import OptionsDashboardPage from "@/features/options/OptionsDashboardPage";
import { OpsPage } from "@/features/ops/OpsPage";
import { PipelinePage } from "@/features/pipeline/PipelinePage";
import { OnboardingPage } from "@/features/auth/OnboardingPage";
import { RequireAuth } from "./guards";
import { RouteError } from "./RouteError";

export const router = createBrowserRouter([
  { path: "/login", element: <LoginPage /> },
  { path: "/signup", element: <SignupPage /> },
  { path: "/onboarding", element: <RequireAuth><OnboardingPage /></RequireAuth> },
  {
    path: "/",
    element: <RequireAuth><AppShell /></RequireAuth>,
    // Any uncaught render error in a child route lands here instead of blanking
    // the whole SPA.
    errorElement: <RouteError />,
    children: [
      // "What's Happening Today" is the trader's first read — Stage 1+2
      // of The Cascade — so it replaces the generic dashboard as the
      // index route.  The old /dashboard remains for capital + positions.
      { index: true, element: <MarketPulsePage /> },
      { path: "pulse", element: <MarketPulsePage /> },
      // Cascade Stage 3 — drill-in from the pulse sector heatmap.
      { path: "rotation", element: <RotationPage /> },
      // Cascade Stage 4 — tradeable watchlist built on top of rotation.
      { path: "shortlist", element: <ShortlistPage /> },
      // Cascade Stage 5 — single-name setup preview + @RiskGuard breakdown.
      { path: "setup/:symbol", element: <SetupPage /> },
      { path: "dashboard", element: <DashboardPage /> },
      { path: "positions", element: <PositionsPage /> },
      // Post-trade feedback loop — monthly earning tracker.
      { path: "monthly", element: <MonthlyPage /> },
      { path: "agents", element: <AgentConsolePage /> },
      { path: "agents/:runId", element: <AgentConsolePage /> },
      { path: "strategies", element: <StrategyBuilderPage /> },
      { path: "swing-scanner", element: <SwingScannerPage /> },
      { path: "basket", element: <BasketPage /> },
      { path: "backtester", element: <BacktesterPage /> },
      { path: "pyramid", element: <PyramidPage /> },
      // Scalping Simulator — bin-pressure scalper with live tick replay.
      { path: "scalping", element: <ScalpingPage /> },
      // Options Desk — single-screen trader cockpit covering every options
      // strategy (straddle, vertical spreads, iron condor, pyramid).
      { path: "options", element: <OptionsDashboardPage /> },
      { path: "brokers", element: <BrokerLinkPage /> },
      // Broker telemetry — live SmartAPI rate-limit, call volume, queues.
      { path: "broker-monitor", element: <BrokerMonitorPage /> },
      // TradingView Manager — dedicated page for webhook links, named
      // watchlists, and grouped incoming signals.
      { path: "tradingview", element: <TradingViewPage /> },
      // Developer ops console — stream any manage.py command (owner-gated).
      { path: "ops", element: <OpsPage /> },
      // Daily-pipeline debugger — status + manual triggers (owner-gated).
      { path: "pipeline", element: <PipelinePage /> },
    ],
  },
]);
