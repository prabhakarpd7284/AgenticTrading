# AlphaDesk — Product Vision

> One-stop, AI-assisted trading & portfolio platform for the Indian markets.
> An always-on virtual trading team that watches the market, plans trades,
> defends capital, and explains every decision.

---

## 1. Why this exists

Indian retail and semi-pro traders are drowning. They juggle 6+ tools — broker terminal,
charting platform, options chain, news feed, journal spreadsheet, Telegram tip-channels —
and still take impulsive trades, mismanage option positions, and never journal honestly.

Existing platforms (Sensibull, Streak, TradingView, Tijori) each solve **one** slice. None
of them act on the trader's behalf with explicit, deterministic risk controls and a transparent
agentic reasoning trail.

**AlphaDesk's promise:** an AI virtual desk — Data Analyst, Directional Trader, Options
Strategist, Risk Guard, Portfolio Tracker — that runs 24×7, surfaces high-conviction setups,
manages open positions, enforces hard risk limits, and produces an investor-grade journal.

## 2. Target customers

| Segment | Persona | Pain | Willingness to pay |
|---------|---------|------|---------------------|
| **Retail SaaS** | Active intraday/F&O trader, ₹2–25 L capital | Blows up on revenge trades, no system, no journal | ₹999–4,999 / month |
| **Wealth Advisor (B2B)** | Independent RIA / sub-broker managing 20–200 client books | Cannot scale personalised attention; manual reporting | ₹15,000–60,000 / month / seat |
| **Prop / Fund desk** | 5–20 trader prop firm or AIF Cat-III | Needs auditable AI co-pilot, capital allocation rails, compliance log | ₹2–15 L / year per firm |

All three share the same core engine; they differ in tenancy, RBAC, branding, billing.

## 3. The product in one paragraph

A web app where the user (or RM, or trader) lands on a live dashboard showing portfolio,
open positions with live P&L, and an "AI desk" panel. They can chat with named agents
("@DirectionalTrader, scan NIFTY50 for breakout setups"), see the agent's reasoning
streamed token-by-token with the data and RAG context it pulled, accept/reject the
proposed trade, and watch RiskGuard validate before any order goes out. Strategies are
built in a no-code form, backtested against historical candles, and promoted to live
(paper or real) with one click. Every decision — accepted, rejected, auto-closed — lands
in an immutable journal that doubles as compliance evidence.

## 4. Core jobs to be done

1. **Watch the market for me** — universe scan, regime detection, alert when a setup fires.
2. **Plan a trade with explicit reasoning** — entry, SL, target, sizing, R:R, with citations.
3. **Manage my open positions** — straddle adjustments, trailing SL, expiry-day exits.
4. **Stop me from blowing up** — deterministic risk gates that the LLM cannot override.
5. **Show me what happened and why** — searchable journal + analytics.
6. **Let me build & test ideas** — strategy DSL + backtester + walk-forward.
7. **Onboard cleanly** — broker linking, paper-trade by default, KYC-light.

## 5. Differentiation

- **Named agents with bounded responsibilities** (not a single monolithic chat).
- **Deterministic RiskGuard** — last gate, never an LLM. Sellable to compliance.
- **Pluggable agentic RAG** — strategies/retrievers are plugins, not core code.
  Lets us (and partners) ship new strategies as `pip install alphadesk-strategy-x`.
- **Paper-first** — the product is useful and demoable without ever connecting real money.
- **Indian-market-native** — NSE/BSE symbology, NFO options chain, F&O lot sizes,
  Indian taxation (STT/STCG), holiday calendar, broker integrations (Angel, Zerodha, Fyers).

## 6. Non-goals (v1)

- US/crypto markets.
- Mobile native apps (responsive web only).
- Auto-trading without human approval (always human-in-the-loop unless the firm explicitly enables auto-execute with hard caps).
- Social/copy-trading features.
- Tax filing.

## 7. Success metrics (north stars)

| Metric | Target (12 mo) |
|--------|----------------|
| WAU / MAU | > 0.55 |
| Median trades journalled per active user / week | > 15 |
| % paper-users converting to paid in 30 days | > 8% |
| Net retention (B2B) | > 110% |
| RiskGuard "block" rate | 5–15% (proves it's doing something) |
| Agent recommendation acceptance rate | 30–55% |

## 8. Pricing snapshot

| Plan | Price | Limits |
|------|-------|--------|
| Paper (free) | ₹0 | 1 portfolio, 50 agent runs/month, basic journal |
| Trader | ₹1,499 / mo | 1 broker link, unlimited journal, all strategies |
| Pro | ₹3,999 / mo | 3 broker links, custom strategies, API access |
| Advisor (B2B) | ₹25k / mo + ₹500/client | RBAC, white-label, client reports |
| Desk (B2B) | Annual contract | SSO, audit logs, on-prem option, SLA |
