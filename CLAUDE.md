# AgenticTrading — Claude Code Context

## What This Is

An AI-driven Indian stock market trading system built on Django + LangGraph + Angel One SmartAPI.
Supports two parallel trading workflows:

- **Directional Trading** — BUY/SELL equity intraday positions (NSE cash market)
- **Options Management** — Short straddle lifecycle management (NFO options, NIFTY/BANKNIFTY)

Capital: 500,000 INR | Mode: paper (default) | Broker: Angel One SmartAPI

---

## AI Virtual Team

Each "agent" is a named role with a specific responsibility. No agent crosses into another's domain.

| Agent | Role | Files | LLM? |
|-------|------|-------|------|
| **@DataAnalyst** | Fetches and enriches market data from Angel One | `trading/services/data_service.py`, `trading/options/data_service.py` | No |
| **@DirectionalTrader** | Plans BUY/SELL equity trades from intraday structure | `trading/agents/planner.py`, `trading/graph/trading_graph.py` | Yes (Claude) |
| **@OptionsStrategist** | Manages straddle positions — HOLD/CLOSE/HEDGE/ROLL | `trading/options/straddle/graph.py`, `trading/options/straddle/prompts.py` | Yes (Claude) |
| **@RiskGuard** | Deterministic risk validation — no LLM, no exceptions | `trading/services/risk_engine.py`, `trading/options/straddle/graph.py:validate_action_node` | Never |
| **@PortfolioTracker** | Tracks capital, P&L, daily loss, open positions | `trading/rag/retriever.py`, `trading/models.py:PortfolioSnapshot` | No |

### Agent Interaction Rules
- **@RiskGuard** is the last gate before every execution. It cannot be bypassed.
- **@DataAnalyst** runs first in every workflow. No trading decision without live data.
- **@DirectionalTrader** and **@OptionsStrategist** never share state — separate graphs.
- **@PortfolioTracker** is read-only during planning; updated only after execution.

---

## How to Run

### Setup
```bash
python manage.py migrate
python manage.py run_trading_agent --seed-strategies
python manage.py run_trading_agent --init-portfolio 500000
```

### Directional Trading (Equity)
```bash
# Plan a trade
python manage.py run_trading_agent "Plan a BUY trade for HDFCBANK"
python manage.py run_trading_agent "Analyze NIFTY50 and plan a trade"

# View journal
python manage.py run_trading_agent --show-journal
```

### Straddle Management (Options)
```bash
# Step 1: Register a position (after selling the straddle)
python manage.py manage_straddle --register \
    --underlying NIFTY --strike 24200 --expiry 2026-03-10 \
    --ce-symbol NIFTY10MAR2624200CE --ce-token 45482 --ce-sell 394.85 \
    --pe-symbol NIFTY10MAR2624200PE --pe-token 45483 --pe-sell 138.35 \
    --lots 1

# Step 2: Run full analysis + LLM recommendation (call every 15-30 min)
python manage.py manage_straddle --analyze --position 1

# Check status without LLM (instant)
python manage.py manage_straddle --status --position 1

# Force-execute an action (bypass LLM)
python manage.py manage_straddle --execute CLOSE_BOTH -r-position 1

# List all positions
python manage.py manage_straddle --list
```

---

## Architecture

```
Directional Workflow:
  fetch_data → retrieve_context → planner(@DirectionalTrader) → risk(@RiskGuard) → execute → journal

Straddle Workflow:
  fetch_market_data(@DataAnalyst)
    → analyze_position (pure Python, @DataAnalyst)
      → generate_action(@OptionsStrategist)
        → validate_action(@RiskGuard)
          → execute_action
            → journal_action
```

Both workflows:
- Use Angel One SmartAPI (paper or live via `TRADING_MODE`)
- Log every decision to `AuditLog` (non-blocking)
- Save outcomes to `TradeJournal` / `StraddlePosition`
- Are validated deterministically before execution

---

## Key Invariants (Never Break These)

1. **@RiskGuard is always the last gate** before execution. LLM cannot bypass it.
2. **`TRADING_MODE=paper` by default** — set to `live` explicitly and deliberately.
3. **Never trust LLM for position sizing** — always computed deterministically (% of capital).
4. **Journal every decision** — rejected trades are also recorded.
5. **Expiry-day positions** — `@OptionsStrategist` must recommend closing before 3:15 PM IST.
6. **Audit log is non-blocking** — logging failures never stop trading flow.
7. **Short straddle hard stop** — if combined premium > what was sold, close immediately.

---

## File Map

```
trading/
├── agents/planner.py              @DirectionalTrader — dual-mode Claude (CLI/API)
├── graph/
│   ├── state.py                   TradingState, TradePlan, RiskResult, ExecutionResult
│   └── trading_graph.py           Equity LangGraph workflow (6 nodes)
├── services/
│   ├── data_service.py            @DataAnalyst — equity OHLCV from Angel One
│   ├── risk_engine.py             @RiskGuard — 9-criteria deterministic validator
│   ├── broker_service.py          Order execution (paper + live)
│   └── backtester.py              Historical candle replay
├── options/
│   ├── data_service.py            @DataAnalyst — NFO options LTP + VIX + NIFTY candles
│   └── straddle/
│       ├── state.py               StraddleState, StraddleAction, StraddleAnalysis
│       ├── prompts.py             @OptionsStrategist system prompt + user prompt builder
│       ├── analyzer.py            P&L, delta, market phase, scenarios (pure Python)
│       └── graph.py               Straddle LangGraph workflow (6 nodes)
├── rag/retriever.py               @PortfolioTracker — RAG from DB
├── models.py                      TradeJournal, StraddlePosition, PortfolioSnapshot, AuditLog
└── management/commands/
    ├── run_trading_agent.py       CLI: equity trading
    ├── manage_straddle.py         CLI: straddle lifecycle
    └── run_backtest.py            CLI: backtesting
```

---

## Environment Variables

```bash
# Broker
SMARTAPI_KEY=...
SMARTAPI_USERNAME=...
SMARTAPI_PASSWORD=...
SMARTAPI_TOTP_SECRET=...

# Trading mode
TRADING_MODE=paper        # paper | live
PLANNER_MODE=cli          # cli (Max plan, free) | api (Anthropic credits)
LLM_MODEL=claude-sonnet-4-6

# Risk limits
DEFAULT_CAPITAL=500000
MAX_RISK_PER_TRADE_PCT=1.0
MAX_DAILY_LOSS_PCT=3.0
MAX_POSITION_SIZE_PCT=10.0
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Django 5.1 |
| Agent orchestration | LangGraph 0.6+ |
| LLM | Claude (Anthropic) — CLI or API |
| Broker | Angel One SmartAPI (`SmartApi` package) |
| Database | SQLite (dev) / PostgreSQL (prod) |
| Data processing | Pandas, Pydantic |
| Logging | Logzero |
