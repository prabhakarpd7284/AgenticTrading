# Options Module — Claude Code Context

## Purpose

Manages the lifecycle of short straddle positions on NIFTY/BANKNIFTY NFO options.
Runs as a parallel LangGraph workflow alongside the equity directional trading graph.

## Module Owner: @OptionsStrategist

The options module is owned by `@OptionsStrategist`. It:
- Fetches live NFO option prices via Angel One SmartAPI
- Computes P&L, delta, VIX phase, market phase (no LLM — pure Python)
- Runs the straddle management prompt through Claude
- Validates the recommendation deterministically (@RiskGuard)
- Executes buy-to-close orders or adds futures hedges

## File Structure

```
trading/options/
├── __init__.py
├── CLAUDE.md                  ← this file
├── data_service.py            ← NFO data fetching (OptionsDataService)
└── straddle/
    ├── __init__.py
    ├── state.py               ← StraddleState + StraddleAction + StraddleAnalysis
    ├── prompts.py             ← THE straddle management prompt (system + user builder)
    ├── analyzer.py            ← Pure-Python analytics: P&L, delta, phases, scenarios
    └── graph.py               ← LangGraph 6-node straddle workflow
```

## The Straddle Workflow

```
fetch_market_data    → NIFTY spot + VIX + CE/PE LTPs + 5-min candles (Angel One)
analyze_position     → P&L, delta, market phase, expiry scenarios (pure Python)
generate_action      → LLM recommendation: HOLD/CLOSE_BOTH/CLOSE_CE/CLOSE_PE/HEDGE/ROLL
validate_action      → Hard rules: underwater→CLOSE, expiry+ROLL→CLOSE, low-conf→MONITOR
execute_action       → BUY-to-close CE/PE or hedge with NIFTY futures
journal_action       → Update StraddlePosition in DB, append to management_log
```

## Management Action Reference

| Action | When to use |
|--------|------------|
| `HOLD` | Premium decaying, NIFTY range-bound, VIX stable, >1 DTE |
| `CLOSE_BOTH` | Expiry day, underwater, VIX spike, large directional move |
| `CLOSE_CE` | NIFTY approaching/above strike, CE ITM and growing |
| `CLOSE_PE` | NIFTY crashing through strike, PE deep ITM |
| `HEDGE_FUTURES` | Delta > ±0.5, position profitable, NIFTY trending |
| `ROLL` | >3 DTE, tested leg near strike, can collect more premium |

## Key Thresholds (hardcoded in validator)

- **Confidence < 0.6** → override to MONITOR (no action)
- **Combined premium > sold** → override to CLOSE_BOTH IMMEDIATE
- **Expiry tomorrow + ROLL** → override to CLOSE_BOTH IMMEDIATE
- **HEDGE_FUTURES + hedge_lots = 0** → rejected

## Data Sources

| Data | Source | Token |
|------|--------|-------|
| NIFTY 50 spot | NSE via SmartAPI `ltpData` | `99926000` |
| India VIX | NSE via SmartAPI `ltpData` | `99926017` |
| NIFTY CE/PE options | NFO via SmartAPI `ltpData` | From NFO scrip master |
| NIFTY 5-min candles | NSE via SmartAPI `getCandleData` | `99926000` |

## Adding a New Options Strategy

1. Create a new folder under `trading/options/` (e.g., `strangle/`, `ironCondor/`)
2. Follow the same file pattern: `state.py`, `prompts.py`, `analyzer.py`, `graph.py`
3. Add a new management command under `trading/management/commands/`
4. Add a new Django model to `trading/models.py` for position tracking
5. Register in root `CLAUDE.md` virtual team table

## Common Issues

**"NFO token not found"**
→ The NFO scrip master is downloaded fresh on each session. Check if the expiry format matches Angel One's format (e.g., `10MAR26` not `2026-03-10`).

**"Position underwater" override**
→ The validator overrides HOLD → CLOSE_BOTH when combined premium > sold. This is intentional. Do not change this rule.

**"Expiry tomorrow + ROLL rejected"**
→ Rolling on expiry day is dangerous (gamma). The validator blocks it by design.
