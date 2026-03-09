"""
RAG Retriever — Postgres-backed, no vector DB.

Simple and effective:
  1. Pull last N trades for the symbol from TradeJournal
  2. Pull active strategy rules from StrategyDoc
  3. Format as context string for the planner LLM

Later upgrade path: add pgvector extension to Postgres
and switch to embedding-based retrieval.
"""
import os
import sys
import django

# Bootstrap Django ORM if not already configured
if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ["DJANGO_SETTINGS_MODULE"] = "config.settings"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    django.setup()

from logzero import logger
from typing import Optional


def retrieve_context(
    symbol: str,
    last_n_trades: int = 20,
    include_strategies: bool = True,
) -> str:
    """
    Build RAG context from Postgres for a given symbol.

    Returns a formatted string ready to inject into the planner prompt.
    """
    from trading.models import TradeJournal, StrategyDoc

    sections = []

    # ──────────────────────────────────────────
    # 1. Recent trades for this symbol
    # ──────────────────────────────────────────
    recent_trades = (
        TradeJournal.objects
        .filter(symbol=symbol)
        .order_by("-created_at")[:last_n_trades]
    )

    if recent_trades.exists():
        trade_lines = []
        wins = 0
        losses = 0
        for t in recent_trades:
            pnl_str = f"P&L: {t.pnl:+.0f} INR" if t.pnl is not None else "P&L: pending"
            trade_lines.append(
                f"  - {t.created_at.strftime('%Y-%m-%d')} | {t.side} {t.quantity}x @ {t.entry_price:.2f} | "
                f"SL: {t.stop_loss:.2f} | Target: {t.target:.2f} | {t.status} | {pnl_str}"
            )
            if t.pnl is not None:
                if t.pnl > 0:
                    wins += 1
                elif t.pnl < 0:
                    losses += 1

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        sections.append(
            f"RECENT TRADES FOR {symbol} (last {len(trade_lines)}):\n"
            f"  Win Rate: {win_rate:.0f}% ({wins}W / {losses}L out of {total_trades} closed)\n"
            + "\n".join(trade_lines)
        )
    else:
        sections.append(f"RECENT TRADES FOR {symbol}: No previous trades found.")

    # ──────────────────────────────────────────
    # 2. All-symbol recent trades (portfolio context)
    # ──────────────────────────────────────────
    all_recent = (
        TradeJournal.objects
        .exclude(symbol=symbol)
        .order_by("-created_at")[:10]
    )

    if all_recent.exists():
        other_lines = []
        for t in all_recent:
            pnl_str = f"{t.pnl:+.0f}" if t.pnl is not None else "open"
            other_lines.append(
                f"  - {t.symbol} {t.side} {t.quantity}x @ {t.entry_price:.2f} [{t.status}] {pnl_str}"
            )
        sections.append("OTHER RECENT TRADES:\n" + "\n".join(other_lines))

    # ──────────────────────────────────────────
    # 3. Strategy documents
    # ──────────────────────────────────────────
    if include_strategies:
        strategies = StrategyDoc.objects.filter(is_active=True)

        if strategies.exists():
            strat_lines = []
            for s in strategies:
                strat_lines.append(f"  [{s.category}] {s.title}:\n    {s.content}")

            sections.append("ACTIVE STRATEGY RULES:\n" + "\n\n".join(strat_lines))
        else:
            sections.append(
                "ACTIVE STRATEGY RULES: None configured.\n"
                "  Tip: Add rules via Django admin or StrategyDoc.objects.create()"
            )

    context = "\n\n---\n\n".join(sections)

    logger.info(
        f"RAG context built for {symbol}: "
        f"{len(recent_trades)} trades, "
        f"{StrategyDoc.objects.filter(is_active=True).count()} strategy docs, "
        f"{len(context)} chars"
    )

    return context


def retrieve_portfolio_context() -> str:
    """
    Build portfolio-level context (not symbol-specific).
    Used when orchestrator needs to decide WHAT to trade.
    """
    from trading.models import TradeJournal, PortfolioSnapshot

    sections = []

    # Latest portfolio snapshot
    try:
        snap = PortfolioSnapshot.objects.latest()
        sections.append(
            f"PORTFOLIO STATE ({snap.snapshot_date}):\n"
            f"  Capital: {snap.capital:,.0f} INR\n"
            f"  Invested: {snap.invested:,.0f} INR\n"
            f"  Available: {snap.available_cash:,.0f} INR\n"
            f"  Today's P&L: {snap.daily_pnl:+,.0f} INR\n"
            f"  Total P&L: {snap.total_pnl:+,.0f} INR\n"
            f"  Today's Losses: {snap.daily_loss:,.0f} INR\n"
            f"  Open Positions: {snap.open_positions}"
        )
    except Exception:
        sections.append("PORTFOLIO STATE: No snapshot available.")

    # Today's trades
    from datetime import date
    todays_trades = TradeJournal.objects.filter(trade_date=date.today()).order_by("-created_at")
    if todays_trades.exists():
        lines = []
        for t in todays_trades:
            pnl_str = f"{t.pnl:+.0f}" if t.pnl is not None else "open"
            lines.append(f"  - {t.symbol} {t.side} {t.quantity}x [{t.status}] {pnl_str}")
        sections.append(f"TODAY'S TRADES ({len(todays_trades)}):\n" + "\n".join(lines))

    return "\n\n".join(sections) if sections else "No portfolio data available."
