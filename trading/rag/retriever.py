"""
RAG context retriever — fetch relevant history for the planner.

Reads from the v2 Postgres tables:
  apps.trading.Trade            — trade history (was trading.TradeJournal)
  apps.rag.KnowledgeDoc         — strategy rules (was trading.StrategyDoc)
  apps.trading.PortfolioSnapshot — capital/PnL state (renamed fields)

Returns plain text formatted for direct injection into the planner prompt.
"""
from __future__ import annotations

from logzero import logger


def retrieve_context(
    symbol: str,
    last_n_trades: int = 20,
    include_strategies: bool = True,
) -> str:
    """Build symbol-specific RAG context for the planner prompt."""
    from apps.rag.models import KnowledgeDoc
    from apps.trading.models import Trade

    sections: list[str] = []

    # ── 1. Recent trades for this symbol ──
    recent_trades = list(
        Trade.objects
        .filter(symbol=symbol)
        .order_by("-created_at")[:last_n_trades]
    )

    if recent_trades:
        trade_lines: list[str] = []
        wins = 0
        losses = 0
        for t in recent_trades:
            pnl = float(t.realized_pnl) if t.realized_pnl is not None else None
            pnl_str = f"P&L: {pnl:+.0f} INR" if pnl is not None else "P&L: pending"
            trade_lines.append(
                f"  - {t.created_at.strftime('%Y-%m-%d')} | "
                f"{t.side} {t.quantity}x @ {float(t.entry_price):.2f} | "
                f"SL: {float(t.stop_loss):.2f} | Target: {float(t.target):.2f} | "
                f"{t.status} | {pnl_str}"
            )
            if pnl is not None:
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1

        total = wins + losses
        wr = (wins / total * 100) if total else 0
        sections.append(
            f"RECENT TRADES FOR {symbol} (last {len(trade_lines)}):\n"
            f"  Win Rate: {wr:.0f}% ({wins}W / {losses}L out of {total} closed)\n"
            + "\n".join(trade_lines)
        )
    else:
        sections.append(f"RECENT TRADES FOR {symbol}: No previous trades found.")

    # ── 2. Other recent trades for portfolio context ──
    other_recent = list(
        Trade.objects
        .exclude(symbol=symbol)
        .order_by("-created_at")[:10]
    )
    if other_recent:
        other_lines = []
        for t in other_recent:
            pnl = float(t.realized_pnl) if t.realized_pnl is not None else None
            pnl_str = f"{pnl:+.0f}" if pnl is not None else "open"
            other_lines.append(
                f"  - {t.symbol} {t.side} {t.quantity}x @ {float(t.entry_price):.2f} "
                f"[{t.status}] {pnl_str}"
            )
        sections.append("OTHER RECENT TRADES:\n" + "\n".join(other_lines))

    # ── 3. Active strategy rules from the knowledge base ──
    if include_strategies:
        strategies = list(KnowledgeDoc.objects.filter(is_active=True))
        if strategies:
            strat_lines = [
                f"  [{s.category}] {s.title}:\n    {s.content}"
                for s in strategies
            ]
            sections.append("ACTIVE STRATEGY RULES:\n" + "\n\n".join(strat_lines))
        else:
            sections.append(
                "ACTIVE STRATEGY RULES: None configured.\n"
                "  Tip: seed via apps.rag.KnowledgeDoc.objects.create(...)"
            )

    context = "\n\n---\n\n".join(sections)
    logger.info(
        f"RAG context built for {symbol}: {len(recent_trades)} trade(s), "
        f"{len(other_recent)} other, {'strategies on' if include_strategies else 'strategies off'}"
    )
    return context


def retrieve_portfolio_context() -> str:
    """Portfolio-level context — capital, today's PnL, open positions."""
    from datetime import date

    from apps.trading.models import PortfolioSnapshot, Trade

    sections: list[str] = []

    # Latest snapshot. The v2 PortfolioSnapshot has a leaner shape than
    # the legacy version — derive the missing fields where possible.
    snap = PortfolioSnapshot.objects.order_by("-captured_at").first()
    if snap is not None:
        equity = float(snap.equity)
        day_pnl = float(snap.day_pnl)
        unrealized = float(snap.unrealized_pnl)
        daily_loss = max(0.0, -day_pnl)
        sections.append(
            f"PORTFOLIO STATE ({snap.captured_at.date()}):\n"
            f"  Equity (capital):  {equity:,.0f} INR\n"
            f"  Today's P&L:       {day_pnl:+,.0f} INR\n"
            f"  Unrealized P&L:    {unrealized:+,.0f} INR\n"
            f"  Today's Losses:    {daily_loss:,.0f} INR\n"
            f"  Open Positions:    {snap.open_positions}"
        )
    else:
        sections.append("PORTFOLIO STATE: No snapshot available yet.")

    todays = list(
        Trade.objects.filter(trade_date=date.today()).order_by("-created_at")
    )
    if todays:
        lines = []
        for t in todays:
            pnl = float(t.realized_pnl) if t.realized_pnl is not None else None
            pnl_str = f"{pnl:+.0f}" if pnl is not None else "open"
            lines.append(
                f"  - {t.symbol} {t.side} {t.quantity}x [{t.status}] {pnl_str}"
            )
        sections.append(f"TODAY'S TRADES ({len(todays)}):\n" + "\n".join(lines))

    return "\n\n".join(sections) if sections else "No portfolio data available."
