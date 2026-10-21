"""Which ``Portfolio`` row plays which role (docs/REDESIGN_PORTFOLIO_SOURCE.md).

A tenant's book is the set of its ``Portfolio`` rows:

* the **trading** portfolio is the oldest row (``order_by("created_at")``);
* **mutual-fund** portfolios hold ``MFHolding`` rows (on this install the one
  named "Mutual Funds", capital 0) and are never part of the trading book.

Anything that reads ``PortfolioSnapshot`` must scope it to a portfolio: with
two portfolios per tenant, "the tenant's newest snapshot" is a coin flip
between the trading book and the MF book (equity 0). That bug made the Desk
show the full 500k as invested (found in UX redesign R0).
"""
from __future__ import annotations

from django.db.models import Min, OuterRef, Subquery


def trading_portfolio(tenant):
    """The tenant's trading portfolio (oldest row), or ``None``."""
    from apps.trading.models import Portfolio

    if tenant is None:
        return None
    return Portfolio.objects.filter(tenant=tenant).order_by("created_at").first()


def mf_portfolio_ids(tenant) -> set:
    """Portfolios that hold mutual funds: any with an ``MFHolding`` row, plus
    capital-0 books named "Mutual Funds" (created before the first CAS import)."""
    from apps.mutual_funds.models import MFHolding
    from apps.trading.models import Portfolio

    ids = set(MFHolding.objects.filter(tenant=tenant).values_list("portfolio_id", flat=True).distinct())
    for pid, name, capital in Portfolio.objects.filter(tenant=tenant).values_list("id", "name", "capital"):
        if (name or "").strip().lower().startswith("mutual fund") and not capital:
            ids.add(pid)
    return ids


def book_portfolios(tenant):
    """The tenant's trading-book portfolios: every portfolio except MF books.
    Never empty while the tenant has a trading portfolio (the oldest row is
    always part of the book, even if it somehow also holds MFs)."""
    from apps.trading.models import Portfolio

    oldest = trading_portfolio(tenant)
    if oldest is None:
        return []
    mf = mf_portfolio_ids(tenant) - {oldest.id}
    return list(Portfolio.objects.filter(tenant=tenant).exclude(id__in=mf).order_by("created_at"))


def latest_trading_snapshot(tenant=None):
    """Newest ``PortfolioSnapshot`` of the trading portfolio.

    ``tenant=None`` serves the legacy single-tenant code paths (``trading/``
    package): it takes the newest snapshot among every tenant's *trading*
    portfolio (the oldest row per tenant), never an MF book's.
    """
    from apps.trading.models import Portfolio, PortfolioSnapshot

    if tenant is not None:
        p = trading_portfolio(tenant)
        if p is None:
            return None
        return PortfolioSnapshot.objects.filter(portfolio=p).order_by("-captured_at").first()

    first_created = Portfolio.objects.filter(tenant=OuterRef("tenant")).values("tenant").annotate(
        m=Min("created_at")).values("m")[:1]
    trading_ids = Portfolio.objects.filter(created_at=Subquery(first_created)).values("id")
    return (PortfolioSnapshot.objects.filter(portfolio_id__in=trading_ids)
            .order_by("-captured_at").first())
