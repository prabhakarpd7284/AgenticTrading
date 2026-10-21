"""A tenant can now hold a second, MF-only Portfolio (capital 0). Code that
picks "the" portfolio for a tenant must deterministically take the oldest
(the trading one), not whichever UUID sorts first."""
from __future__ import annotations

from decimal import Decimal

from apps.agents_core.domain.contracts import RetrievalQuery
from apps.rag.retrievers.portfolio import PortfolioRetriever
from tests.factories import PortfolioFactory


def test_portfolio_retriever_uses_oldest_portfolio(tenant):
    trading = PortfolioFactory(tenant=tenant, name="Default", capital=Decimal("500000"))
    for _ in range(5):  # several MF-style portfolios with random UUIDs
        PortfolioFactory(tenant=tenant, name="Mutual Funds", capital=Decimal("0"))
    docs = PortfolioRetriever().retrieve(RetrievalQuery(text="x", filters={"tenant_id": tenant.id}))
    assert docs[0].metadata["portfolio_id"] == str(trading.id)
    assert "capital=500000" in docs[0].text
