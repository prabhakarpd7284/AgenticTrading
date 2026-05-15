"""Portfolio retriever — reads live portfolio state and materializes as retrievable docs."""
from __future__ import annotations

from apps.agents_core.domain.contracts import RetrievalQuery, RetrievedDoc


class PortfolioRetriever:
    name = "portfolio"

    def retrieve(self, q: RetrievalQuery, k: int = 5) -> list[RetrievedDoc]:
        from apps.trading.models import Portfolio, Position

        tenant_id = q.filters.get("tenant_id")
        if not tenant_id:
            return []
        portfolio = Portfolio.objects.filter(tenant_id=tenant_id).first()
        if portfolio is None:
            return []

        docs: list[RetrievedDoc] = [
            RetrievedDoc(
                source="portfolio.capital",
                score=1.0,
                text=(
                    f"capital={portfolio.capital}; used={portfolio.used_capital}; "
                    f"day_pnl={portfolio.day_pnl}; realized={portfolio.realized_pnl}"
                ),
                metadata={"portfolio_id": str(portfolio.id)},
            )
        ]
        for pos in Position.objects.filter(portfolio=portfolio, status="open")[:k]:
            docs.append(
                RetrievedDoc(
                    source="portfolio.position",
                    score=0.9,
                    text=(
                        f"{pos.symbol} {pos.side} qty={pos.qty} avg={pos.avg_price} "
                        f"ltp={pos.last_ltp} unrealized={pos.unrealized_pnl}"
                    ),
                    metadata={"position_id": str(pos.id)},
                )
            )
        return docs
