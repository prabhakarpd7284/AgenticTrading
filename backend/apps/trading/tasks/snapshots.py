from celery import shared_task


@shared_task(ignore_result=True)
def refresh_all() -> int:
    from apps.trading.models import Portfolio, PortfolioSnapshot, Position

    n = 0
    for p in Portfolio.objects.all():
        open_positions = Position.objects.filter(portfolio=p, status="open").count()
        PortfolioSnapshot.objects.create(
            tenant_id=p.tenant_id,
            portfolio=p,
            equity=p.capital + p.day_pnl,
            day_pnl=p.day_pnl,
            unrealized_pnl=0,
            open_positions=open_positions,
        )
        n += 1
    return n
