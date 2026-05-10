from django.apps import AppConfig


class LegacyConfig(AppConfig):
    """Thin DRF bridge over the existing `trading` Django app + dashboard_utils.

    The v2 REST surface still evolves in `apps.*`; this app exists so the
    React UI can render the 700+ rows of real data that already live in
    the legacy sqlite file (TradeJournal, StraddlePosition, AuditLog,
    PortfolioSnapshot) without waiting for the v2 migration to complete.
    """

    name = "apps.legacy"
    label = "legacy"
