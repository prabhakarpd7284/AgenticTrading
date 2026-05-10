"""Multi-database router — pins legacy trading models to `legacy`.

This lets the v2 stack run on Postgres (via `DATABASE_URL`) while the
legacy `trading.*` models (TradeJournal, StraddlePosition, AuditLog,
PortfolioSnapshot, WatchlistEntry, SystemControl) continue to read
from the seeded sqlite file at the repo root. No port/db juggling.

Rules:
  * All reads for the `trading` app go to the `legacy` alias.
  * All writes for the `trading` app go to the `legacy` alias.
  * `trading` migrations are only applied against `legacy`.
  * Everything else (apps.*) stays on `default`.
  * Relations across dbs are disallowed (ORM never joins across).

If the `legacy` alias isn't configured (e.g. under `config.settings.test`
where we keep a single in-memory sqlite), the router is a no-op for
`trading` too — the single DB handles every app.
"""
from __future__ import annotations

from django.conf import settings

LEGACY_APP_LABEL = "trading"
LEGACY_DB_ALIAS = "legacy"


def _legacy_configured() -> bool:
    return LEGACY_DB_ALIAS in getattr(settings, "DATABASES", {})


class LegacyRouter:
    def db_for_read(self, model, **hints):
        if model._meta.app_label == LEGACY_APP_LABEL and _legacy_configured():
            return LEGACY_DB_ALIAS
        return None

    def db_for_write(self, model, **hints):
        if model._meta.app_label == LEGACY_APP_LABEL and _legacy_configured():
            return LEGACY_DB_ALIAS
        return None

    def allow_relation(self, obj1, obj2, **hints):
        labels = {obj1._meta.app_label, obj2._meta.app_label}
        # Allow relations within the legacy app, or within the v2 apps;
        # disallow cross-database relations.
        if labels == {LEGACY_APP_LABEL}:
            return True
        if LEGACY_APP_LABEL not in labels:
            return True
        return False

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if not _legacy_configured():
            return None  # single-DB setup — defer to default policy
        if app_label == LEGACY_APP_LABEL:
            return db == LEGACY_DB_ALIAS
        # Non-legacy apps must never touch the legacy alias.
        return db != LEGACY_DB_ALIAS
