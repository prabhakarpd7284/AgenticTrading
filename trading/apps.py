"""AppConfig for the legacy `trading` package.

Re-installed in INSTALLED_APPS so Django's management-command discovery
picks up the operator CLIs under `trading/management/commands/` —
they're the primary surface area for the dev-ops console.

Two non-obvious bits:

* `label = "trading_legacy"` dodges the app-label collision with the
  new `apps.trading` app (Phase 4c). Django requires unique labels.

* `import_models()` is overridden to a no-op so the SQLite-era models
  in `trading/models.py` (TradeJournal, StraddlePosition, AuditLog, …)
  are NOT registered or migrated. They were retired in Phase 6 and only
  remain on disk because a few helper modules (`trading.options.*`,
  `trading.utils.*`) still live alongside them.
"""
from django.apps import AppConfig


class TradingLegacyConfig(AppConfig):
    name = "trading"
    label = "trading_legacy"
    verbose_name = "Trading (legacy CLIs)"

    def import_models(self) -> None:
        # Skip loading trading/models.py (see module docstring). We still
        # need self.models initialised to an empty dict, because Django's
        # apps registry iterates self.models.values() in several places.
        self.models = {}
