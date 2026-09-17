from django.apps import AppConfig


class MarketDataConfig(AppConfig):
    name = "apps.market_data"

    def ready(self) -> None:
        # Absorbed from the old BrokerConfig in Phase 4b — broker adapter
        # registry autoload now belongs to the markets domain.
        from django.conf import settings
        if settings.ALPHADESK.get("BROKER_REGISTRY_AUTOLOAD", True):
            from apps.agents_core.registry import broker_registry
            broker_registry.load_entry_points()
