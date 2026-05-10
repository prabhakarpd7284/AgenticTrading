from django.apps import AppConfig


class BrokerConfig(AppConfig):
    name = "apps.broker"

    def ready(self) -> None:
        from django.conf import settings
        if settings.ALPHADESK.get("BROKER_REGISTRY_AUTOLOAD", True):
            from apps.agents_core.registry import broker_registry
            broker_registry.load_entry_points()
