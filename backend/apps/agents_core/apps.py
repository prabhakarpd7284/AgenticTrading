from django.apps import AppConfig


class AgentsCoreConfig(AppConfig):
    name = "apps.agents_core"

    def ready(self) -> None:
        from django.conf import settings
        if settings.ALPHADESK.get("STRATEGY_REGISTRY_AUTOLOAD", True):
            from apps.agents_core.registry import strategy_registry
            strategy_registry.load_entry_points()
