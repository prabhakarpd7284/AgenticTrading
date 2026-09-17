from django.apps import AppConfig


class RagConfig(AppConfig):
    name = "apps.rag"

    def ready(self) -> None:
        from django.conf import settings
        if settings.ALPHADESK.get("RAG_REGISTRY_AUTOLOAD", True):
            from apps.agents_core.registry import retriever_registry
            retriever_registry.load_entry_points()
