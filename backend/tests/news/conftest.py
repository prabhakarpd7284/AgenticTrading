import pytest


@pytest.fixture(autouse=True)
def _no_real_embeddings(monkeypatch):
    """Never call Voyage from tests (the dev .env carries a real key)."""
    from apps.news.services import indexing

    monkeypatch.setattr(indexing, "index_items",
                        lambda items: {"candidates": len(items), "indexed": 0, "ok": True,
                                       "error": "disabled in tests"})
