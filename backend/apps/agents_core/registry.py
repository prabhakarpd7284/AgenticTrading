"""Entry-point-driven registries for Strategies, Retrievers, BrokerAdapters.

Plugin authors declare their class in `pyproject.toml` under `alphadesk.strategies`
/ `alphadesk.retrievers` / `alphadesk.brokers`. At Django boot, each registry walks
`importlib.metadata.entry_points(group=...)` and imports them.

Tests can register in-memory plugins without touching the filesystem.
"""
from __future__ import annotations

import importlib.metadata as md
from threading import RLock
from typing import Generic, TypeVar

import structlog

log = structlog.get_logger()

T = TypeVar("T")


class PluginRegistry(Generic[T]):
    def __init__(self, group: str) -> None:
        self._group = group
        self._items: dict[str, T] = {}
        self._lock = RLock()

    def register(self, name: str, obj: T) -> None:
        with self._lock:
            self._items[name] = obj
            log.info("plugin.registered", group=self._group, name=name)

    def get(self, name: str) -> T:
        with self._lock:
            if name not in self._items:
                raise LookupError(f"{self._group} plugin not found: {name}")
            return self._items[name]

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._items)

    def items(self) -> list[tuple[str, T]]:
        with self._lock:
            return list(self._items.items())

    def load_entry_points(self) -> None:
        try:
            eps = md.entry_points(group=self._group)
        except TypeError:  # py<3.10 compat
            eps = md.entry_points().get(self._group, [])  # type: ignore[assignment]
        for ep in eps:
            try:
                obj = ep.load()
                instance = obj() if callable(obj) else obj
                self.register(ep.name, instance)
            except Exception:  # noqa: BLE001
                log.exception("plugin.load_failed", group=self._group, name=ep.name)


strategy_registry: PluginRegistry = PluginRegistry("alphadesk.strategies")
retriever_registry: PluginRegistry = PluginRegistry("alphadesk.retrievers")
broker_registry: PluginRegistry = PluginRegistry("alphadesk.brokers")
