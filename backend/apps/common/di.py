"""Tiny dependency-injection container. Zero dependencies, fully typed.

Usage:
    container.bind(RiskGuard, lambda: DeterministicRiskGuard(limits=settings.ALPHADESK))
    container.resolve(RiskGuard)
"""
from __future__ import annotations

from threading import RLock
from typing import Callable, TypeVar

T = TypeVar("T")


class Container:
    def __init__(self) -> None:
        self._factories: dict[type, Callable[[], object]] = {}
        self._singletons: dict[type, object] = {}
        self._lock = RLock()

    def bind(self, iface: type[T], factory: Callable[[], T], *, singleton: bool = True) -> None:
        with self._lock:
            self._factories[iface] = factory
            self._singletons.pop(iface, None)
            setattr(factory, "_singleton", singleton)

    def resolve(self, iface: type[T]) -> T:
        with self._lock:
            if iface in self._singletons:
                return self._singletons[iface]  # type: ignore[return-value]
            factory = self._factories.get(iface)
            if factory is None:
                raise LookupError(f"No binding for {iface!r}")
            instance = factory()
            if getattr(factory, "_singleton", True):
                self._singletons[iface] = instance
            return instance  # type: ignore[return-value]

    def reset(self) -> None:
        with self._lock:
            self._factories.clear()
            self._singletons.clear()


container = Container()
