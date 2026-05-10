"""Domain event dispatch — simple pub/sub. Used by services; NOT by ORM signals."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[dict], None]]] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable[[dict], None]) -> None:
        self._subs[event].append(handler)

    def publish(self, event: str, payload: dict) -> None:
        for handler in self._subs.get(event, []):
            try:
                handler(payload)
            except Exception:  # noqa: BLE001 — subscribers must not break publishers
                import structlog
                structlog.get_logger().exception("event.handler_error", event=event)


bus = EventBus()
