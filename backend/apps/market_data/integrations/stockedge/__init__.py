"""StockEdge advisory-overlay ingestion.

StockEdge analytics are pulled in as an independent research / confirmation
overlay (NOT an execution feed, NOT tenant-scoped). See
``docs/integrations/STOCKEDGE_INTEGRATION.md`` for the full design.

Public surface:
  - ``parse_breadth_payload(payload)`` / ``summarize_breadth(rows)`` — pure
    normalization, no Django/Playwright needed.
  - ``capture_breadth(...)`` — production Playwright capture (lazy import).
"""
from __future__ import annotations

from .parser import parse_breadth_payload, summarize_breadth

__all__ = ["parse_breadth_payload", "summarize_breadth", "capture_breadth"]


def capture_breadth(*args, **kwargs):
    """Lazy proxy to :func:`harness.capture_breadth`.

    Kept thin so importing this package never imports Playwright.
    """
    from .harness import capture_breadth as _capture
    return _capture(*args, **kwargs)
