"""Pure normalization for StockEdge Market-Breadth payloads.

No Django, no Playwright, no I/O — just dict-in / dict-out so it's trivially
unit-testable and reusable from the management command, Celery tasks, or the
Playwright harness.

A captured breadth payload looks like::

    {
      "dataset": "market_breadth",
      "as_of_date": "2026-06-25",
      "exchange": "NSE",
      "unit": "percent",
      "source_url": "https://web.stockedge.com/market-breadth",
      "columns": ["rs_pos", "sma20", "sma50", "sma100", "sma200"],
      "rows": [
        {"index_name": "Nifty 50", "constituent_count": 50,
         "rs_pos": 54, "sma20": 64, "sma50": 50, "sma100": 46, "sma200": 50},
        ...
      ]
    }

``parse_breadth_payload`` validates + coerces this into a normalized dict with
clean Python types; ``summarize_breadth`` derives a one-line regime read.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

# The breadth percentage columns we flatten into StockEdgeBreadthRow.
BREADTH_COLUMNS = ("rs_pos", "sma20", "sma50", "sma100", "sma200")

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _to_iso_date(value: Any) -> str:
    """Normalize a captured ``as_of_date`` to ``'YYYY-MM-DD'``.

    Accepts already-ISO dates, ISO datetimes (``'2026-06-25T00:00:00Z'``), and the
    common Indian display formats ``DD-MM-YYYY`` / ``DD/MM/YYYY`` that show up in
    user-captured payloads. Raises ``ValueError`` on anything else so the caller
    surfaces a clean error instead of crashing downstream on ``strptime``.
    """
    if value is None or value == "":
        raise ValueError("missing 'as_of_date'")
    s = str(value).strip()
    if "T" in s:  # ISO datetime -> keep the date part
        s = s.split("T", 1)[0].strip()
    if _ISO_DATE.match(s):
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return s
        except ValueError as exc:
            raise ValueError(f"invalid 'as_of_date' {value!r}: {exc}")
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unrecognized 'as_of_date' format {value!r}; expected YYYY-MM-DD")


def _to_int(value: Any) -> int | None:
    """Coerce to int, tolerating strings/floats/'' /None. Returns None if blank."""
    if value is None or value == "":
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    """Coerce to float, tolerating strings/'%'-suffixed/None. Returns None if blank."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = value.strip().rstrip("%").strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_breadth_payload(payload: dict) -> dict:
    """Validate + normalize a captured Market-Breadth payload.

    Returns::

        {
          "dataset": "market_breadth",
          "as_of_date": "2026-06-25",
          "exchange": "NSE",
          "source_url": "...",
          "rows": [
            {"index_name": ..., "constituent_count": int|None,
             "rs_pos": float|None, "sma20": ..., "sma50": ...,
             "sma100": ..., "sma200": ...},
            ...
          ]
        }

    Raises ``ValueError`` on malformed input (not a dict, no rows, or a row
    missing ``index_name``).
    """
    if not isinstance(payload, dict):
        raise ValueError("StockEdge breadth payload must be a dict")

    rows_in = payload.get("rows")
    if not isinstance(rows_in, list) or not rows_in:
        raise ValueError("StockEdge breadth payload missing non-empty 'rows' list")

    dataset = str(payload.get("dataset") or "market_breadth")
    exchange = str(payload.get("exchange") or "NSE").upper()
    try:
        as_of_date = _to_iso_date(payload.get("as_of_date"))
    except ValueError as exc:
        raise ValueError(f"StockEdge breadth payload {exc}")
    source_url = str(payload.get("source_url") or "")

    rows: list[dict] = []
    for i, raw_row in enumerate(rows_in):
        if not isinstance(raw_row, dict):
            raise ValueError(f"breadth row #{i} is not an object: {raw_row!r}")
        index_name = raw_row.get("index_name") or raw_row.get("index") or raw_row.get("name")
        if not index_name:
            raise ValueError(f"breadth row #{i} missing 'index_name': {raw_row!r}")
        row = {
            "index_name": str(index_name).strip(),
            "constituent_count": _to_int(raw_row.get("constituent_count")),
        }
        for col in BREADTH_COLUMNS:
            row[col] = _to_float(raw_row.get(col))
        rows.append(row)

    return {
        "dataset": dataset,
        "as_of_date": as_of_date,
        "exchange": exchange,
        "source_url": source_url,
        "rows": rows,
    }


def _row_mean(row: dict) -> float | None:
    """Mean of the available breadth columns for one normalized row."""
    vals = [row.get(col) for col in BREADTH_COLUMNS]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _avg(rows: list[dict], col: str) -> float | None:
    vals = [r.get(col) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def summarize_breadth(rows: list[dict]) -> dict:
    """Derive a one-line market regime read from normalized breadth rows.

    ``rows`` are the normalized rows from :func:`parse_breadth_payload`.

    Returns::

        {
          "avg_sma20": float|None,
          "avg_sma50": float|None,
          "avg_sma200": float|None,
          "broad_regime": "risk-on" | "neutral" | "risk-off",
          "strongest_index": str|None,
          "weakest_index": str|None,
        }

    Regime thresholds key off the medium-term SMA50 breadth: ``>=60`` risk-on,
    ``>=45`` neutral, else risk-off.
    """
    rows = rows or []
    avg_sma20 = _avg(rows, "sma20")
    avg_sma50 = _avg(rows, "sma50")
    avg_sma200 = _avg(rows, "sma200")

    if avg_sma50 is None:
        broad_regime = "neutral"
    elif avg_sma50 >= 60:
        broad_regime = "risk-on"
    elif avg_sma50 >= 45:
        broad_regime = "neutral"
    else:
        broad_regime = "risk-off"

    ranked = [(r["index_name"], _row_mean(r)) for r in rows]
    ranked = [(name, score) for name, score in ranked if score is not None]
    strongest_index = max(ranked, key=lambda x: x[1])[0] if ranked else None
    weakest_index = min(ranked, key=lambda x: x[1])[0] if ranked else None

    return {
        "avg_sma20": avg_sma20,
        "avg_sma50": avg_sma50,
        "avg_sma200": avg_sma200,
        "broad_regime": broad_regime,
        "strongest_index": strongest_index,
        "weakest_index": weakest_index,
    }
