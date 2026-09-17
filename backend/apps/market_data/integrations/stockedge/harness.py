"""Production capture harness for StockEdge Market Breadth (Playwright).

Why a browser harness and not a plain HTTP client: StockEdge's gateway
fingerprints out-of-band requests and returns 504 even with a valid Bearer
token, while the app's own identical requests return 200 (see
``docs/integrations/STOCKEDGE_INTEGRATION.md`` §2.3). The robust, ToS-defensible
mechanism is to ride a genuine logged-in session and capture what the app
itself fetches — both the JSON the page requests (``page.on('response')``) and,
as a backstop, the rendered grid in the DOM.

Playwright is an OPTIONAL dependency: it is imported lazily *inside* the
functions, so this module imports fine on a host without Playwright installed
(the management command, parser, and tests don't need it). Only ``--live``
capture pulls it in.

Usage (production)::

    from apps.market_data.integrations.stockedge.harness import capture_breadth
    payload = capture_breadth(storage_state_path="/secrets/stockedge_state.json")

Where ``storage_state.json`` is a Playwright storage-state file holding a
logged-in StockEdge session (cookies + localStorage), produced once via the
login flow (Google OIDC) and refreshed when the refresh token lapses.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

import structlog

log = structlog.get_logger(__name__)

BREADTH_URL = "https://web.stockedge.com/market-breadth"
SOURCE_URL = BREADTH_URL
# api.stockedge.com responses whose path hints at breadth data — captured as a
# fallback to the DOM scrape.
_API_HINTS = ("breadth", "marketbreadth", "MarketBreadth")

_PLAYWRIGHT_HINT = (
    "Playwright is required for live StockEdge capture. Install it with:\n"
    "    pip install playwright && playwright install chromium\n"
    "and provide a logged-in session via --storage-state /path/to/storage_state.json\n"
    "(generate it once by logging into https://web.stockedge.com with a persistent "
    "Playwright context). For the MVP without a live session, ingest the bundled "
    "sample with: manage.py pull_stockedge_breadth --from-json <sample_breadth.json>."
)


def _require_playwright():
    """Import sync Playwright lazily; raise a helpful RuntimeError if absent."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without playwright
        raise RuntimeError(_PLAYWRIGHT_HINT) from exc
    return sync_playwright


def capture_breadth(
    storage_state_path: str | None = None,
    headless: bool = True,
    *,
    timeout_ms: int = 60_000,
    exchange: str = "NSE",
) -> dict:
    """Capture today's Market Breadth grid from a logged-in StockEdge session.

    Steps:
      1. Launch headless Chromium, restoring ``storage_state_path`` (the
         persisted logged-in session) if provided.
      2. Register a ``page.on('response')`` hook that snapshots any
         ``api.stockedge.com`` breadth JSON body (fallback source).
      3. ``page.goto(BREADTH_URL)`` and wait for the breadth grid to render.
      4. Extract the grid from the DOM into the payload shape that
         :func:`parser.parse_breadth_payload` expects.
      5. If the DOM scrape yields no rows, fall back to the captured JSON.

    Returns a payload dict ready for ``parse_breadth_payload``. Raises
    ``RuntimeError`` if Playwright is unavailable, or if neither the DOM nor the
    API capture produced any rows.
    """
    sync_playwright = _require_playwright()

    captured_json: list[dict[str, Any]] = []

    def _on_response(response) -> None:  # pragma: no cover - needs a live browser
        url = response.url
        if "api.stockedge.com" not in url:
            return
        if not any(hint in url for hint in _API_HINTS):
            return
        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - non-JSON / streamed bodies
            return
        captured_json.append({"url": url, "body": body})

    rows: list[dict] = []
    with sync_playwright() as p:  # pragma: no cover - needs a live browser
        browser = p.chromium.launch(headless=headless)
        context_kwargs: dict[str, Any] = {}
        if storage_state_path:
            context_kwargs["storage_state"] = storage_state_path
        context = browser.new_context(**context_kwargs)
        page = context.new_page()
        page.on("response", _on_response)
        try:
            page.goto(BREADTH_URL, wait_until="networkidle", timeout=timeout_ms)
            # The breadth grid is an Ionic <app-market-breadth-heatmap-chart>
            # heatmap (NOT an HTML <table>) — wait for its value cells to render.
            try:
                page.wait_for_selector(
                    "app-market-breadth-heatmap-chart ion-text.block-value",
                    timeout=timeout_ms,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("stockedge.breadth.grid_wait_failed", error=str(exc))
            rows = _extract_rows_from_dom(page)
        finally:
            context.close()
            browser.close()

    if not rows:
        rows = _rows_from_captured_json(captured_json)

    if not rows:
        raise RuntimeError(
            "StockEdge breadth capture produced no rows — the session may be "
            "logged out or the page layout changed. Re-generate storage_state "
            "and retry, or ingest a captured sample with --from-json."
        )

    payload = {
        "dataset": "market_breadth",
        "as_of_date": date.today().isoformat(),
        "exchange": exchange,
        "unit": "percent",
        "source_url": SOURCE_URL,
        "columns": list(("rs_pos", "sma20", "sma50", "sma100", "sma200")),
        "rows": rows,
    }
    log.info("stockedge.breadth.captured", rows=len(rows),
             via="dom" if rows and not captured_json else "mixed")
    return payload


def _extract_rows_from_dom(page) -> list[dict]:  # pragma: no cover - needs a live browser
    """Scrape the breadth grid from the rendered DOM via in-page JS.

    StockEdge renders breadth as ``<app-market-breadth-heatmap-chart>`` — an
    Ionic heatmap, NOT an HTML ``<table>``. Each ``ion-row`` of the master
    ``ion-grid`` carries the index name (``ion-label``), the constituent count,
    and five ``ion-text.block-value`` percentage cells in column order
    ``RS>0, SMA20, SMA50, SMA100, SMA200``. The grid also *duplicates* each row
    in an overlay layer and *virtualizes* (only visible rows are in the DOM), so
    we scroll-accumulate and dedupe by index name.

    This JS was validated verbatim against the live page (Nifty Bank →
    64/93/93/64/79, count 14). Returns row dicts in the captured-payload shape.
    """
    js = r"""
    async () => {
      const sleep = (ms) => new Promise(r => setTimeout(r, ms));
      const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();
      const num = (s) => {
        if (s == null) return null;
        const m = norm(String(s)).replace('%', '').match(/-?\d+(\.\d+)?/);
        return m ? parseFloat(m[0]) : null;
      };
      const host = document.querySelector('app-market-breadth-heatmap-chart');
      if (!host) return [];
      const acc = new Map();
      const harvest = () => {
        const grids = Array.from(host.querySelectorAll('ion-grid'));
        // master grid = the one carrying BOTH index labels and value cells
        const master = grids.find(
          (g) => g.querySelector('ion-label') && g.querySelector('ion-text.block-value')
        );
        if (!master) return;
        for (const r of Array.from(master.querySelectorAll(':scope > ion-row'))) {
          const label = r.querySelector('ion-label');
          if (!label) continue;
          const name = norm(label.innerText);
          if (!name || acc.has(name)) continue;
          const blocks = Array.from(r.querySelectorAll('ion-text.block-value'))
            .map((b) => num(b.innerText));
          if (blocks.length < 5) continue;
          // constituent count = the bare integer between the name and the first %
          const rowText = norm(r.innerText);
          const after = rowText.indexOf(name) === 0
            ? rowText.slice(name.length).trim() : rowText;
          const cm = after.replace(/\d+(\.\d+)?%.*/, '').match(/\d+/);
          acc.set(name, {
            index_name: name,
            constituent_count: cm ? parseInt(cm[0], 10) : null,
            rs_pos: blocks[0], sma20: blocks[1], sma50: blocks[2],
            sma100: blocks[3], sma200: blocks[4],
          });
        }
      };
      // Scroll the Ionic content's inner scroller (shadow DOM), else the window,
      // harvesting after each step to defeat virtualization.
      const content = document.querySelector('ion-content');
      const scroller = content && content.shadowRoot
        ? content.shadowRoot.querySelector('.inner-scroll') : null;
      const getMax = () => scroller ? scroller.scrollHeight : document.body.scrollHeight;
      const getClient = () => scroller ? scroller.clientHeight : window.innerHeight;
      const setTop = (y) => scroller ? scroller.scrollTo(0, y) : window.scrollTo(0, y);
      setTop(0); await sleep(150); harvest();
      let pos = 0, prev = -1, stable = 0;
      for (let i = 0; i < 60; i++) {
        pos += Math.max(150, Math.floor(getClient() * 0.6));
        setTop(pos); await sleep(160); harvest();
        if (acc.size === prev) { if (++stable >= 2 && pos >= getMax()) break; }
        else { stable = 0; }
        prev = acc.size;
        if (pos >= getMax() + getClient()) break;
      }
      setTop(0);
      return Array.from(acc.values());
    }
    """
    try:
        rows = page.evaluate(js)
    except Exception as exc:  # noqa: BLE001
        log.warning("stockedge.breadth.dom_eval_failed", error=str(exc))
        return []
    return [r for r in (rows or []) if r.get("index_name")]


def _rows_from_captured_json(captured: list[dict[str, Any]]) -> list[dict]:
    """Best-effort normalization of a captured api.stockedge.com breadth body.

    The exact JSON shape isn't enumerable out-of-band (§2.3 of the design doc),
    so this scans the captured bodies for a list of objects carrying an
    index name + SMA-ish keys and maps them into our row shape. Returns ``[]``
    if nothing matches (caller then errors clearly).
    """
    def _coerce_records(body: Any) -> list[dict]:
        if isinstance(body, list):
            return [r for r in body if isinstance(r, dict)]
        if isinstance(body, dict):
            for key in ("Data", "data", "Result", "result", "rows", "Records"):
                val = body.get(key)
                if isinstance(val, list):
                    return [r for r in val if isinstance(r, dict)]
        return []

    def _pick(rec: dict, *names: str):
        for n in names:
            for k, v in rec.items():
                if k.lower().replace("_", "").replace("%", "") == n:
                    return v
        return None

    for cap in captured:
        records = _coerce_records(cap.get("body"))
        rows: list[dict] = []
        for rec in records:
            name = _pick(rec, "indexname", "index", "name", "securityname")
            if not name:
                continue
            rows.append({
                "index_name": str(name),
                "constituent_count": _pick(rec, "constituentcount", "count", "numconstituents"),
                "rs_pos": _pick(rec, "rspos", "rspositive", "rs"),
                "sma20": _pick(rec, "sma20", "abovesma20", "pctsma20"),
                "sma50": _pick(rec, "sma50", "abovesma50", "pctsma50"),
                "sma100": _pick(rec, "sma100", "abovesma100", "pctsma100"),
                "sma200": _pick(rec, "sma200", "abovesma200", "pctsma200"),
            })
        if rows:
            return rows
    return []


def load_storage_state(path: str) -> dict:
    """Load a Playwright storage_state JSON file (small helper for callers)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
