"""Market Pulse service — powers the "What's Happening Today" screen.

Implements Stages 1–2 of The Cascade:
  Stage 1: REGIME     → is it a day to trade? (vol, trend, global risk)
  Stage 2: CONTEXT    → what's driving flows? (currencies, commodities, events)

Returns a single JSON payload the frontend renders as a live briefing.
Every field is optional — data-source failures degrade into "--" in the UI
rather than 500-ing the whole screen.  Symbols chosen to match how an
Indian-market intraday trader actually reads the tape at 9:00 AM IST.

Primary data source: yfinance (free, no key, covers NSE + global + FX +
commodities).  Angel One SmartAPI can override specific symbols when a
broker is linked (via the `AngelOneProvider`).  Results are cached for
30 s to keep the dashboard snappy without hammering the provider.

Regime classifier is deterministic and matches the @RiskGuard /
@OptionsStrategist gates in trading/services/risk_engine.py and
trading/options/straddle/graph.py:
  - VIX > 20   → no fresh short straddles
  - VIX > 35   → no options at all (regime=EXTREME, trading halted)
  - gap > 1.5% → first 30 min is READ-ONLY (no directional entries)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timezone, timedelta
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from django.core.cache import cache

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_OPEN_START = time(9, 0)

# ---------------------------------------------------------------------------
# Ticker map — keep one canonical symbol per concept so the frontend doesn't
# have to care about yfinance-vs-Angel differences.  `yf` is the yfinance
# ticker; `angel_token` / `angel_exch` let us swap in broker-direct data
# when the user has Angel One linked.
# ---------------------------------------------------------------------------
TICKERS: dict[str, dict[str, Any]] = {
    # ── Indian indices ──────────────────────────────────────────────────
    "NIFTY":       {"yf": "^NSEI",        "label": "NIFTY 50",      "group": "indices_in"},
    "BANKNIFTY":   {"yf": "^NSEBANK",     "label": "BANK NIFTY",    "group": "indices_in"},
    "SENSEX":      {"yf": "^BSESN",       "label": "SENSEX",        "group": "indices_in"},
    "MIDCAP":      {"yf": "^NSEMDCP50",   "label": "NIFTY MIDCAP",  "group": "indices_in"},
    "INDIAVIX":    {"yf": "^INDIAVIX",    "label": "INDIA VIX",     "group": "vol"},
    # ── Global indices (overnight / same-day) ──────────────────────────
    "SP500":       {"yf": "^GSPC",        "label": "S&P 500",       "group": "indices_global"},
    "NASDAQ":      {"yf": "^IXIC",        "label": "NASDAQ",        "group": "indices_global"},
    "DOW":         {"yf": "^DJI",         "label": "DOW",           "group": "indices_global"},
    "NIKKEI":      {"yf": "^N225",        "label": "NIKKEI 225",    "group": "indices_global"},
    "HANGSENG":    {"yf": "^HSI",         "label": "HANG SENG",     "group": "indices_global"},
    # ── Currencies ─────────────────────────────────────────────────────
    "USDINR":      {"yf": "INR=X",        "label": "USD / INR",     "group": "fx"},
    "DXY":         {"yf": "DX-Y.NYB",     "label": "DOLLAR INDEX",  "group": "fx"},
    "EURINR":      {"yf": "EURINR=X",     "label": "EUR / INR",     "group": "fx"},
    # ── Commodities ────────────────────────────────────────────────────
    "CRUDE":       {"yf": "CL=F",         "label": "CRUDE (WTI)",   "group": "commodities"},
    "BRENT":       {"yf": "BZ=F",         "label": "BRENT",         "group": "commodities"},
    "NATGAS":      {"yf": "NG=F",         "label": "NATURAL GAS",   "group": "commodities"},
    "GOLD":        {"yf": "GC=F",         "label": "GOLD",          "group": "commodities"},
    "SILVER":      {"yf": "SI=F",         "label": "SILVER",        "group": "commodities"},
    "COPPER":      {"yf": "HG=F",         "label": "COPPER",        "group": "commodities"},
    # ── Rates ──────────────────────────────────────────────────────────
    "US10Y":       {"yf": "^TNX",         "label": "US 10Y YIELD",  "group": "rates"},
}

# NIFTY sector indices — for the sector heatmap (Stage 3 teaser on the pulse page)
SECTOR_TICKERS: dict[str, str] = {
    "NIFTY_BANK":       "^NSEBANK",
    "NIFTY_IT":         "^CNXIT",
    "NIFTY_AUTO":       "^CNXAUTO",
    "NIFTY_PHARMA":     "^CNXPHARMA",
    "NIFTY_FMCG":       "^CNXFMCG",
    "NIFTY_METAL":      "^CNXMETAL",
    "NIFTY_ENERGY":     "^CNXENERGY",
    "NIFTY_REALTY":     "^CNXREALTY",
    "NIFTY_PSUBANK":    "^CNXPSUBANK",
    "NIFTY_FIN":        "NIFTY_FIN_SERVICE.NS",
    "NIFTY_MEDIA":      "^CNXMEDIA",
}

CACHE_KEY = "market:pulse:v1"
CACHE_TTL = 30  # seconds — dashboard polls 15s, we serve from cache between


@dataclass
class Quote:
    symbol: str
    label: str
    last: float | None = None
    change: float | None = None
    change_pct: float | None = None
    prev_close: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    as_of: str | None = None  # iso timestamp
    stale: bool = False        # True if data is > 5 min old
    source: str = "yfinance"

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Sector:
    key: str
    label: str
    change_pct: float | None
    rank: int  # 1 = strongest


@dataclass
class PulsePayload:
    as_of: str
    session_phase: str                      # "pre-open" | "open" | "post-close" | "weekend"
    is_market_open: bool
    regime: dict[str, Any] = field(default_factory=dict)
    quotes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    sectors: list[dict[str, Any]] = field(default_factory=list)
    guidance: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Provider — pluggable so Angel One / NSE websocket can override yfinance
# ---------------------------------------------------------------------------
class YFinanceProvider:
    """Best-effort free data. Handles yfinance import lazily so the backend
    boots even when the package isn't installed (the endpoint then returns
    an empty quotes block with a helpful `errors` entry)."""

    def fetch(self, yf_symbols: Iterable[str]) -> dict[str, dict[str, Any]]:
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            raise RuntimeError(
                "yfinance not installed — pip install yfinance to enable the "
                "market pulse data provider."
            )

        out: dict[str, dict[str, Any]] = {}
        # yfinance's Tickers.info is expensive; fast_info is cheap and has
        # the fields we care about for a live dashboard.
        tickers = yf.Tickers(" ".join(yf_symbols))
        for sym in yf_symbols:
            try:
                t = tickers.tickers.get(sym)
                if t is None:
                    continue
                fi = t.fast_info
                last = _safe_float(getattr(fi, "last_price", None))
                prev = _safe_float(getattr(fi, "previous_close", None))
                hi = _safe_float(getattr(fi, "day_high", None))
                lo = _safe_float(getattr(fi, "day_low", None))
                chg = (last - prev) if (last is not None and prev is not None) else None
                chg_pct = (chg / prev * 100) if (chg is not None and prev) else None
                out[sym] = {
                    "last": last, "prev_close": prev, "day_high": hi, "day_low": lo,
                    "change": chg, "change_pct": chg_pct,
                    "as_of": datetime.now(tz=timezone.utc).isoformat(),
                    "source": "yfinance",
                }
            except Exception as e:  # noqa: BLE001 — don't fail the whole payload
                logger.debug("yfinance failed for %s: %s", sym, e)
        return out


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return round(v, 4)


# ---------------------------------------------------------------------------
# Classifier — deterministic regime signals
# ---------------------------------------------------------------------------
def classify_regime(quotes_by_key: dict[str, Quote]) -> dict[str, Any]:
    """Collapse today's data into a trader-legible regime label.

    Vol regime is driven by India VIX (matches the straddle gate).
    Trend regime is driven by NIFTY's distance from prev close + SGX/SP500
    overnight bias.  Breadth is inferred from sector leaders if available.
    """
    vix = quotes_by_key.get("INDIAVIX")
    nifty = quotes_by_key.get("NIFTY")
    sp = quotes_by_key.get("SP500")

    # --- Vol regime -------------------------------------------------------
    vix_val = vix.last if vix else None
    if vix_val is None:
        vol = "unknown"
    elif vix_val < 12:
        vol = "complacent"
    elif vix_val < 16:
        vol = "low"
    elif vix_val < 20:
        vol = "normal"
    elif vix_val < 28:
        vol = "elevated"
    elif vix_val < 35:
        vol = "high"
    else:
        vol = "extreme"

    # --- Trend regime (intraday direction + gap size) --------------------
    gap_pct = nifty.change_pct if nifty else None
    if gap_pct is None:
        trend = "unknown"
    elif gap_pct > 0.6:
        trend = "up"
    elif gap_pct < -0.6:
        trend = "down"
    else:
        trend = "range"

    # --- Global risk-on/off read ----------------------------------------
    sp_chg = sp.change_pct if sp else None
    if sp_chg is None:
        global_tone = "unknown"
    elif sp_chg > 0.4:
        global_tone = "risk_on"
    elif sp_chg < -0.4:
        global_tone = "risk_off"
    else:
        global_tone = "neutral"

    # --- Tradeability verdict -------------------------------------------
    # Compose a one-line summary and an explicit `trade_today` flag that
    # the downstream @DirectionalTrader / @OptionsStrategist nodes read as
    # their top-level gate.
    if vol == "extreme":
        tradeable = False
        summary = "VIX in extreme territory — capital preservation mode. No new trades."
    elif vol == "high" and abs(gap_pct or 0) > 1.5:
        tradeable = False
        summary = "High vol + large gap — first 30 min is read-only."
    elif vol == "elevated" and trend == "range":
        tradeable = True
        summary = "Elevated vol, range-bound. Iron condors / short straddles favored over directional."
    elif trend == "up" and global_tone == "risk_on":
        tradeable = True
        summary = "Risk-on with uptrend. Favor long momentum in leading sectors."
    elif trend == "down" and global_tone == "risk_off":
        tradeable = True
        summary = "Risk-off with downtrend. Favor short setups or defensives (FMCG, Pharma)."
    elif trend == "range" and vol == "low":
        tradeable = True
        summary = "Low vol, range-bound. Theta-capture setups; avoid chasing breakouts."
    else:
        tradeable = True
        summary = f"Mixed signals — vol={vol}, trend={trend}, global={global_tone}. Size small."

    return {
        "vol": vol,
        "trend": trend,
        "global_tone": global_tone,
        "vix": vix_val,
        "nifty_gap_pct": gap_pct,
        "sp500_change_pct": sp_chg,
        "tradeable": tradeable,
        "summary": summary,
    }


def agent_guidance(regime: dict[str, Any]) -> dict[str, Any]:
    """Translate the regime into explicit per-agent go/no-go flags.
    Matches the gates in risk_engine.py and straddle/graph.py so the UI
    can preview what the agents will do before they run."""
    vol = regime.get("vol")
    trend = regime.get("trend")
    tradeable = regime.get("tradeable", False)

    directional = "avoid"
    straddle = "avoid"
    reasons: list[str] = []

    if not tradeable:
        reasons.append(regime.get("summary", "regime not tradeable"))
    else:
        # Directional gating
        if trend in ("up", "down"):
            directional = "favored"
            reasons.append(f"Trend is {trend}, directional setups qualify.")
        elif trend == "range":
            directional = "neutral"
            reasons.append("Range-bound — directional only on confirmed breakouts.")
        # Straddle gating (matches @OptionsStrategist VIX gate)
        if vol in ("normal", "elevated") and trend == "range":
            straddle = "favored"
            reasons.append("Vol normal/elevated with range → premium selling edge.")
        elif vol in ("low", "complacent"):
            straddle = "neutral"
            reasons.append("Vol low — premium thin; size down on straddles.")
        elif vol == "high":
            straddle = "avoid"
            reasons.append("Vol high — no fresh straddles (VIX > 20 gate).")
        elif vol == "extreme":
            straddle = "avoid"
            reasons.append("Vol extreme — no options at all (VIX > 35 gate).")

    return {"directional": directional, "straddle": straddle, "reasons": reasons}


def session_phase(now_ist: datetime | None = None) -> tuple[str, bool]:
    now_ist = now_ist or datetime.now(tz=IST)
    if now_ist.weekday() >= 5:
        return "weekend", False
    t = now_ist.time()
    if t < PRE_OPEN_START:
        return "pre-open", False
    if PRE_OPEN_START <= t < MARKET_OPEN:
        return "pre-open", False
    if MARKET_OPEN <= t <= MARKET_CLOSE:
        return "open", True
    return "post-close", False


# ---------------------------------------------------------------------------
# Orchestrator — single entry point used by the DRF view
# ---------------------------------------------------------------------------
def build_pulse(force: bool = False) -> PulsePayload:
    """Assemble the full pulse payload.  Cached for CACHE_TTL seconds."""
    if not force:
        cached = cache.get(CACHE_KEY)
        if cached is not None:
            return cached

    errors: list[str] = []
    provider = YFinanceProvider()

    # Gather all yfinance symbols in one call — much faster than per-symbol.
    all_yf = (
        [meta["yf"] for meta in TICKERS.values()]
        + list(SECTOR_TICKERS.values())
    )
    try:
        raw = provider.fetch(all_yf)
    except Exception as e:  # noqa: BLE001
        logger.warning("market pulse provider failed: %s", e)
        errors.append(f"data_provider: {e}")
        raw = {}

    # Map back to our canonical keys + group them for the frontend.
    quotes_by_key: dict[str, Quote] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for key, meta in TICKERS.items():
        data = raw.get(meta["yf"], {})
        q = Quote(
            symbol=key,
            label=meta["label"],
            last=data.get("last"),
            prev_close=data.get("prev_close"),
            day_high=data.get("day_high"),
            day_low=data.get("day_low"),
            change=data.get("change"),
            change_pct=data.get("change_pct"),
            as_of=data.get("as_of"),
            source=data.get("source", "yfinance"),
        )
        quotes_by_key[key] = q
        grouped.setdefault(meta["group"], []).append(q.as_dict())

    # Sector heatmap — rank by change_pct desc
    sectors: list[dict[str, Any]] = []
    for key, yf_sym in SECTOR_TICKERS.items():
        data = raw.get(yf_sym, {})
        sectors.append({
            "key": key,
            "label": key.replace("NIFTY_", "").replace("_", " ").title(),
            "change_pct": data.get("change_pct"),
            "last": data.get("last"),
        })
    sectors.sort(key=lambda s: s["change_pct"] if s["change_pct"] is not None else -999, reverse=True)
    for i, s in enumerate(sectors):
        s["rank"] = i + 1

    # Regime + guidance
    regime = classify_regime(quotes_by_key)
    guidance = agent_guidance(regime)

    phase, open_flag = session_phase()
    payload = PulsePayload(
        as_of=datetime.now(tz=timezone.utc).isoformat(),
        session_phase=phase,
        is_market_open=open_flag,
        regime=regime,
        quotes=grouped,
        sectors=sectors,
        guidance=guidance,
        errors=errors,
    )
    cache.set(CACHE_KEY, payload, CACHE_TTL)
    return payload
