"""Market data constants.

Migrated from the legacy `dashboard_utils.market_scanner` module so the
Streamlit dashboard can be retired. Kept lean — just the symbol lists
the v2 services need.
"""
from __future__ import annotations


# ── NIFTY 50 constituents (as of 2026) ────────────────────────────────
NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "TITAN",
    "SUNPHARMA", "HCLTECH", "ASIANPAINT", "NTPC", "M&M",
    "WIPRO", "ULTRACEMCO", "POWERGRID", "ONGC", "NESTLEIND",
    "JSWSTEEL", "TECHM", "ADANIENT", "ADANIPORTS",
    "TATASTEEL", "BAJAJFINSV", "COALINDIA", "HINDALCO", "GRASIM",
    "DIVISLAB", "BPCL", "CIPLA", "DRREDDY", "EICHERMOT",
    "APOLLOHOSP", "HEROMOTOCO", "INDUSINDBK", "TATACONSUM", "BRITANNIA",
    "SBILIFE", "BAJAJ-AUTO", "HDFCLIFE", "ETERNAL", "SHRIRAMFIN",
]

# ── NIFTY Next 50 + popular mid-caps ──────────────────────────────────
NIFTY_NEXT50_SYMBOLS = [
    "ABB", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "BANKBARODA",
    "BEL", "BERGEPAINT", "BOSCHLTD", "CANBK", "CHOLAFIN",
    "COLPAL", "DLF", "GODREJCP", "HAVELLS", "HAL",
    "ICICIPRULI", "INDHOTEL", "IOC", "IRCTC", "JINDALSTEL",
    "JIOFIN", "LICI", "LUPIN", "MAXHEALTH", "NHPC",
    "NAUKRI", "OFSS", "PAGEIND", "PFC", "PIDILITIND",
    "PNB", "POLYCAB", "RECLTD", "SBICARD", "SIEMENS",
    "SRF", "TORNTPHARM", "TRENT", "TVSMOTOR", "UNIONBANK",
    "VEDL", "DMART", "PAYTM", "PIIND",
    "MFSL", "IDFCFIRSTB", "FEDERALBNK", "TATAPOWER", "MOTHERSON",
]

# ── Full screener universe (NIFTY 50 + Next 50) ───────────────────────
SCREENER_UNIVERSE = NIFTY_50_SYMBOLS + NIFTY_NEXT50_SYMBOLS
