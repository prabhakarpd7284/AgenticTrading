"""
Stock Universe — liquid NSE stocks for intraday trading.

We trade NIFTY 50 components only:
  - Best liquidity → tightest spreads → reliable fills
  - High volumes → no impact cost on our position sizes
  - Well-covered by news → information edge possible
"""

# NIFTY 50 components (updated March 2026)
# Symbol → Angel One trading symbol format
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
    "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "ITC", "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK",
    "LT", "M&M", "MARUTI", "NTPC", "NESTLEIND",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "M&M", "TATAPOWER",
    "TATASTEEL", "TECHM", "TITAN", "ULTRACEMCO", "WIPRO",
]

# High-volume subset for faster scanning (top 20 by avg daily volume)
# These are the stocks where price structures work best
HIGH_VOLUME_20 = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "SBIN", "AXISBANK", "BHARTIARTL", "M&M", "KOTAKBANK",
    "BAJFINANCE", "LT", "ITC", "HINDUNILVR", "TATASTEEL",
    "HINDALCO", "ADANIENT", "SUNPHARMA", "MARUTI", "WIPRO",
]

# Bank Nifty components (for sector plays)
BANKNIFTY = [
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "INDUSINDBK", "BAJFINANCE", "BAJAJFINSV", "BANDHANBNK", "FEDERALBNK",
    "IDFCFIRSTB", "PNB",
]


def get_universe(tier: str = "nifty50") -> list:
    """
    Get stock universe by tier.

    Args:
        tier: 'nifty50' (50 stocks), 'high_volume' (top 20), 'banknifty' (12 banks)
    """
    if tier == "high_volume":
        return HIGH_VOLUME_20.copy()
    elif tier == "banknifty":
        return BANKNIFTY.copy()
    else:
        return NIFTY50.copy()
