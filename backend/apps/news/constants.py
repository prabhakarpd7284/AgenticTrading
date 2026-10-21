"""Symbol ↔ company-name mapping for headline matching.

``Symbol.name`` in the DB is unreliable (RELIANCE's name is "RELIANCE"), so
the NIFTY 50 names are listed here. Each entry:

    symbol: (google_query_name, [aliases], [exclusions])

Aliases are matched on word boundaries — case-sensitively for single words
(ITC, SBI, Infosys, Titan …), case-insensitively for multi-word names. A headline matching an exclusion (e.g. "SBI Life" for
SBIN) does not count as a mention of that symbol.
"""
from __future__ import annotations

import re
from functools import lru_cache

COMPANIES: dict[str, tuple[str, list[str], list[str]]] = {
    "RELIANCE": ("Reliance Industries", ["Reliance Industries", "RIL", "Reliance Jio", "Jio Platforms"], []),
    "TCS": ("Tata Consultancy Services", ["TCS", "Tata Consultancy"], []),
    "HDFCBANK": ("HDFC Bank", ["HDFC Bank"], []),
    "INFY": ("Infosys", ["Infosys"], []),
    "ICICIBANK": ("ICICI Bank", ["ICICI Bank"], []),
    "HINDUNILVR": ("Hindustan Unilever", ["Hindustan Unilever", "HUL"], []),
    "ITC": ("ITC Ltd", ["ITC"], ["ITC Hotels"]),
    "SBIN": ("State Bank of India", ["State Bank of India", "SBI"],
             ["SBI Life", "SBI Card", "SBI Mutual", "SBI MF", "SBI Funds", "SBI General"]),
    "BHARTIARTL": ("Bharti Airtel", ["Bharti Airtel", "Airtel"], ["Bharti Hexacom"]),
    "KOTAKBANK": ("Kotak Mahindra Bank", ["Kotak Mahindra Bank", "Kotak Bank", "Kotak Mahindra"], []),
    "LT": ("Larsen & Toubro", ["Larsen & Toubro", "Larsen and Toubro", "L&T"],
           ["L&T Finance", "L&T Technology", "L&T Tech", "LTIMindtree"]),
    "AXISBANK": ("Axis Bank", ["Axis Bank"], []),
    "BAJFINANCE": ("Bajaj Finance", ["Bajaj Finance"], []),
    "MARUTI": ("Maruti Suzuki", ["Maruti Suzuki", "Maruti"], []),
    "TITAN": ("Titan Company", ["Titan Company", "Titan"], []),
    "SUNPHARMA": ("Sun Pharmaceutical", ["Sun Pharma", "Sun Pharmaceutical"], []),
    "HCLTECH": ("HCLTech", ["HCLTech", "HCL Tech", "HCL Technologies"], []),
    "ASIANPAINT": ("Asian Paints", ["Asian Paints"], []),
    "NTPC": ("NTPC", ["NTPC"], ["NTPC Green"]),
    "M&M": ("Mahindra & Mahindra", ["Mahindra & Mahindra", "Mahindra and Mahindra", "M&M"], []),
    "WIPRO": ("Wipro", ["Wipro"], []),
    "ULTRACEMCO": ("UltraTech Cement", ["UltraTech Cement", "UltraTech", "Ultratech"], []),
    "POWERGRID": ("Power Grid Corporation", ["Power Grid", "PowerGrid"], []),
    "ONGC": ("ONGC", ["ONGC", "Oil and Natural Gas"], []),
    "NESTLEIND": ("Nestle India", ["Nestle India", "Nestle"], []),
    "JSWSTEEL": ("JSW Steel", ["JSW Steel"], []),
    "TECHM": ("Tech Mahindra", ["Tech Mahindra", "TechM"], []),
    "ADANIENT": ("Adani Enterprises", ["Adani Enterprises"], []),
    "ADANIPORTS": ("Adani Ports", ["Adani Ports", "APSEZ"], []),
    "TATASTEEL": ("Tata Steel", ["Tata Steel"], []),
    "BAJAJFINSV": ("Bajaj Finserv", ["Bajaj Finserv"], []),
    "COALINDIA": ("Coal India", ["Coal India"], []),
    "HINDALCO": ("Hindalco", ["Hindalco"], []),
    "GRASIM": ("Grasim Industries", ["Grasim"], []),
    "DIVISLAB": ("Divi's Laboratories", ["Divi's Lab", "Divis Lab", "Divi's Laboratories"], []),
    "BPCL": ("Bharat Petroleum", ["BPCL", "Bharat Petroleum"], []),
    "CIPLA": ("Cipla", ["Cipla"], []),
    "DRREDDY": ("Dr Reddy's Laboratories", ["Dr Reddy", "Dr. Reddy", "Dr Reddy's"], []),
    "EICHERMOT": ("Eicher Motors", ["Eicher Motors", "Royal Enfield"], []),
    "APOLLOHOSP": ("Apollo Hospitals", ["Apollo Hospitals"], []),
    "HEROMOTOCO": ("Hero MotoCorp", ["Hero MotoCorp", "Hero Moto"], []),
    "INDUSINDBK": ("IndusInd Bank", ["IndusInd Bank", "IndusInd"], []),
    "TATACONSUM": ("Tata Consumer Products", ["Tata Consumer"], []),
    "BRITANNIA": ("Britannia Industries", ["Britannia"], []),
    "SBILIFE": ("SBI Life Insurance", ["SBI Life"], []),
    "BAJAJ-AUTO": ("Bajaj Auto", ["Bajaj Auto"], []),
    "HDFCLIFE": ("HDFC Life", ["HDFC Life"], []),
    "ETERNAL": ("Eternal Zomato", ["Eternal Ltd", "Eternal", "Zomato", "Blinkit"], []),
    "SHRIRAMFIN": ("Shriram Finance", ["Shriram Finance"], []),
    "TRENT": ("Trent Ltd", ["Trent"], []),
    "BEL": ("Bharat Electronics", ["Bharat Electronics", "BEL"], []),
    "JIOFIN": ("Jio Financial Services", ["Jio Financial"], []),
    "INDIGO": ("InterGlobe Aviation IndiGo", ["InterGlobe", "IndiGo", "Indigo airline"], []),
}

# Index rows for the market-mood line (levels only, no per-stock news).
INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY"]


def company_name(symbol: str) -> str:
    entry = COMPANIES.get(symbol)
    return entry[0] if entry else symbol


def _alias_regex(alias: str) -> re.Pattern:
    esc = re.escape(alias)
    pat = rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])"
    # Single words ("Infosys", "Titan", "ITC") match case-sensitively so a
    # lowercase common word ("eternal", "titan of industry") doesn't count;
    # multi-word names ("Bajaj Finance") match case-insensitively.
    return re.compile(pat) if " " not in alias else re.compile(pat, re.IGNORECASE)


@lru_cache(maxsize=1)
def _compiled() -> dict[str, tuple[list[re.Pattern], list[re.Pattern]]]:
    out = {}
    for sym, (_, aliases, excl) in COMPANIES.items():
        out[sym] = ([_alias_regex(a) for a in aliases + [sym] if len(a) >= 2],
                    [_alias_regex(e) for e in excl])
    return out


def mentions(text: str, symbol: str) -> bool:
    """Does ``text`` mention ``symbol``'s company (honouring exclusions)?"""
    pats = _compiled().get(symbol)
    if pats is None:
        return bool(_alias_regex(symbol).search(text or ""))
    inc, exc = pats
    cleaned = text or ""
    for e in exc:
        cleaned = e.sub(" ", cleaned)
    return any(p.search(cleaned) for p in inc)


def match_symbols(text: str, universe: list[str]) -> list[str]:
    return [s for s in universe if mentions(text, s)]
