"""
Expiry utilities — convert between date formats for Angel One NFO options.

Angel One uses: DDMMMYY (e.g. "17MAR26") in symbols, DDMMMYYYY (e.g. "17MAR2026") in API params.
We store: YYYY-MM-DD (ISO format) in Django models.
"""
import re
from datetime import date, datetime
from typing import Optional, Tuple

MONTH_NAMES = {
    "01": "JAN", "02": "FEB", "03": "MAR", "04": "APR",
    "05": "MAY", "06": "JUN", "07": "JUL", "08": "AUG",
    "09": "SEP", "10": "OCT", "11": "NOV", "12": "DEC",
}

MONTH_NUMS = {v: k for k, v in MONTH_NAMES.items()}


def iso_to_angel(iso_date: str) -> Optional[str]:
    """
    Convert ISO date to Angel One symbol format.
    "2026-03-17" → "17MAR26"
    """
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', iso_date)
    if not m:
        return None
    year, month, day = m.group(1), m.group(2), m.group(3)
    mon = MONTH_NAMES.get(month)
    if not mon:
        return None
    return f"{day}{mon}{year[2:]}"


def iso_to_angel_long(iso_date: str) -> Optional[str]:
    """
    Convert ISO date to Angel One API format (used by optionGreek).
    "2026-03-17" → "17MAR2026"
    """
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', iso_date)
    if not m:
        return None
    year, month, day = m.group(1), m.group(2), m.group(3)
    mon = MONTH_NAMES.get(month)
    if not mon:
        return None
    return f"{day}{mon}{year}"


def angel_to_iso(angel_str: str) -> Optional[str]:
    """
    Convert Angel One format to ISO date.
    "17MAR26" → "2026-03-17"
    "17MAR2026" → "2026-03-17"
    """
    m = re.match(r'^(\d{1,2})([A-Z]{3})(\d{2,4})$', angel_str.strip().upper())
    if not m:
        return None
    day, mon, year = m.group(1), m.group(2), m.group(3)
    month_num = MONTH_NUMS.get(mon)
    if not month_num:
        return None
    if len(year) == 2:
        year = f"20{year}"
    return f"{year}-{month_num}-{day.zfill(2)}"


def normalize_expiry(expiry_str: str) -> Optional[str]:
    """
    Normalize any expiry format to DDMMMYY (Angel One symbol format).
    Accepts: "17MAR26", "17MAR2026", "2026-03-17"
    Returns: "17MAR26"
    """
    # Already in DDMMMYY format
    m = re.match(r'^(\d{1,2})([A-Z]{3})(\d{2})$', expiry_str.strip().upper())
    if m:
        return f"{m.group(1).zfill(2)}{m.group(2)}{m.group(3)}"

    # DDMMMYYYY format
    m = re.match(r'^(\d{1,2})([A-Z]{3})(\d{4})$', expiry_str.strip().upper())
    if m:
        return f"{m.group(1).zfill(2)}{m.group(2)}{m.group(3)[2:]}"

    # ISO format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', expiry_str):
        return iso_to_angel(expiry_str)

    return None


def days_to_expiry(expiry: str) -> int:
    """
    Return calendar days to expiry. 0 = expiry today.
    Accepts ISO ("2026-03-17") or date object.
    """
    if isinstance(expiry, date):
        return max(0, (expiry - date.today()).days)
    try:
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        return max(0, (expiry_date - date.today()).days)
    except (ValueError, TypeError):
        return 999


def next_expiry_date(underlying: str = "NIFTY") -> Optional[date]:
    """
    Find the next expiry date from the actual scrip master.
    Falls back to hardcoded weekday if scrip master unavailable.
    """
    from datetime import timedelta
    today = date.today()

    # Try scrip master first — authoritative source
    try:
        from trading.services.ticker_service import ticker_service
        ticker_service._ensure_loaded()

        expiries = set()
        for key, inst in ticker_service._nfo_by_key.items():
            if inst.get("name") == underlying and inst.get("instrumenttype") == "OPTIDX":
                exp_str = inst.get("expiry", "")
                if exp_str:
                    try:
                        # Angel One format: "30MAR2026" or "24MAR2026"
                        exp_date = datetime.strptime(exp_str, "%d%b%Y").date()
                        if exp_date >= today:
                            expiries.add(exp_date)
                    except (ValueError, TypeError):
                        pass

        if expiries:
            return min(expiries)
    except Exception:
        pass

    # Fallback: hardcoded weekday
    expiry_weekday = 1 if underlying == "NIFTY" else 2  # Tue=1, Wed=2
    days_ahead = (expiry_weekday - today.weekday()) % 7
    if days_ahead == 0:
        return today
    return today + timedelta(days=days_ahead)
