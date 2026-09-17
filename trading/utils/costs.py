"""
Trading cost models — slippage + brokerage + taxes.

Used by the backtester to produce realistic P&L.
Without costs, backtests overstate returns by 5-15%.
"""


def estimate_slippage(entry_price: float, atr: float, side: str) -> float:
    """
    Realistic fill price = entry ± slippage.

    Slippage is 5% of ATR — accounts for spread + partial fills.
    On liquid NIFTY50 stocks this is 1-3 pts. On options, 2-5 pts.

    Returns the slipped entry price (worse than requested).
    """
    slip = atr * 0.05
    if side == "BUY":
        return entry_price + slip  # Buy higher than requested
    return entry_price - slip      # Sell lower than requested


def estimate_charges(entry_price: float, exit_price: float, quantity: int,
                     exchange: str = "NSE", product: str = "INTRADAY") -> float:
    """
    Angel One charges for intraday equity/options.

    Returns total charges in INR (deducted from P&L).

    Components:
      - Brokerage: Rs 20 per order (buy + sell = Rs 40)
      - STT: 0.0125% on sell side (intraday equity)
      - Exchange txn: 0.00345% on turnover
      - GST: 18% on brokerage + exchange txn
      - SEBI charges: Rs 10 per crore
      - Stamp duty: 0.003% on buy side
    """
    buy_value = entry_price * quantity
    sell_value = exit_price * quantity
    turnover = buy_value + sell_value

    # Brokerage: flat Rs 20 per order (Angel One)
    brokerage = 40  # buy + sell

    # STT: 0.0125% on sell side (intraday equity)
    if exchange == "NSE" and product == "INTRADAY":
        stt = sell_value * 0.000125
    elif exchange == "NFO":
        stt = sell_value * 0.000625  # 0.0625% on options sell
    else:
        stt = sell_value * 0.001  # delivery

    # Exchange transaction charges
    if exchange == "NSE":
        exchange_txn = turnover * 0.0000345
    else:  # NFO
        exchange_txn = turnover * 0.0000495

    # GST: 18% on (brokerage + exchange txn)
    gst = (brokerage + exchange_txn) * 0.18

    # SEBI charges: Rs 10 per crore
    sebi = turnover * 0.000001

    # Stamp duty: 0.003% on buy side
    stamp = buy_value * 0.00003

    total = brokerage + stt + exchange_txn + gst + sebi + stamp
    return round(total, 2)


def net_pnl(entry: float, exit_price: float, quantity: int, side: str,
            atr: float = 0, include_slippage: bool = True,
            exchange: str = "NSE") -> float:
    """
    Calculate net P&L after slippage + charges.

    Args:
        entry: Intended entry price
        exit_price: Actual exit price (SL, target, or EOD)
        quantity: Number of shares/lots
        side: "BUY" or "SELL"
        atr: Average True Range (for slippage calc)
        include_slippage: Whether to apply slippage
        exchange: "NSE" or "NFO"

    Returns:
        Net P&L in INR after all costs.
    """
    # Apply slippage to entry
    if include_slippage and atr > 0:
        slipped_entry = estimate_slippage(entry, atr, side)
    else:
        slipped_entry = entry

    # Gross P&L
    if side == "BUY":
        gross = (exit_price - slipped_entry) * quantity
    else:
        gross = (slipped_entry - exit_price) * quantity

    # Charges
    charges = estimate_charges(slipped_entry, exit_price, quantity, exchange)

    return round(gross - charges, 2)
