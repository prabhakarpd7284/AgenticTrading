"""
Oliver Kell Cycle of Price Action — Swing/Positional Scanner.

Operates on daily/weekly charts to detect cycle phases:
  Bullish: RE (Reversal Extension) → WP (Wedge Pop) → EC (EMA Crossback) → BB (Basin Break)
  Bearish: EX (Exhaustion Extension) → WD (Wedge Drop) → EC (Bear Crossback) → BB (Bear Break)

Multi-timeframe trend confirmation: daily + weekly EMA stacking.
"""
