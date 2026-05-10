"""
Morning Basket Strategy — intraday equity + ATM index options.

Workflow:
  1. Assess market mood (A/D ratio, VIX, gap, first-hour range)
  2. Generate equity signals from OK scanner + 5 EMA / BB momentum
  3. Generate ATM option signal (CE if bullish, PE if bearish)
  4. Execute with scale-in tranches + pyramid on winners
  5. Manage with trailing SL at 5 EMA, breakeven, EOD close
"""
