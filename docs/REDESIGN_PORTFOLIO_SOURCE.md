# Decision: the portfolio source of truth (UX redesign R0)

Date: 2026-10-21. This document records the decision only. The R2 overview endpoint and page are built later.

## Decision

**The v2 `apps.trading` models, read per `Portfolio`, are the single source of truth for the Portfolio
destination.** Each tenant's book is the set of its `Portfolio` rows, and each row has a role:

- the trading portfolio, which is the oldest row (on this machine, `Default`);
- the `Mutual Funds` portfolio, which holds `MFHolding`.

Position-level truth comes from these tables:

| Asset class | Table |
|---|---|
| Equity | `Position` and `Trade` |
| Options | `OptionsPosition` and `OptionsLeg` |
| Mutual funds | `MFHolding` (valued at the latest `MFNav`) |

History (the equity curve and drawdown) comes from `PortfolioSnapshot`, **filtered by portfolio**.

Two other data sources keep a narrower role:

- **`BrokerPositionSnapshot`** is the reconciliation input that shows what the broker account actually holds. It is not the
  book. It is shown next to the book with its own freshness. It never silently replaces the book.
- **`GET /api/v1/portfolios/summary/`** (`apps.trading.api.legacy_compat_views.portfolio`) is the
  legacy summary. It stays only as a fallback until the R2 `GET /api/v1/portfolio/overview/` endpoint replaces it.
  After that it is deleted, together with `DashboardPage`'s mixed queries.

## What the database holds today

The tenant is `istpduckymomo`. I queried the tables on this machine on 2026-10-21:

| Table | Rows | Notes |
|---|---|---|
| `trading.Portfolio` | 2 | `Default` (paper, capital 500,000, realized 0) and `Mutual Funds` (paper, capital 0) |
| `trading.PortfolioSnapshot` | 3,504 | 1,765 per portfolio, written every 60 s by `refresh-portfolio-snapshots`. All of them show equity 0 / P&L 0 / 0 open positions |
| `trading.Position`, `Trade`, `Order`, `OptionsPosition` | 0 | No paper or live trades yet |
| `mutual_funds.MFHolding` / `MFTransaction` | 0 | No CAS imported yet (MF-2) |
| `market_data.BrokerPositionSnapshot` | 1,873 | Angel One link, refreshed every 30 s. `positions=[]`, `holdings=[]`, margin 0 |
| Legacy SQLite `db.sqlite3` | none | The file is gone from both the repo root and `backend/`. The legacy tables no longer exist on this machine |

**No table holds non-zero portfolio data yet.** The decision therefore rests on structure, not on where the data is today:

- the v2 tables are tenant-scoped and per-portfolio;
- they are what the order path (`place_order`, then `RiskEngine`, then paper fills) writes;
- they are already where MF holdings attach.

There is no competing populated store to migrate from.

## Bug found while checking (fix in R2, not now)

`legacy_compat_views.portfolio` reads `PortfolioSnapshot.objects.filter(tenant=tenant).order_by("-captured_at").first()`.
It does **not** filter by portfolio. Now that the tenant has two portfolios, the newest snapshot can belong to `Mutual Funds`, whose equity is 0. The view
then back-computes `invested = capital - equity + day_pnl = 500,000`, so the Desk can show the full capital as invested and 0
available. The R2 overview must read snapshots per portfolio.

## Rules for R2 that follow from this

- The total value is the trading portfolio's equity, plus the options mark-to-market, plus MF holdings at the latest NAV. Each
  component carries its own freshness: the snapshot `captured_at`, the option LTP time, and the MF `catalog_nav_date`.
- A component with no data is shown as **"not connected"** or **"no holdings"**, never as 0. Today that applies to every
  component.
- Broker-side holdings (`BrokerPositionSnapshot`) appear as a "held at broker" reconciliation panel. If the broker and the book disagree, the page
  shows the discrepancy rather than picking one.
- Mutual funds stay in their own `Portfolio`. Code that needs "the trading portfolio" keeps choosing the oldest row
  (`order_by("created_at")`), as it does today.
