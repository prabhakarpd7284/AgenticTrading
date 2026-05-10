"""
Monthly Report Service — aggregates trading activity into a feedback report.

The report answers:
  - What did AlphaDesk earn/lose this month?
  - What signals did it detect vs act on?
  - For each stock: how much did it move, how much was captured?
  - What did @RiskGuard block, and would those have been profitable?
  - AI-generated lessons for continuous improvement.

Data sources:
  - Position (portfolio app) — trades taken, P&L
  - TradeJournal (legacy trading app) — trade decisions + outcomes
  - SignalLog (legacy trading app) — all screener/scanner signals
  - AuditLog (legacy trading app) — risk rejections with gate details
  - WatchlistEntry (legacy trading app) — premarket scan outcomes
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional

from django.core.cache import cache
from django.db.models import Count, Q, Sum
from django.utils import timezone

CACHE_TTL = 120  # seconds


# ══════════════════════════════════════════════════════════════════════
# Data classes — mirror the frontend contract
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PositionLeg:
    id: str
    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: Optional[float]
    entry_date: str
    exit_date: Optional[str]
    target_price: Optional[float]
    stop_price: Optional[float]
    pnl: float
    status: str
    lot_size: Optional[int] = None
    notes: str = ""


@dataclass
class UnderlyingRoll:
    underlying: str
    asset_class: str
    capital_deployed: float
    exposure: float
    running_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    target_total: float
    risk_total: float
    trade_count: int
    winning_trades: int
    losing_trades: int
    avg_rr: float
    days_in_position: int
    pct_of_month: float
    legs: list[PositionLeg]


@dataclass
class MonthGroup:
    month: str
    month_label: str
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    capital_deployed: float
    exposure: float
    trade_count: int
    win_rate: float
    by_asset_class: dict  # {cash: [...], fno: [...], commodity: [...]}


@dataclass
class YtdMonthBar:
    month: str
    month_label: str
    pnl: float


@dataclass
class YtdSummary:
    capital_base: float
    total_pnl: float
    best_month: str
    worst_month: str
    win_rate: float
    trade_count: int
    months: list[YtdMonthBar]


@dataclass
class SignalSummary:
    strategy: str
    rr: float
    outcome: str
    potential_pnl: float = 0.0


@dataclass
class StockCapture:
    symbol: str
    month_move_pct: float
    signals_fired: int
    trades_taken: int
    trades_skipped: int
    captured_pnl: float
    potential_pnl: float
    capture_rate_pct: float
    best_signal: Optional[dict] = None
    worst_miss: Optional[dict] = None


@dataclass
class StrategyPerf:
    count: int
    win_rate: float
    avg_rr: float


@dataclass
class SignalAudit:
    total_signals: int
    by_outcome: dict  # {TRADED: 5, REJECTED: 3, ...}
    by_source: dict   # {SCREENER: 10, OK_SCANNER: 5}
    by_strategy: dict  # {strategy_name: {count, win_rate, avg_rr}}
    profitable_if_taken: int
    loss_avoided: int


@dataclass
class RejectionReview:
    symbol: str
    date: str
    reason: str
    would_have_profited: bool
    hypothetical_pnl: float


@dataclass
class EquityCurvePoint:
    date: str               # "2026-04-01"
    pnl: float              # day's P&L
    cumulative: float       # running total
    trades: int             # trades closed that day
    drawdown: float         # drawdown from peak (negative)


@dataclass
class EquityCurve:
    points: list[EquityCurvePoint]
    max_drawdown: float     # worst peak-to-trough
    max_drawdown_date: str
    peak_equity: float
    final_equity: float


@dataclass
class HourBucket:
    hour: int               # 9, 10, ... 15 (IST)
    label: str              # "09:00"
    trades: int
    wins: int
    losses: int
    pnl: float
    win_rate: float


@dataclass
class DayOfWeekBucket:
    day: int                # 0=Mon .. 4=Fri
    label: str              # "Mon"
    trades: int
    wins: int
    pnl: float
    win_rate: float


@dataclass
class SectorBucket:
    sector: str
    trades: int
    pnl: float
    win_rate: float
    symbols: list[str]


@dataclass
class Analytics:
    by_hour: list[HourBucket]
    by_day_of_week: list[DayOfWeekBucket]
    by_sector: list[SectorBucket]


@dataclass
class BenchmarkComparison:
    portfolio_return_pct: float
    nifty_return_pct: float
    alpha_pct: float         # portfolio - nifty
    trading_days: int
    nifty_start: float       # NIFTY close at month start
    nifty_end: float         # NIFTY close at month end


@dataclass
class MonthlyReport:
    paper_mode: bool
    current_month: str
    generated_at: str
    ytd: YtdSummary
    months: list[MonthGroup]
    capture_matrix: list[StockCapture]
    signal_audit: SignalAudit
    rejections: list[RejectionReview]
    lessons: list[str]
    equity_curve: EquityCurve
    analytics: Analytics
    benchmark: BenchmarkComparison


# ══════════════════════════════════════════════════════════════════════
# Sector mapping (NIFTY50 / common NSE symbols)
# ══════════════════════════════════════════════════════════════════════

SECTOR_MAP = {
    # Banking & Finance
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
    "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSINDBK": "Banking",
    "BAJFINANCE": "Finance", "BAJAJFINSV": "Finance", "HDFCLIFE": "Finance",
    "SBILIFE": "Finance", "MFSL": "Finance",
    # IT
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "LTIM": "IT",
    # Auto
    "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto",
    "BAJAJ-AUTO": "Auto", "EICHERMOT": "Auto", "HEROMOTOCO": "Auto",
    # Metals & Mining
    "TATASTEEL": "Metals", "HINDALCO": "Metals", "JSWSTEEL": "Metals",
    "COALINDIA": "Metals",
    # Pharma & Healthcare
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "APOLLOHOSP": "Healthcare", "DIVISLAB": "Pharma",
    # FMCG
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG",
    "TATACONSUM": "FMCG", "BRITANNIA": "FMCG",
    # Oil & Gas / Energy
    "RELIANCE": "Energy", "ONGC": "Energy", "NTPC": "Power",
    "POWERGRID": "Power", "ADANIENT": "Conglomerate",
    "ADANIPORTS": "Infra",
    # Others
    "ULTRACEMCO": "Cement", "GRASIM": "Cement", "SHREECEM": "Cement",
    "TITAN": "Consumer", "ASIANPAINT": "Consumer",
    "LT": "Infra", "BHARTIARTL": "Telecom",
}

def _get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "Other")


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

FNO_UNDERLYING = re.compile(r"^(NIFTY|BANKNIFTY|FINNIFTY|MIDCPNIFTY)")


def _month_label(month_str: str) -> str:
    """'2026-04' -> 'Apr 2026'"""
    y, m = month_str.split("-")
    return f"{MONTH_NAMES[int(m)]} {y}"


def _month_short(month_str: str) -> str:
    """'2026-04' -> 'Apr'"""
    return MONTH_NAMES[int(month_str.split("-")[1])]


def _derive_asset_class(exchange: str) -> str:
    mapping = {"NSE": "cash", "NFO": "fno", "MCX": "commodity"}
    return mapping.get(exchange.upper(), "cash")


def _extract_underlying(symbol: str, exchange: str = "NSE") -> str:
    if exchange == "NSE":
        return symbol
    m = FNO_UNDERLYING.match(symbol)
    if m:
        return m.group(1)
    # Try splitting on space or digits
    parts = re.split(r"[\s\d]", symbol, maxsplit=1)
    return parts[0] if parts else symbol


def _dec(v) -> float:
    """Decimal/None → float."""
    if v is None:
        return 0.0
    return float(v)


def _month_range(month_str: str) -> tuple[date, date]:
    """Return (first_day, last_day) for a YYYY-MM string."""
    y, m = int(month_str[:4]), int(month_str[5:7])
    first = date(y, m, 1)
    if m == 12:
        last = date(y + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(y, m + 1, 1) - timedelta(days=1)
    return first, last


def _trading_days_in_month(first: date, last: date) -> int:
    """Approximate trading days (weekdays) in a month."""
    count = 0
    d = first
    while d <= last:
        if d.weekday() < 5:
            count += 1
        d += timedelta(days=1)
    return max(count, 1)


# ══════════════════════════════════════════════════════════════════════
# Core build function
# ══════════════════════════════════════════════════════════════════════

def build_monthly_report(
    tenant,
    portfolio,
    month: Optional[str] = None,
    force: bool = False,
) -> dict:
    """
    Build the full monthly feedback report.

    Args:
        tenant: Tenant object from request
        portfolio: Portfolio object
        month: Optional specific month "YYYY-MM" (default: current)
        force: Bypass cache

    Returns:
        dict matching MonthlyReport shape
    """
    cache_key = f"monthly_report:{tenant.id}:{portfolio.id}:{month or 'all'}"
    if not force:
        cached = cache.get(cache_key)
        if cached:
            return cached

    now = timezone.now()
    current_month = now.strftime("%Y-%m")

    # ── 1. Build month groups from Position data ──
    months_data = _build_month_groups(tenant, portfolio, month, current_month)

    # ── 2. Build YTD summary ──
    ytd = _build_ytd(months_data, portfolio)

    # ── 3-5. Feedback sections ──
    # When viewing a specific month, scope to that month.
    # When viewing all (default), use the most recent month that has data.
    if month:
        target_month = month
    elif months_data:
        target_month = months_data[0].month  # most recent with data
    else:
        target_month = current_month

    capture_matrix = _build_capture_matrix(target_month)
    signal_audit = _build_signal_audit(target_month)
    rejections = _build_rejections(target_month)

    # ── 6. Equity curve + drawdown ──
    equity_curve = _build_equity_curve(target_month)

    # ── 7. Analytics (time-of-day, day-of-week, sector) ──
    analytics = _build_analytics(target_month)

    # ── 8. Benchmark comparison ──
    capital_base = _dec(portfolio.capital) if portfolio else 500_000
    month_pnl = months_data[0].total_pnl if months_data else 0
    benchmark = _build_benchmark(target_month, capital_base, month_pnl)

    # ── 9. Lessons ──
    lessons = _build_lessons(months_data, capture_matrix, signal_audit, rejections)

    result = {
        "paper_mode": portfolio.mode == "paper",
        "current_month": current_month,
        "generated_at": now.isoformat(),
        "ytd": asdict(ytd),
        "months": [_month_group_to_dict(m) for m in months_data],
        "capture_matrix": [asdict(c) for c in capture_matrix],
        "signal_audit": asdict(signal_audit),
        "rejections": [asdict(r) for r in rejections],
        "lessons": lessons,
        "equity_curve": asdict(equity_curve),
        "analytics": asdict(analytics),
        "benchmark": asdict(benchmark),
    }

    cache.set(cache_key, result, CACHE_TTL)
    return result


def _month_group_to_dict(mg: MonthGroup) -> dict:
    """Serialize a MonthGroup including nested UnderlyingRoll objects."""
    d = asdict(mg)
    # by_asset_class contains UnderlyingRoll dataclass instances in a plain dict
    d["by_asset_class"] = {
        ac: [asdict(roll) for roll in rolls]
        for ac, rolls in mg.by_asset_class.items()
    }
    return d


# ══════════════════════════════════════════════════════════════════════
# Section builders
# ══════════════════════════════════════════════════════════════════════

def _build_month_groups(tenant, portfolio, month, current_month) -> list[MonthGroup]:
    """Build MonthGroup objects from Position data."""
    from apps.portfolio.models import Position

    positions = Position.objects.filter(
        tenant=tenant, portfolio=portfolio,
    ).order_by("-opened_at")

    # Group positions by month
    month_buckets = defaultdict(list)
    for pos in positions:
        # Use closed_at month for closed positions, opened_at for open
        if pos.status == "closed" and pos.closed_at:
            m = pos.closed_at.strftime("%Y-%m")
        else:
            m = pos.opened_at.strftime("%Y-%m")
        month_buckets[m].append(pos)

    if month:
        # Filter to specific month
        month_buckets = {k: v for k, v in month_buckets.items() if k == month}

    # Also check legacy TradeJournal if no positions exist
    if not month_buckets:
        month_buckets = _legacy_month_groups(month, current_month)

    months = []
    for m_key in sorted(month_buckets.keys(), reverse=True)[:12]:
        positions_in_month = month_buckets[m_key]
        mg = _positions_to_month_group(m_key, positions_in_month)
        months.append(mg)

    # If still empty, return a single empty current month
    if not months:
        months = [MonthGroup(
            month=current_month,
            month_label=_month_label(current_month),
            total_pnl=0, realized_pnl=0, unrealized_pnl=0,
            capital_deployed=0, exposure=0, trade_count=0, win_rate=0,
            by_asset_class={"cash": [], "fno": [], "commodity": []},
        )]

    return months


def _legacy_month_groups(month, current_month) -> dict:
    """Fallback: build month groups from legacy TradeJournal."""
    try:
        from trading.models import TradeJournal
        # Include FILLED, PAPER, EXECUTED, CANCELLED (all have P&L data).
        # Exclude REJECTED (blocked signals — tracked in rejections section).
        # Exclude PENDING, PLANNED, APPROVED (pre-execution, no outcome).
        qs = TradeJournal.objects.filter(
            status__in=["FILLED", "PAPER", "EXECUTED", "CANCELLED"]
        )
        if month:
            y, m = int(month[:4]), int(month[5:7])
            first, last = _month_range(month)
            qs = qs.filter(trade_date__gte=first, trade_date__lte=last)

        buckets = defaultdict(list)
        for tj in qs:
            m_key = tj.trade_date.strftime("%Y-%m")
            buckets[m_key].append(tj)
        return buckets
    except Exception:
        return {}


def _positions_to_month_group(month_key: str, positions) -> MonthGroup:
    """Convert a list of Position objects (or TradeJournal) to MonthGroup."""
    first_day, last_day = _month_range(month_key)
    trading_days = _trading_days_in_month(first_day, last_day)
    now = date.today()

    by_asset = {"cash": defaultdict(list), "fno": defaultdict(list), "commodity": defaultdict(list)}

    for pos in positions:
        # Handle both Position (backend) and TradeJournal (legacy)
        if hasattr(pos, "exchange"):
            ac = _derive_asset_class(pos.exchange)
            underlying = _extract_underlying(pos.symbol, pos.exchange)
            leg = _position_to_leg(pos)
        else:
            # TradeJournal
            ac = "cash"  # legacy doesn't have exchange field
            underlying = pos.symbol
            leg = _tradejournal_to_leg(pos)

        by_asset[ac][underlying].append(leg)

    by_asset_class = {"cash": [], "fno": [], "commodity": []}
    total_pnl = 0
    total_realized = 0
    total_unrealized = 0
    total_capital = 0
    total_exposure = 0
    total_trades = 0
    total_wins = 0
    total_losses = 0

    for ac in ("cash", "fno", "commodity"):
        for underlying, legs in by_asset[ac].items():
            roll = _legs_to_roll(underlying, ac, legs, first_day, last_day, trading_days)
            by_asset_class[ac].append(roll)
            total_pnl += roll.running_pnl
            total_realized += roll.realized_pnl
            total_unrealized += roll.unrealized_pnl
            total_capital += roll.capital_deployed
            total_exposure += roll.exposure
            total_trades += roll.trade_count
            total_wins += roll.winning_trades
            total_losses += roll.losing_trades

    wr = total_wins / max(1, total_wins + total_losses)

    return MonthGroup(
        month=month_key,
        month_label=_month_label(month_key),
        total_pnl=round(total_pnl, 2),
        realized_pnl=round(total_realized, 2),
        unrealized_pnl=round(total_unrealized, 2),
        capital_deployed=round(total_capital, 2),
        exposure=round(total_exposure, 2),
        trade_count=total_trades,
        win_rate=round(wr, 3),
        by_asset_class=by_asset_class,
    )


def _position_to_leg(pos) -> PositionLeg:
    """Convert a backend Position to PositionLeg."""
    closed = pos.status == "closed"
    pnl = _dec(pos.realized_pnl) if closed else _dec(pos.unrealized_pnl)
    return PositionLeg(
        id=str(pos.id),
        symbol=pos.symbol,
        side=pos.side,
        quantity=pos.qty,
        entry_price=_dec(pos.avg_price),
        exit_price=_dec(pos.exit_price) if pos.exit_price else None,
        entry_date=pos.opened_at.strftime("%Y-%m-%d"),
        exit_date=pos.closed_at.strftime("%Y-%m-%d") if pos.closed_at else None,
        target_price=_dec(pos.tp) if pos.tp else None,
        stop_price=_dec(pos.sl) if pos.sl else None,
        pnl=round(pnl, 2),
        status="CLOSED" if closed else "OPEN",
    )


def _tradejournal_to_leg(tj) -> PositionLeg:
    """Convert a legacy TradeJournal to PositionLeg."""
    closed = tj.status in ("FILLED", "PAPER", "CANCELLED")
    pnl = tj.pnl or 0.0
    return PositionLeg(
        id=str(tj.id),
        symbol=tj.symbol,
        side=tj.side,
        quantity=tj.quantity,
        entry_price=tj.entry_price,
        exit_price=tj.fill_price,
        entry_date=tj.trade_date.strftime("%Y-%m-%d"),
        exit_date=tj.trade_date.strftime("%Y-%m-%d") if closed else None,
        target_price=tj.target,
        stop_price=tj.stop_loss,
        pnl=round(pnl, 2),
        status="CLOSED" if closed else "OPEN",
        notes=tj.reasoning[:100] if tj.reasoning else "",
    )


def _legs_to_roll(
    underlying: str, ac: str, legs: list[PositionLeg],
    first_day: date, last_day: date, trading_days: int,
) -> UnderlyingRoll:
    """Roll up legs into an UnderlyingRoll."""
    realized = sum(l.pnl for l in legs if l.status == "CLOSED")
    unrealized = sum(l.pnl for l in legs if l.status == "OPEN")
    exposure = sum(l.entry_price * l.quantity for l in legs)
    margin = 0.2 if ac != "cash" else 1.0
    capital = exposure * margin

    target_total = sum(
        max(0, (l.target_price - l.entry_price) * (1 if l.side == "BUY" else -1)) * l.quantity
        for l in legs if l.target_price
    )
    risk_total = sum(
        max(0, (l.entry_price - l.stop_price) * (1 if l.side == "BUY" else -1)) * l.quantity
        for l in legs if l.stop_price
    )

    closed = [l for l in legs if l.status == "CLOSED"]
    winners = [l for l in closed if l.pnl > 0]
    losers = [l for l in closed if l.pnl < 0]

    # Avg R:R from target/SL prices (not realized P&L)
    rr_values = []
    for l in legs:
        if l.stop_price and l.target_price and l.entry_price:
            risk = abs(l.entry_price - l.stop_price)
            reward = abs(l.target_price - l.entry_price)
            if risk > 0:
                rr_values.append(reward / risk)
    avg_rr = sum(rr_values) / max(1, len(rr_values)) if rr_values else 0

    # Days in position (approximate)
    days = 0
    for l in legs:
        start = datetime.strptime(l.entry_date, "%Y-%m-%d").date()
        end = datetime.strptime(l.exit_date, "%Y-%m-%d").date() if l.exit_date else date.today()
        days += (end - start).days + 1

    return UnderlyingRoll(
        underlying=underlying,
        asset_class=ac,
        capital_deployed=round(capital, 2),
        exposure=round(exposure, 2),
        running_pnl=round(realized + unrealized, 2),
        realized_pnl=round(realized, 2),
        unrealized_pnl=round(unrealized, 2),
        target_total=round(target_total, 2),
        risk_total=round(risk_total, 2),
        trade_count=len(legs),
        winning_trades=len(winners),
        losing_trades=len(losers),
        avg_rr=round(avg_rr, 2),
        days_in_position=days,
        pct_of_month=round(min(1.0, days / trading_days), 2),
        legs=[asdict(l) for l in legs],
    )


def _build_ytd(months: list[MonthGroup], portfolio) -> YtdSummary:
    """Build YTD summary from month groups."""
    capital_base = _dec(portfolio.capital) if portfolio else 500_000

    total_pnl = sum(m.total_pnl for m in months)
    trade_count = sum(m.trade_count for m in months)

    # Weighted win rate
    if trade_count > 0:
        win_rate = sum(m.win_rate * m.trade_count for m in months) / trade_count
    else:
        win_rate = 0

    best = max(months, key=lambda m: m.total_pnl) if months else None
    worst = min(months, key=lambda m: m.total_pnl) if months else None

    bars = [
        YtdMonthBar(
            month=m.month,
            month_label=_month_short(m.month),
            pnl=m.total_pnl,
        )
        for m in reversed(months[:12])
    ]

    return YtdSummary(
        capital_base=capital_base,
        total_pnl=round(total_pnl, 2),
        best_month=best.month if best else "",
        worst_month=worst.month if worst else "",
        win_rate=round(win_rate, 3),
        trade_count=trade_count,
        months=bars,
    )


def _build_capture_matrix(month: str) -> list[StockCapture]:
    """Build per-stock capture rate from SignalLog + TradeJournal."""
    try:
        from trading.models import SignalLog, TradeJournal
    except Exception:
        return []

    first, last = _month_range(month)

    # Get all signals for this month
    signals = list(SignalLog.objects.filter(
        signal_date__gte=first, signal_date__lte=last,
    ))

    # Get all trades for this month (include CANCELLED — they have real P&L)
    trades = list(TradeJournal.objects.filter(
        trade_date__gte=first, trade_date__lte=last,
        status__in=["FILLED", "PAPER", "EXECUTED", "CANCELLED"],
    ))

    # Group by symbol
    sym_signals = defaultdict(list)
    sym_trades = defaultdict(list)
    all_symbols = set()

    for s in signals:
        sym_signals[s.symbol].append(s)
        all_symbols.add(s.symbol)
    for t in trades:
        sym_trades[t.symbol].append(t)
        all_symbols.add(t.symbol)

    matrix = []
    for symbol in sorted(all_symbols):
        sigs = sym_signals.get(symbol, [])
        tds = sym_trades.get(symbol, [])

        traded = [s for s in sigs if s.outcome == "TRADED"]
        skipped = [s for s in sigs if s.outcome in ("REJECTED", "SKIPPED", "EXPIRED")]

        captured_pnl = sum(t.pnl or 0 for t in tds)

        # Potential P&L from max favorable moves
        potential_pnl = sum(
            (s.max_favorable_move or 0) for s in sigs
            if s.max_favorable_move is not None
        )

        # Month move % — use first/last signal eod prices as proxy
        enriched = [s for s in sigs if s.eod_price and s.entry_price > 0]
        if enriched:
            first_entry = min(enriched, key=lambda s: s.signal_date).entry_price
            last_eod = max(enriched, key=lambda s: s.signal_date).eod_price
            move_pct = ((last_eod - first_entry) / first_entry) * 100
        else:
            move_pct = 0

        capture_rate = (captured_pnl / potential_pnl * 100) if potential_pnl > 0 else 0

        # Best signal (highest R:R)
        best = max(sigs, key=lambda s: s.risk_reward, default=None)
        worst_miss = max(
            [s for s in sigs if s.outcome != "TRADED" and (s.max_favorable_move or 0) > 0],
            key=lambda s: s.max_favorable_move or 0,
            default=None,
        )

        matrix.append(StockCapture(
            symbol=symbol,
            month_move_pct=round(move_pct, 2),
            signals_fired=len(sigs),
            trades_taken=max(len(traded), len(tds)),
            trades_skipped=len(skipped),
            captured_pnl=round(captured_pnl, 2),
            potential_pnl=round(potential_pnl, 2),
            capture_rate_pct=round(capture_rate, 2),
            best_signal={
                "strategy": best.strategy,
                "rr": best.risk_reward,
                "outcome": best.outcome,
            } if best else None,
            worst_miss={
                "strategy": worst_miss.strategy,
                "rr": worst_miss.risk_reward,
                "potential_pnl": round(worst_miss.max_favorable_move or 0, 2),
            } if worst_miss else None,
        ))

    # Sort by potential P&L descending
    matrix.sort(key=lambda x: x.potential_pnl, reverse=True)
    return matrix


def _build_signal_audit(month: str) -> SignalAudit:
    """Build signal outcome breakdown."""
    try:
        from trading.models import SignalLog
    except Exception:
        return SignalAudit(
            total_signals=0, by_outcome={}, by_source={},
            by_strategy={}, profitable_if_taken=0, loss_avoided=0,
        )

    first, last = _month_range(month)
    signals = list(SignalLog.objects.filter(
        signal_date__gte=first, signal_date__lte=last,
    ))

    by_outcome = defaultdict(int)
    by_source = defaultdict(int)
    by_strategy = defaultdict(lambda: {"count": 0, "wins": 0, "rr_sum": 0})

    profitable_if_taken = 0
    loss_avoided = 0

    for s in signals:
        by_outcome[s.outcome] += 1
        by_source[s.source] += 1
        by_strategy[s.strategy]["count"] += 1

        # Was it profitable if the max favorable move exceeded risk?
        if s.max_favorable_move is not None:
            risk = abs(s.entry_price - s.stoploss)
            hit_target = s.max_favorable_move >= abs(s.target - s.entry_price) * 0.8
            hit_sl = (s.max_adverse_move or 0) >= risk

            if s.outcome == "TRADED":
                if s.trade_journal and (s.trade_journal.pnl or 0) > 0:
                    by_strategy[s.strategy]["wins"] += 1
                by_strategy[s.strategy]["rr_sum"] += s.risk_reward
            elif s.outcome in ("REJECTED", "SKIPPED", "EXPIRED"):
                if hit_target and not hit_sl:
                    profitable_if_taken += 1
                elif hit_sl and not hit_target:
                    loss_avoided += 1

    # Finalize strategy stats
    strat_perf = {}
    for strat, stats in by_strategy.items():
        c = stats["count"]
        strat_perf[strat] = {
            "count": c,
            "win_rate": round(stats["wins"] / max(1, c), 3),
            "avg_rr": round(stats["rr_sum"] / max(1, c), 2),
        }

    return SignalAudit(
        total_signals=len(signals),
        by_outcome=dict(by_outcome),
        by_source=dict(by_source),
        by_strategy=strat_perf,
        profitable_if_taken=profitable_if_taken,
        loss_avoided=loss_avoided,
    )


def _build_rejections(month: str) -> list[RejectionReview]:
    """Build risk rejection reviews with hindsight profitability.

    Primary source: TradeJournal with status=REJECTED (most reliable).
    Secondary source: AuditLog with event_type=RISK_REJECT (older straddle data).
    """
    try:
        from trading.models import AuditLog, SignalLog, TradeJournal
    except Exception:
        return []

    first, last = _month_range(month)
    seen = set()  # (symbol, date) dedup
    reviews = []

    # ── Primary: TradeJournal REJECTED (equity screening rejections) ──
    rejected_trades = TradeJournal.objects.filter(
        status="REJECTED",
        trade_date__gte=first,
        trade_date__lte=last,
    )
    for tj in rejected_trades:
        key = (tj.symbol, tj.trade_date.isoformat(), tj.risk_reason)
        if key in seen:
            continue
        seen.add(key)

        # Check SignalLog for post-hoc price data
        sig = SignalLog.objects.filter(
            symbol=tj.symbol,
            signal_date=tj.trade_date,
            max_favorable_move__isnull=False,
        ).first()

        would_profit = False
        hypo_pnl = 0.0
        if sig and sig.max_favorable_move is not None:
            risk = abs(sig.entry_price - sig.stoploss)
            would_profit = sig.max_favorable_move >= risk
            hypo_pnl = sig.max_favorable_move

        reviews.append(RejectionReview(
            symbol=tj.symbol,
            date=tj.trade_date.isoformat(),
            reason=tj.risk_reason or "Unknown",
            would_have_profited=would_profit,
            hypothetical_pnl=round(hypo_pnl, 2),
        ))

    # ── Secondary: AuditLog RISK_REJECT (straddle/options rejections) ──
    audit_rejects = AuditLog.objects.filter(
        event_type="RISK_REJECT",
        created_at__date__gte=first,
        created_at__date__lte=last,
    )
    for r in audit_rejects:
        key = (r.symbol, r.created_at.date().isoformat(),
               (r.risk_details or {}).get("reason", ""))
        if key in seen:
            continue
        seen.add(key)

        reason = (r.risk_details or {}).get("reason", "")
        if not reason and r.risk_details:
            reason = (r.risk_details or {}).get("rule", "")
        reviews.append(RejectionReview(
            symbol=r.symbol,
            date=r.created_at.strftime("%Y-%m-%d"),
            reason=reason or "Unknown",
            would_have_profited=False,
            hypothetical_pnl=0.0,
        ))

    return reviews


# ══════════════════════════════════════════════════════════════════════
# Equity curve + drawdown
# ══════════════════════════════════════════════════════════════════════

def _build_equity_curve(month: str) -> EquityCurve:
    """Build day-by-day equity curve with drawdown from peak."""
    try:
        from trading.models import TradeJournal
    except Exception:
        return _empty_equity_curve()

    first, last = _month_range(month)
    trades = TradeJournal.objects.filter(
        trade_date__gte=first, trade_date__lte=last,
        status__in=["FILLED", "PAPER", "EXECUTED", "CANCELLED"],
    )

    daily = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
    for t in trades:
        d = t.trade_date.isoformat()
        daily[d]["pnl"] += t.pnl or 0
        daily[d]["trades"] += 1
        if (t.pnl or 0) > 0:
            daily[d]["wins"] += 1

    if not daily:
        return _empty_equity_curve()

    points = []
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    max_dd_date = ""

    for d in sorted(daily.keys()):
        day = daily[d]
        cumulative += day["pnl"]
        peak = max(peak, cumulative)
        dd = cumulative - peak  # negative or zero
        if dd < max_dd:
            max_dd = dd
            max_dd_date = d
        points.append(EquityCurvePoint(
            date=d,
            pnl=round(day["pnl"], 2),
            cumulative=round(cumulative, 2),
            trades=day["trades"],
            drawdown=round(dd, 2),
        ))

    return EquityCurve(
        points=points,
        max_drawdown=round(max_dd, 2),
        max_drawdown_date=max_dd_date,
        peak_equity=round(peak, 2),
        final_equity=round(cumulative, 2),
    )


def _empty_equity_curve() -> EquityCurve:
    return EquityCurve(
        points=[], max_drawdown=0, max_drawdown_date="",
        peak_equity=0, final_equity=0,
    )


# ══════════════════════════════════════════════════════════════════════
# Analytics — time-of-day, day-of-week, sector
# ══════════════════════════════════════════════════════════════════════

IST_OFFSET = 5.5  # UTC+5:30

def _build_analytics(month: str) -> Analytics:
    """Break down trades by hour (IST), day-of-week, and sector."""
    try:
        from trading.models import TradeJournal
    except Exception:
        return Analytics(by_hour=[], by_day_of_week=[], by_sector=[])

    first, last = _month_range(month)
    trades = list(TradeJournal.objects.filter(
        trade_date__gte=first, trade_date__lte=last,
        status__in=["FILLED", "PAPER", "EXECUTED", "CANCELLED"],
    ))

    # ── By hour (IST) ──
    hour_data = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0})
    for t in trades:
        # Convert UTC created_at to IST hour
        ist_hour = (t.created_at.hour + int(IST_OFFSET)) % 24
        # Clamp to market hours 9-15
        if ist_hour < 9:
            ist_hour = 9
        elif ist_hour > 15:
            ist_hour = 15
        bucket = hour_data[ist_hour]
        bucket["trades"] += 1
        pnl = t.pnl or 0
        bucket["pnl"] += pnl
        if pnl > 0:
            bucket["wins"] += 1
        elif pnl < 0:
            bucket["losses"] += 1

    by_hour = []
    for h in range(9, 16):
        d = hour_data[h]
        total = d["wins"] + d["losses"]
        by_hour.append(HourBucket(
            hour=h, label=f"{h:02d}:00",
            trades=d["trades"], wins=d["wins"], losses=d["losses"],
            pnl=round(d["pnl"], 2),
            win_rate=round(d["wins"] / max(1, total), 3),
        ))

    # ── By day of week ──
    dow_data = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        dow = t.trade_date.weekday()  # 0=Mon
        dow_data[dow]["trades"] += 1
        pnl = t.pnl or 0
        dow_data[dow]["pnl"] += pnl
        if pnl > 0:
            dow_data[dow]["wins"] += 1

    DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    by_dow = []
    for d in range(5):
        dd = dow_data[d]
        by_dow.append(DayOfWeekBucket(
            day=d, label=DOW_LABELS[d],
            trades=dd["trades"], wins=dd["wins"],
            pnl=round(dd["pnl"], 2),
            win_rate=round(dd["wins"] / max(1, dd["trades"]), 3),
        ))

    # ── By sector ──
    sector_data = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "symbols": set()})
    for t in trades:
        sector = _get_sector(t.symbol)
        sector_data[sector]["trades"] += 1
        sector_data[sector]["symbols"].add(t.symbol)
        pnl = t.pnl or 0
        sector_data[sector]["pnl"] += pnl
        if pnl > 0:
            sector_data[sector]["wins"] += 1

    by_sector = []
    for sector, sd in sorted(sector_data.items(), key=lambda x: -abs(x[1]["pnl"])):
        by_sector.append(SectorBucket(
            sector=sector,
            trades=sd["trades"],
            pnl=round(sd["pnl"], 2),
            win_rate=round(sd["wins"] / max(1, sd["trades"]), 3),
            symbols=sorted(sd["symbols"]),
        ))

    return Analytics(by_hour=by_hour, by_day_of_week=by_dow, by_sector=by_sector)


# ══════════════════════════════════════════════════════════════════════
# Benchmark comparison
# ══════════════════════════════════════════════════════════════════════

def _build_benchmark(month: str, capital_base: float, portfolio_pnl: float) -> BenchmarkComparison:
    """Compare portfolio return to NIFTY50 for the month.

    Tries to fetch NIFTY data from Angel One via the legacy BrokerClient.
    Falls back to 0 if data unavailable.
    """
    first, last = _month_range(month)
    trading_days = _trading_days_in_month(first, last)
    port_return = (portfolio_pnl / capital_base * 100) if capital_base > 0 else 0

    nifty_start = 0.0
    nifty_end = 0.0
    nifty_return = 0.0

    try:
        from trading.services.data_service import BrokerClient
        broker = BrokerClient.get_instance()
        broker.ensure_login()
        # NIFTY token = 99926000
        candles = broker.fetch_candles(
            "99926000",
            f"{first.isoformat()} 09:15",
            f"{last.isoformat()} 15:30",
            interval="ONE_DAY",
            exchange="NSE",
        )
        if candles and len(candles) >= 2:
            # candles: [[ts, o, h, l, c, v], ...]
            first_candle = candles[0]
            last_candle = candles[-1]
            nifty_start = float(first_candle[1] if isinstance(first_candle, list) else first_candle.get("open", 0))
            nifty_end = float(last_candle[4] if isinstance(last_candle, list) else last_candle.get("close", 0))
            if nifty_start > 0:
                nifty_return = ((nifty_end - nifty_start) / nifty_start) * 100
    except Exception:
        pass  # Benchmark unavailable — not critical

    return BenchmarkComparison(
        portfolio_return_pct=round(port_return, 2),
        nifty_return_pct=round(nifty_return, 2),
        alpha_pct=round(port_return - nifty_return, 2),
        trading_days=trading_days,
        nifty_start=round(nifty_start, 2),
        nifty_end=round(nifty_end, 2),
    )


def _build_lessons(
    months: list[MonthGroup],
    capture_matrix: list[StockCapture],
    signal_audit: SignalAudit,
    rejections: list[RejectionReview],
) -> list[str]:
    """
    Generate rule-based lessons from the month's data.
    A future version can use Claude for deeper insights.
    """
    lessons = []

    # 1. Capture rate insight
    if capture_matrix:
        avg_capture = sum(s.capture_rate_pct for s in capture_matrix) / len(capture_matrix)
        if avg_capture < 30:
            lessons.append(
                f"Average capture rate is {avg_capture:.0f}% — most signal potential is being left "
                f"on the table. Consider lowering confidence thresholds or deploying more capital."
            )
        elif avg_capture > 60:
            lessons.append(
                f"Capture rate of {avg_capture:.0f}% is strong — execution is keeping pace with signals."
            )

    # 2. Signal conversion
    if signal_audit.total_signals > 0:
        traded = signal_audit.by_outcome.get("TRADED", 0)
        conversion = traded / signal_audit.total_signals * 100
        if conversion < 20:
            lessons.append(
                f"Only {conversion:.0f}% of signals converted to trades. "
                f"{signal_audit.by_outcome.get('REJECTED', 0)} blocked by risk, "
                f"{signal_audit.by_outcome.get('EXPIRED', 0)} expired. "
                f"Review if risk gates are too tight."
            )

    # 3. Profitable rejections
    if signal_audit.profitable_if_taken > 0:
        total_rej = signal_audit.by_outcome.get("REJECTED", 0) + signal_audit.by_outcome.get("SKIPPED", 0)
        if total_rej > 0:
            profitable_pct = signal_audit.profitable_if_taken / total_rej * 100
            lessons.append(
                f"{signal_audit.profitable_if_taken} of {total_rej} "
                f"skipped/rejected signals ({profitable_pct:.0f}%) would have been profitable. "
                f"Review @RiskGuard gate calibration."
            )

    # 4. Best/worst strategy
    if signal_audit.by_strategy:
        best_strat = max(
            signal_audit.by_strategy.items(),
            key=lambda kv: kv[1]["win_rate"],
        )
        worst_strat = min(
            signal_audit.by_strategy.items(),
            key=lambda kv: kv[1]["win_rate"],
        )
        if best_strat[1]["count"] >= 3:
            lessons.append(
                f"Best strategy: {best_strat[0]} — {best_strat[1]['win_rate']:.0%} win rate "
                f"across {best_strat[1]['count']} signals."
            )
        if worst_strat[0] != best_strat[0] and worst_strat[1]["count"] >= 3:
            lessons.append(
                f"Weakest strategy: {worst_strat[0]} — {worst_strat[1]['win_rate']:.0%} win rate. "
                f"Consider reducing allocation or disabling."
            )

    # 5. Win rate
    if months:
        current = months[0]
        if current.win_rate < 0.4 and current.trade_count >= 5:
            lessons.append(
                f"Win rate this month is {current.win_rate:.0%} across {current.trade_count} trades — "
                f"below 40%. Review entry criteria quality."
            )

    if not lessons:
        lessons.append("Insufficient data for actionable lessons. More trading days needed.")

    return lessons
