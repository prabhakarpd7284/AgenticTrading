"""Signals → trade paths → portfolio simulation → metrics.

Two phases, so memory stays flat (one symbol's candles at a time):

Phase 1 — ``resolve_symbol(data, intents, cfg)`` (per symbol, qty-free)
    Each ``Intent`` becomes a ``Candidate`` path using only that symbol's 1m
    bars: entry at the OPEN of the first 1m bar with ``t >= intent.signal_t``
    (the moment the signal is known = close of the signal bar), slippage
    against us; stop/target checked intrabar on every 1m bar (entry bar
    included); a bar touching both → STOP first; a bar opening through the
    stop fills at its open (worse), through the target at its open (better);
    owner trail (breakeven at +1R, then 1.5×ATR of the trading timeframe,
    ratchet only) updated AFTER each bar using that bar's extreme and the ATR
    published at the bar's close, effective from the next bar; optional time
    exit at the open of the first bar with ``t >= time_exit_t``; otherwise
    flat at the close of the last bar ≤ 15:14. Exit price and time do not
    depend on position size, which is why this can run before sizing.

Phase 2 — ``PortfolioSimulator(cfg, is_trading_day).run(candidates, sessions)``
    Walks candidates in entry-time order across all symbols and applies:
    cooldown state machine (3 loss days → 2 trading days no entries, logged as
    ``shadow`` rows → half risk until the first profitable day); one position
    per symbol; RiskEngine max open positions; RiskEngine daily-loss gate
    (realized net loss so far ≥ 3 % → no entries; loss + new trade risk > 4.5 %
    → reject); sizing = floor(capital × risk% × multiplier / per-share risk),
    capped by the RiskEngine position cap (% of capital notional) and the
    RiskEngine max-risk-per-trade; NSE costs on actual fills. Capital is fixed
    (no compounding). A position frees its slot for entries strictly after
    the bar it exited in.
"""
from __future__ import annotations

import bisect
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date

from plugins.strategy_pivot_boss.bars import Bar, hhmm, ist_minute

from .costs import CostModel, slip
from .data import SymbolData, day_of, resample
from .rules import COOLDOWN, CooldownTracker, live_window_atr_series, trail_next_stop

# ── config ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SimConfig:
    label: str = "base"
    capital: float = 500_000.0
    risk_pct: float = 0.5                 # % of capital risked per trade (before multiplier)
    # RiskEngine (settings.ALPHADESK)
    max_position_pct: float = 10.0        # notional cap per position, % of capital
    max_open_positions: int = 3
    max_risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 3.0
    min_rr: float | None = None           # RiskEngine MIN_RISK_REWARD_RATIO; None = not enforced
    # Live RiskEngine criterion 7 + _check_fields, exactly: an intent with no
    # target is rejected ("Prices must be positive": target <= 0), and
    # round(reward/risk, 2) < min_rr (default settings MIN_RISK_REWARD_RATIO=1.5
    # when min_rr is None) is rejected. Off by default (research exits often
    # have no target); turn on to see what @RiskGuard would actually let through.
    live_rr_gate: bool = False
    # execution
    slippage_pct: float = 0.02            # per side
    last_entry_min: int = 14 * 60 + 30    # no entries on a bar opening after 14:30 IST
    one_position_per_symbol: bool = True
    # owner rules (TRADING_RULES.md)
    trail: bool = True
    trail_breakeven_r: float = 1.0
    trail_atr_mult: float = 1.5
    trail_atr_period: int = 14
    cooldown: bool = True
    cooldown_loss_days: int = 3
    cooldown_trading_days: int = 2
    reduced_risk_mult: float = 0.5
    costs: CostModel = field(default_factory=CostModel)

    @classmethod
    def from_settings(cls, **overrides) -> SimConfig:
        """Limits from settings.ALPHADESK / TRADING_RULES (the live RiskEngine's)."""
        from django.conf import settings
        a = getattr(settings, "ALPHADESK", {})
        r = getattr(settings, "TRADING_RULES", {})
        base = dict(
            max_position_pct=float(a.get("MAX_POSITION_SIZE_PCT", 10.0)),
            max_open_positions=int(a.get("MAX_OPEN_POSITIONS", 3)),
            max_risk_per_trade_pct=float(a.get("MAX_RISK_PER_TRADE_PCT", 1.0)),
            max_daily_loss_pct=float(a.get("MAX_DAILY_LOSS_PCT", 3.0)),
            trail_breakeven_r=float(r.get("TRAIL_BREAKEVEN_R", 1.0)),
            trail_atr_mult=float(r.get("TRAIL_ATR_MULT", 1.5)),
            trail_atr_period=int(r.get("TRAIL_ATR_PERIOD", 14)),
            cooldown_loss_days=int(r.get("COOLDOWN_LOSS_DAYS", 3)),
            cooldown_trading_days=int(r.get("COOLDOWN_TRADING_DAYS", 2)),
            reduced_risk_mult=float(r.get("REDUCED_RISK_MULTIPLIER", 0.5)),
        )
        base.update(overrides)
        return cls(**base)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["costs"] = asdict(self.costs)
        return d


# ── intents and candidates ──────────────────────────────────────────────────


@dataclass
class Intent:
    """What a strategy emits. Prices are absolute; ``side`` +1 long / −1 short."""
    strategy: str
    symbol: str
    day: date
    side: int
    signal_t: int                 # epoch s when the signal is KNOWN (close of signal bar)
    stop: float                   # initial stop (defines 1R)
    target: float | None = None   # hard target (None = trail / time / EOD only)
    trail: bool = True            # apply the owner trail (if SimConfig.trail)
    trail_tf: int = 5             # trading timeframe (minutes) for the trail ATR
    time_exit_t: int | None = None
    priority: float = 0.0         # tie-break when several entries share a bar (higher first)
    meta: dict = field(default_factory=dict)


@dataclass
class Candidate:
    intent: Intent
    entry_t: int
    entry_raw: float
    entry_fill: float
    exit_t: int                   # open time of the 1m bar in which the exit happened
    exit_raw: float
    exit_fill: float
    reason: str                   # stop | breakeven | trail | target | time | eod
    risk_per_share: float         # |entry_fill − initial stop|
    mfe_r: float                  # best favourable excursion (bar extremes) in R
    mae_r: float
    trail_moves: int
    final_stop: float


def _atr_lookup(data: SymbolData, tf: int, period: int) -> Callable[[int], float | None]:
    """ATR(period) of the ``tf``-minute series (continuous over loaded sessions),
    published at each bar's scheduled close (start + tf min)."""
    bars_tf: list[Bar] = []
    for s in data.sessions:
        bars_tf.extend(resample(s.bars, tf))
    atrs = live_window_atr_series(bars_tf, period)
    pub = [b.t + tf * 60 for b in bars_tf]

    def at(t: int) -> float | None:
        k = bisect.bisect_right(pub, t) - 1
        return atrs[k] if k >= 0 else None
    return at


def resolve_symbol(data: SymbolData, intents: Iterable[Intent], cfg: SimConfig
                   ) -> tuple[list[Candidate], Counter]:
    """Phase 1 for one symbol. Returns (candidates, reject-reason counts)."""
    by_day = {s.day: s.bars for s in data.sessions if s.in_period}
    atr_fns: dict[int, Callable] = {}
    out: list[Candidate] = []
    rej: Counter = Counter()
    for it in intents:
        bars = by_day.get(it.day)
        if not bars:
            rej["no_session"] += 1
            continue
        fn = None
        if cfg.trail and it.trail:
            fn = atr_fns.get(it.trail_tf)
            if fn is None:
                fn = atr_fns[it.trail_tf] = _atr_lookup(data, it.trail_tf, cfg.trail_atr_period)
        c = resolve_intent(it, bars, cfg, fn)
        if isinstance(c, str):
            rej[c] += 1
        else:
            out.append(c)
    return out, rej


def resolve_intent(it: Intent, bars: Sequence[Bar], cfg: SimConfig,
                   atr_at: Callable[[int], float | None] | None) -> Candidate | str:
    """One intent through one day's 1m bars (≤ 15:14). Returns a Candidate or a
    reject reason string."""
    side = 1 if it.side > 0 else -1
    times = [b.t for b in bars]
    i0 = bisect.bisect_left(times, it.signal_t)
    if i0 >= len(bars):
        return "no_bar_after_signal"
    if ist_minute(bars[i0].t) > cfg.last_entry_min:
        return "after_last_entry"
    raw = bars[i0].o
    entry = slip(raw, side, cfg.slippage_pct, entering=True)
    stop0 = it.stop
    if side * (entry - stop0) <= 0:
        return "entry_beyond_stop"
    if it.target is not None and side * (it.target - entry) <= 0:
        return "entry_beyond_target"
    risk = abs(entry - stop0)
    stop = stop0
    best = entry
    worst = entry
    moves = 0
    use_trail = cfg.trail and it.trail and atr_at is not None
    exit_px = None
    reason = ""
    exit_i = len(bars) - 1
    for i in range(i0, len(bars)):
        b = bars[i]
        if it.time_exit_t is not None and i > i0 and b.t >= it.time_exit_t:
            exit_px, reason, exit_i = b.o, "time", i
            break
        if side > 0:
            if b.l <= stop:
                exit_px = min(b.o, stop)
            elif it.target is not None and b.h >= it.target:
                exit_px, reason = max(b.o, it.target), "target"
        else:
            if b.h >= stop:
                exit_px = max(b.o, stop)
            elif it.target is not None and b.l <= it.target:
                exit_px, reason = min(b.o, it.target), "target"
        if exit_px is not None:
            if not reason:
                reason = ("stop" if stop == stop0 else
                          "breakeven" if abs(stop - entry) < 1e-9 else "trail")
            exit_i = i
            break
        if side > 0:
            best, worst = max(best, b.h), min(worst, b.l)
        else:
            best, worst = min(best, b.l), max(worst, b.h)
        if use_trail:
            new = trail_next_stop(
                side=side, entry=entry, initial_stop=stop0, best=best,
                atr=atr_at(b.t + 60), current_stop=stop,
                breakeven_r=cfg.trail_breakeven_r, atr_mult=cfg.trail_atr_mult)
            if new != stop:
                moves += 1
                stop = new
    if exit_px is None:
        exit_px, reason, exit_i = bars[-1].c, "eod", len(bars) - 1
    # the exit bar counts toward excursions only up to the exit price
    if reason == "target":
        best = max(best, exit_px) if side > 0 else min(best, exit_px)
    elif reason in ("stop", "breakeven", "trail"):
        worst = min(worst, exit_px) if side > 0 else max(worst, exit_px)
    return Candidate(
        intent=it, entry_t=bars[i0].t, entry_raw=raw, entry_fill=entry,
        exit_t=bars[exit_i].t, exit_raw=exit_px,
        exit_fill=slip(exit_px, side, cfg.slippage_pct, entering=False),
        reason=reason, risk_per_share=risk,
        mfe_r=side * (best - entry) / risk, mae_r=side * (entry - worst) / risk,
        trail_moves=moves, final_stop=stop,
    )


# ── phase 2: portfolio ──────────────────────────────────────────────────────


@dataclass
class TradeRow:
    status: str                  # taken | shadow (cooldown) | skipped
    skip_reason: str
    strategy: str
    symbol: str
    day: str
    side: int
    signal_time: str
    entry_time: str
    exit_time: str
    entry_px: float
    exit_px: float
    stop: float
    target: float | None
    reason: str
    qty: int
    notional: float
    notional_capped: bool
    risk_mult: float
    risk_rs: float
    gross: float
    costs: float
    net: float
    r_net: float
    r_gross: float
    mfe_r: float
    mae_r: float
    trail_moves: int
    entry_t: int
    exit_t: int
    meta: str = ""


@dataclass
class SimResult:
    config: SimConfig
    rows: list[TradeRow]
    daily: list[dict]
    sessions: list[date]
    phase1_rejects: dict = field(default_factory=dict)

    @property
    def taken(self) -> list[TradeRow]:
        return [r for r in self.rows if r.status == "taken"]

    def summary(self) -> dict:
        return summarize(self.taken, self.sessions, self.rows, self.daily,
                         self.config, self.phase1_rejects)


class PortfolioSimulator:
    def __init__(self, cfg: SimConfig, is_trading_day: Callable[[date], bool]):
        self.cfg = cfg
        self.is_trading_day = is_trading_day

    def size(self, entry: float, risk_ps: float, mult: float) -> tuple[int, bool]:
        c = self.cfg
        if risk_ps <= 0 or entry <= 0 or mult <= 0:
            return 0, False
        q_risk = math.floor(c.capital * c.risk_pct * mult / 100.0 / risk_ps)
        q_rmax = math.floor(c.capital * c.max_risk_per_trade_pct * mult / 100.0 / risk_ps)
        q_cap = math.floor(c.capital * c.max_position_pct / 100.0 / entry)
        q = max(0, min(q_risk, q_rmax, q_cap))
        return q, q_cap < min(q_risk, q_rmax)

    def _row(self, cand: Candidate, status: str, skip: str, qty: int, capped: bool,
             mult: float) -> TradeRow:
        it, side = cand.intent, (1 if cand.intent.side > 0 else -1)
        gross = side * (cand.exit_fill - cand.entry_fill) * qty
        costs = self.cfg.costs.round_trip(cand.entry_fill, cand.exit_fill, qty, side) if qty else 0.0
        risk_rs = qty * cand.risk_per_share
        meta = ";".join(f"{k}={v}" for k, v in sorted(it.meta.items())) if it.meta else ""
        return TradeRow(
            status=status, skip_reason=skip, strategy=it.strategy, symbol=it.symbol,
            day=it.day.isoformat(), side=side, signal_time=hhmm(ist_minute(it.signal_t)),
            entry_time=hhmm(ist_minute(cand.entry_t)), exit_time=hhmm(ist_minute(cand.exit_t)),
            entry_px=round(cand.entry_fill, 4), exit_px=round(cand.exit_fill, 4),
            stop=round(it.stop, 4), target=None if it.target is None else round(it.target, 4),
            reason=cand.reason, qty=qty, notional=round(qty * cand.entry_fill, 2),
            notional_capped=capped, risk_mult=mult, risk_rs=round(risk_rs, 2),
            gross=round(gross, 2), costs=round(costs, 2), net=round(gross - costs, 2),
            r_net=round((gross - costs) / risk_rs, 4) if risk_rs else 0.0,
            r_gross=round(gross / risk_rs, 4) if risk_rs else 0.0,
            mfe_r=round(cand.mfe_r, 3), mae_r=round(cand.mae_r, 3),
            trail_moves=cand.trail_moves, entry_t=cand.entry_t, exit_t=cand.exit_t, meta=meta,
        )

    def run(self, candidates: Sequence[Candidate], sessions: Sequence[date],
            phase1_rejects: dict | None = None) -> SimResult:
        c = self.cfg
        cd = CooldownTracker(self.is_trading_day, c.cooldown_loss_days,
                             c.cooldown_trading_days, c.reduced_risk_mult)
        by_day: dict[date, list[Candidate]] = defaultdict(list)
        for cand in candidates:
            by_day[cand.intent.day].append(cand)
        days = sorted(set(sessions) | set(by_day))
        rows: list[TradeRow] = []
        daily: list[dict] = []
        cap = c.capital
        for d in days:
            status, mult = cd.status(d) if c.cooldown else ("normal", 1.0)
            todays = sorted(by_day.get(d, ()),
                            key=lambda x: (x.entry_t, -x.intent.priority, x.intent.symbol,
                                           x.intent.strategy))
            open_pos: list[tuple[int, str]] = []          # (exit_t, symbol)
            closed: list[tuple[int, float]] = []          # (exit_t, net) taken today
            day_rows: list[TradeRow] = []
            max_conc = 0
            for cand in todays:
                et = cand.entry_t
                open_pos = [p for p in open_pos if p[0] >= et]
                if status == COOLDOWN:
                    q, capped = self.size(cand.entry_fill, cand.risk_per_share, c.reduced_risk_mult)
                    day_rows.append(self._row(cand, "shadow", "cooldown", q, capped,
                                              c.reduced_risk_mult))
                    continue
                skip = ""
                if c.one_position_per_symbol and any(p[1] == cand.intent.symbol for p in open_pos):
                    skip = "symbol_busy"
                elif len(open_pos) >= c.max_open_positions:
                    skip = "max_open_positions"
                else:
                    skip = rr_gate_skip(cand, c)
                q, capped = self.size(cand.entry_fill, cand.risk_per_share, mult)
                if not skip and q < 1:
                    skip = "qty_zero"
                if not skip:
                    realized = sum(n for t, n in closed if t < et)
                    loss = max(0.0, -realized)
                    lim = cap * c.max_daily_loss_pct / 100.0
                    if loss >= lim:
                        skip = "daily_loss_limit"
                    elif loss + q * cand.risk_per_share > lim * 1.5:
                        skip = "daily_loss_breach"
                if skip:
                    day_rows.append(self._row(cand, "skipped", skip, q, capped, mult))
                    continue
                row = self._row(cand, "taken", "", q, capped, mult)
                day_rows.append(row)
                open_pos.append((cand.exit_t, cand.intent.symbol))
                closed.append((cand.exit_t, row.net))
                max_conc = max(max_conc, len(open_pos))
            taken = [r for r in day_rows if r.status == "taken"]
            net = sum(r.net for r in taken)
            daily.append({
                "day": d.isoformat(), "status": status, "risk_mult": mult,
                "trades": len(taken), "shadow": sum(r.status == "shadow" for r in day_rows),
                "skipped": sum(r.status == "skipped" for r in day_rows),
                "gross": round(sum(r.gross for r in taken), 2),
                "costs": round(sum(r.costs for r in taken), 2), "net": round(net, 2),
                "shadow_net": round(sum(r.net for r in day_rows if r.status == "shadow"), 2),
                "max_concurrent": max_conc,
            })
            rows.extend(day_rows)
            if c.cooldown and taken:
                cd.close_day(d, net)
        return SimResult(config=c, rows=rows, daily=daily, sessions=list(sessions),
                         phase1_rejects=dict(phase1_rejects or {}))


LIVE_MIN_RR_DEFAULT = 1.5   # settings.ALPHADESK["MIN_RISK_REWARD_RATIO"] default


def rr_gate_skip(cand: Candidate, cfg: SimConfig) -> str:
    """'' if the R:R gate passes, else the skip reason.

    * ``live_rr_gate`` — mirrors the live RiskEngine: no target → ``no_target``
      (``_check_fields`` rejects target <= 0); ``round(rr, 2) < min_rr`` →
      ``min_rr`` (``_rr_below_minimum``). min_rr defaults to 1.5.
    * otherwise, ``min_rr`` set → legacy research gate: only intents WITH a
      target are checked; target-less intents pass.
    """
    it = cand.intent
    if cfg.live_rr_gate:
        if it.target is None or it.target <= 0:
            return "no_target"
        mr = cfg.min_rr if cfg.min_rr is not None else LIVE_MIN_RR_DEFAULT
        rr = abs(it.target - cand.entry_fill) / cand.risk_per_share if cand.risk_per_share > 0 else 0.0
        return "min_rr" if round(rr, 2) < mr else ""
    if cfg.min_rr is not None and it.target is not None and (
            abs(it.target - cand.entry_fill) / cand.risk_per_share < cfg.min_rr - 1e-9):
        return "min_rr"
    return ""


def with_overrides(cfg: SimConfig, **kw) -> SimConfig:
    return replace(cfg, **kw)


# ── metrics ─────────────────────────────────────────────────────────────────


def _block(trades: Sequence[TradeRow], n_sessions: int) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    wins = [t.net for t in trades if t.net > 0]
    losses = [t.net for t in trades if t.net <= 0]
    gp, gl = sum(wins), -sum(losses)
    gross = sum(t.gross for t in trades)
    costs = sum(t.costs for t in trades)
    net = gross - costs
    gross_wins = sum(t.gross for t in trades if t.gross > 0)
    eq = peak = mdd = 0.0
    for t in sorted(trades, key=lambda x: (x.exit_t, x.entry_t)):
        eq += t.net
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    risk = [t.risk_rs for t in trades]
    return {
        "trades": n,
        "win_rate": len(wins) / n,
        "avg_win": gp / len(wins) if wins else 0.0,
        "avg_loss": -gl / len(losses) if losses else 0.0,
        "profit_factor": gp / gl if gl > 0 else (math.inf if gp > 0 else 0.0),
        "gross": gross,
        "costs": costs,
        "net": net,
        "expectancy_rs": net / n,
        "r_per_trade": sum(t.r_net for t in trades) / n,
        "gross_r_per_trade": sum(t.r_gross for t in trades) / n,
        "costs_r_per_trade": sum(t.costs / t.risk_rs for t in trades if t.risk_rs) / n,
        "cost_share_of_gross_wins": costs / gross_wins if gross_wins > 0 else None,
        "max_drawdown": mdd,
        "trades_per_day": n / n_sessions if n_sessions else 0.0,
        "long_trades": sum(1 for t in trades if t.side > 0),
        "avg_risk_rs": sum(risk) / n,
        "avg_notional": sum(t.notional for t in trades) / n,
        "notional_capped_pct": sum(t.notional_capped for t in trades) / n,
        "exit_reasons": dict(Counter(t.reason for t in trades)),
    }


def summarize(taken: Sequence[TradeRow], sessions: Sequence[date], rows: Sequence[TradeRow],
              daily: Sequence[dict], cfg: SimConfig, phase1_rejects: dict | None = None) -> dict:
    ns = len(sessions)
    out = {"config": cfg.label, "sessions": ns, **_block(taken, ns)}
    dn = [d["net"] for d in daily]
    eq = peak = dd = 0.0
    for v in dn:
        eq += v
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    traded = [d for d in daily if d["trades"]]
    out.update({
        "return_pct_of_capital": (sum(dn) / cfg.capital * 100) if cfg.capital else None,
        "max_drawdown_daily": dd,
        "daily_mean": statistics.fmean(dn) if dn else 0.0,
        "daily_stdev": statistics.pstdev(dn) if len(dn) > 1 else 0.0,
        "days_traded": len(traded),
        "days_positive": sum(1 for d in traded if d["net"] > 0),
        "cooldown_days": sum(1 for d in daily if d["status"] == COOLDOWN),
        "reduced_risk_days": sum(1 for d in daily if d["status"] == "reduced"),
        "max_concurrent": max((d["max_concurrent"] for d in daily), default=0),
        "signals_total": len(rows),
        "skipped": dict(Counter(r.skip_reason for r in rows if r.status == "skipped")),
        "shadow_trades": sum(1 for r in rows if r.status == "shadow"),
        "shadow_net": sum(r.net for r in rows if r.status == "shadow"),
        "phase1_rejects": dict(phase1_rejects or {}),
        "per_strategy": {s: _block([t for t in taken if t.strategy == s], ns)
                         for s in sorted({t.strategy for t in taken})},
    })
    return out


def candidate_day(c: Candidate) -> date:
    return day_of(c.entry_t)
