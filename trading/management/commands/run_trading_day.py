"""
Run Trading Day — single persistent process for the full trading session.

Orchestrates ALL agents from pre-market to post-market:
  7:00 AM  — Premarket scan + watchlist
  9:15 AM  — Start equity monitor (5-min cycles) + straddle monitor (10-min cycles)
  3:00 PM  — Force-close all 0 DTE straddles, stop new equity entries
  3:15 PM  — Square off all open equity positions
  3:30 PM  — Daily review, journal, shutdown

One command. One process. Full day coverage.

Usage:
    python manage.py run_trading_day
    python manage.py run_trading_day --universe high_volume --skip-llm
    python manage.py run_trading_day --dry-run  # no execution, analysis only
"""
import json
import os
import signal
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

from django.core.management.base import BaseCommand
from logzero import logger

# ── Shared event log: run_trading_day writes, dashboard reads ──
EVENT_LOG_PATH = Path(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))) / "logs" / "trading_day_events.jsonl"


class Command(BaseCommand):
    help = "Run the full trading day — premarket scan + equity monitor + straddle monitor + auto-close"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._running = True

    def add_arguments(self, parser):
        parser.add_argument("--universe", default="high_volume", help="Stock universe (default: high_volume)")
        parser.add_argument("--skip-llm", action="store_true", dest="skip_llm", help="Skip LLM analysis")
        parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Analyze only, no execution")
        parser.add_argument("--max-positions", type=int, default=3, dest="max_positions")
        parser.add_argument("--straddle-interval", type=int, default=10, dest="straddle_interval",
                            help="Minutes between straddle monitor cycles (default: 10)")
        parser.add_argument("--equity-interval", type=int, default=5, dest="equity_interval",
                            help="Minutes between equity scan cycles (default: 5)")

    def handle(self, *args, **options):
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        self._dry_run = options["dry_run"]
        self._universe = options["universe"]
        self._skip_llm = options["skip_llm"]
        self._max_positions = options["max_positions"]
        self._straddle_interval = options["straddle_interval"]
        self._equity_interval = options["equity_interval"]

        # ── Initialize broker singleton once ──
        from trading.services.data_service import BrokerClient
        self._broker = BrokerClient.get_instance()
        self._broker.ensure_login()

        self._log(
            f"\n{'='*60}\n"
            f"TRADING DAY: {date.today()}\n"
            f"{'='*60}\n"
            f"  Universe       : {self._universe}\n"
            f"  Max positions  : {self._max_positions}\n"
            f"  Equity cycle   : every {self._equity_interval} min\n"
            f"  Straddle cycle : every {self._straddle_interval} min\n"
            f"  Dry run        : {self._dry_run}\n"
            f"  Trading mode   : {os.getenv('TRADING_MODE', 'paper').upper()}\n"
            f"{'='*60}",
            style="SUCCESS",
        )

        # Clear event log only if it's from a previous day
        try:
            EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            if EVENT_LOG_PATH.exists():
                import json as _json
                first_line = EVENT_LOG_PATH.read_text().split("\n")[0].strip()
                if first_line:
                    first_event = _json.loads(first_line)
                    if first_event.get("data", {}).get("date") != date.today().isoformat():
                        EVENT_LOG_PATH.write_text("")  # New day — clear
                    # Same day — append (resume)
            # else: new file, nothing to clear
        except Exception:
            pass

        self._emit("DAY_START", {
            "date": date.today().isoformat(),
            "universe": self._universe,
            "mode": os.getenv("TRADING_MODE", "paper"),
            "dry_run": self._dry_run,
        })

        # ── Startup health check ──
        self._startup_health_check()

        # ── State ──
        self._watchlist = []
        self._equity_state = None
        self._premarket_done = False
        self._closing_done = False
        self._straddle_registered = False
        self._last_equity_snapshot = {"positions": [], "unrealized": 0}

        # Stagger initial cycles to prevent startup rate-limit burst:
        # Straddle fires immediately, equity waits 2 min for API budget to recover
        self._last_equity_cycle = time.monotonic()   # equity waits full interval
        self._last_straddle_cycle = 0.0              # straddle fires first

        # Pre-populate equity snapshot from DB (immediate dashboard data)
        self._refresh_equity_snapshot()

        # ── Main loop ──
        while self._running:
            now = datetime.now()

            # Weekend
            if now.weekday() >= 5:
                self._log("Weekend. Nothing to do.")
                break

            # Before 7 AM — sleep
            if now.hour < 7:
                self._sleep(60)
                continue

            # 7:00 - 9:14 — Premarket phase
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                if not self._premarket_done:
                    self._run_premarket()
                    self._premarket_done = True
                else:
                    mins_to_open = ((9 * 60 + 15) - (now.hour * 60 + now.minute))
                    if mins_to_open > 0 and mins_to_open % 5 == 0:
                        self._log(f"Waiting for market open... {mins_to_open} min")
                    self._sleep(30)
                continue

            # 9:15 - 14:59 — Active trading
            if now.hour < 15:
                # Late start: run premarket if we missed it
                if not self._premarket_done:
                    self._log("Late start — running premarket scan now")
                    self._run_premarket()
                    self._premarket_done = True

                # AI pause check
                from dashboard_utils.data_layer import is_ai_paused
                if is_ai_paused():
                    self._log("AI trading PAUSED. Waiting...", style="WARNING")
                    self._sleep(30)
                    continue

                self._run_active_phase(now)
                continue

            # 15:00 - 15:14 — Closing phase
            if now.hour == 15 and now.minute < 15:
                if not self._closing_done:
                    self._run_closing_phase()
                    self._closing_done = True
                else:
                    self._sleep(10)
                continue

            # 15:15 - 15:30 — Square off + review
            if now.hour == 15 and now.minute < 31:
                self._run_square_off()
                self._run_daily_review()
                break

            # After 15:30 — run review if we haven't yet
            self._run_daily_review()
            break

        self._log("Trading day process ended.", style="SUCCESS")

    # ══════════════════════════════════════════════
    # PHASE: Premarket (7:00 - 9:14)
    # ══════════════════════════════════════════════
    def _run_premarket(self):
        self._log("=" * 50, style="SUCCESS")
        self._log("PHASE: PREMARKET SCAN", style="SUCCESS")
        self._log("=" * 50)

        from trading.intraday.scanner import PremarketScanner
        from trading.intraday.state import IntradayState, Phase, StockSetup, TradeBias, SetupType
        from trading.models import WatchlistEntry

        # Resume: if watchlist already exists in DB for today, load it
        existing = WatchlistEntry.objects.filter(scan_date=date.today()).order_by("-score")
        if existing.exists():
            self._log(f"Resuming — loading {existing.count()} stocks from DB watchlist")
            from trading.services.ticker_service import ticker_service
            from trading.utils.indicators import camarilla_pivots
            self._watchlist = []
            for e in existing:
                token = ticker_service.get_token(e.symbol) or ""
                setup_types = [SetupType(s) for s in e.setups if s in [st.value for st in SetupType]]
                pivots = camarilla_pivots(e.prev_high, e.prev_low, e.prev_close) if e.prev_high > 0 else {}
                self._watchlist.append(StockSetup(
                    symbol=e.symbol, token=token, score=e.score,
                    bias=TradeBias(e.bias) if e.bias in ["LONG", "SHORT", "NEUTRAL"] else TradeBias.NEUTRAL,
                    setups=setup_types,
                    prev_high=e.prev_high, prev_low=e.prev_low,
                    prev_close=e.prev_close, prev_atr=e.prev_atr,
                    pivot_s3=pivots.get("S3", 0), pivot_s4=pivots.get("S4", 0),
                    pivot_r3=pivots.get("R3", 0), pivot_r4=pivots.get("R4", 0),
                    pivot_p=pivots.get("P", 0),
                ))
        else:
            scanner = PremarketScanner(lookback_days=5, top_n=10)
            self._watchlist = scanner.scan(
                universe=self._universe,
                trading_date=date.today().strftime("%Y-%m-%d"),
            )

        self._log(f"Watchlist: {len(self._watchlist)} stocks")
        self._emit("WATCHLIST", {
            "count": len(self._watchlist),
            "stocks": [{"symbol": s.symbol, "score": s.score, "bias": s.bias.value,
                         "setups": [x.value for x in s.setups]} for s in self._watchlist],
        })

        # Initialize equity state
        capital = 500000.0
        try:
            margin = self._broker.margin_available()
            net = float(margin.get("net", 0))
            if net > 0:
                capital = net
                self._log(f"Capital from broker: {capital:,.0f} INR")
        except Exception:
            pass
        try:
            from trading.models import PortfolioSnapshot
            snap = PortfolioSnapshot.objects.latest()
            if snap.capital > 0:
                capital = snap.capital
                self._log(f"Capital from portfolio: {capital:,.0f} INR")
        except Exception:
            pass

        # Count already-open equity positions from today (resume support)
        from trading.models import TradeJournal
        open_today = TradeJournal.objects.filter(
            trade_date=date.today(), status__in=["EXECUTED", "PAPER"]
        ).count()
        if open_today > 0:
            self._log(f"Resuming with {open_today} open equity position(s) from earlier")

        self._equity_state = IntradayState(
            trading_date=date.today().strftime("%Y-%m-%d"),
            capital=capital,
            max_positions=self._max_positions,
            watchlist=self._watchlist,
            phase=Phase.PREMARKET,
            open_positions=open_today,
        )

        # LLM premarket analysis (optional)
        if not self._skip_llm and self._watchlist:
            try:
                from trading.intraday.prompts import build_premarket_prompt
                from trading.intraday.agent import _call_llm
                system_prompt, user_prompt = build_premarket_prompt(self._equity_state)
                analysis = _call_llm(system_prompt, user_prompt)
                self._equity_state.premarket_analysis = analysis
                self._log(f"LLM premarket analysis: {analysis}...")
            except Exception as e:
                self._log(f"LLM analysis failed (non-fatal): {e}", style="WARNING")

    # ══════════════════════════════════════════════
    # PHASE: Active Trading (9:15 - 14:59)
    # ══════════════════════════════════════════════
    def _run_active_phase(self, now: datetime):
        elapsed_equity = time.monotonic() - self._last_equity_cycle
        elapsed_straddle = time.monotonic() - self._last_straddle_cycle

        # Auto-register straddle after market opens (once per day)
        if not self._straddle_registered and now.hour >= 9 and now.minute >= 20:
            # Check if one already exists (resume case)
            from trading.models import StraddlePosition
            if StraddlePosition.objects.filter(status__in=["ACTIVE", "PARTIAL", "HEDGED"]).exists():
                self._straddle_registered = True
                self._log("Straddle already active — skipping auto-registration")
        if not self._straddle_registered and now.hour >= 9 and now.minute >= 20:
            self._maybe_register_straddle()
            self._straddle_registered = True

        # Equity scan cycle
        if elapsed_equity >= (self._equity_interval * 60):
            self._run_equity_cycle(now)
            self._last_equity_cycle = time.monotonic()
            return  # Don't run straddle in same cycle — stagger API calls

        # Straddle monitor cycle (offset from equity to avoid rate limit collision)
        if elapsed_straddle >= (self._straddle_interval * 60):
            self._run_straddle_cycle(now)
            self._last_straddle_cycle = time.monotonic()
            # Straddle workflow makes LTP + LLM calls. Cool down before equity scan.
            time.sleep(5)
            # Push equity timer forward so it doesn't fire immediately after
            if (time.monotonic() - self._last_equity_cycle) < (self._equity_interval * 60 - 30):
                pass  # equity not due yet, fine
            else:
                self._last_equity_cycle = time.monotonic() - (self._equity_interval * 60 - 60)  # defer by 60s

        # Heartbeat every cycle — use cached equity data (no API call here)
        # Live prices are fetched only during equity scan cycles (every 5 min)
        from trading.models import TradeJournal, StraddlePosition

        open_eq = TradeJournal.objects.filter(
            trade_date=date.today(), status__in=["EXECUTED", "PAPER"]
        ).count()
        active_str = StraddlePosition.objects.filter(
            status__in=["ACTIVE", "PARTIAL", "HEDGED"]
        ).count()
        closed = TradeJournal.objects.filter(trade_date=date.today(), status="FILLED")
        realized_pnl = sum(t.pnl or 0 for t in closed)

        self._emit("HEARTBEAT", {
            "phase": "ACTIVE",
            "open_equity": open_eq,
            "active_straddles": active_str,
            "equity_positions": self._last_equity_snapshot.get("positions", []),
            "equity_unrealized": self._last_equity_snapshot.get("unrealized", 0),
            "equity_realized": round(realized_pnl, 0),
            "broker_calls": self._broker.get_stats()["api_calls"],
            "cache_hits": self._broker.get_stats()["cache_hits"],
        })

        # ── Hourly snapshot (on the hour) ──
        if now.minute < 1 and now.hour >= 10 and now.hour <= 15:
            last_snap_hour = getattr(self, '_last_snapshot_hour', 0)
            if now.hour != last_snap_hour:
                self._last_snapshot_hour = now.hour
                self._post_hourly_snapshot(now, open_eq, active_str, realized_pnl)

        # Sleep until next check (every 15 seconds)
        self._sleep(15)

    def _post_hourly_snapshot(self, now: datetime, open_eq: int, active_str: int, realized_pnl: float):
        """Post an hourly portfolio snapshot to event log + Telegram."""
        from trading.models import TradeJournal, StraddlePosition
        from trading.options.data_service import OptionsDataService

        # Equity unrealized
        eq_unrealized = self._last_equity_snapshot.get("unrealized", 0)

        # Options unrealized
        opt_unrealized = 0
        opt_realized = 0
        for p in StraddlePosition.objects.filter(status__in=["ACTIVE", "PARTIAL", "HEDGED"]):
            opt_unrealized += p.current_pnl_inr
            opt_realized += p.realized_pnl
        for p in StraddlePosition.objects.filter(status="CLOSED", trade_date=date.today()):
            opt_realized += p.total_pnl

        total = realized_pnl + eq_unrealized + opt_unrealized + opt_realized

        # NIFTY spot
        nifty_ltp = 0
        try:
            svc = OptionsDataService()
            nifty_ltp = svc.fetch_nifty_spot().get("ltp", 0)
        except Exception:
            pass

        snapshot = {
            "time": now.strftime("%H:%M"),
            "nifty": nifty_ltp,
            "equity_open": open_eq,
            "equity_realized": round(realized_pnl, 0),
            "equity_unrealized": round(eq_unrealized, 0),
            "options_active": active_str,
            "options_realized": round(opt_realized, 0),
            "options_unrealized": round(opt_unrealized, 0),
            "total_pnl": round(total, 0),
            "capital": 500000 + total,
        }

        self._log(
            f"  ── HOURLY SNAPSHOT {now.strftime('%H:00')} ──\n"
            f"    NIFTY: {nifty_ltp:,.0f}\n"
            f"    Equity: {open_eq} open | realized {realized_pnl:+,.0f} | unrealized {eq_unrealized:+,.0f}\n"
            f"    Options: {active_str} active | realized {opt_realized:+,.0f} | unrealized {opt_unrealized:+,.0f}\n"
            f"    DAY TOTAL: {total:+,.0f} | Capital: {500000+total:,.0f}",
            style="SUCCESS",
        )

        self._emit("HOURLY_SNAPSHOT", snapshot)

        # Post to Telegram if configured
        try:
            import os, urllib.request, json
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
            if bot_token and chat_id:
                msg = (
                    f"📊 *{now.strftime('%H:00')} Snapshot*\n"
                    f"NIFTY: `{nifty_ltp:,.0f}`\n"
                    f"Equity: {open_eq} open | `{realized_pnl+eq_unrealized:+,.0f}`\n"
                    f"Options: {active_str} active | `{opt_realized+opt_unrealized:+,.0f}`\n"
                    f"*Day Total: `{total:+,.0f}` INR*"
                )
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                data = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode()
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass  # Telegram is optional

    def _run_equity_cycle(self, now: datetime):
        from trading.intraday.monitor import IntradayMonitor
        from trading.intraday.state import Phase

        if not self._equity_state or not self._equity_state.watchlist:
            return

        # Always check SL/targets + cache live prices for dashboard
        self._check_equity_exits(now)

        if self._equity_state.open_positions >= self._equity_state.max_positions:
            return  # No room for new trades, exit check was enough

        self._equity_state.phase = Phase.ACTIVE
        monitor = IntradayMonitor(self._equity_state)

        self._log(f"--- Equity scan: {len(self._equity_state.watchlist)} stocks | "
                  f"Open: {self._equity_state.open_positions}/{self._equity_state.max_positions} ---")

        signals = monitor.run_scan_cycle()

        # ── Screener signals (V2 strategies — VWAP, Pivot, BB, EMA) ──
        screener_signals = self._run_screener_scan()
        if screener_signals:
            self._log(f"  Screener: {len(screener_signals)} signal(s)")
            # Convert screener Signal → IntradaySignal for unified processing
            from trading.intraday.state import IntradaySignal, SetupType, TradeBias
            for ss in screener_signals:
                try:
                    setup = SetupType.VWAP_RECLAIM if ss.side == "BUY" else SetupType.VWAP_REJECT
                    bias = TradeBias.LONG if ss.side == "BUY" else TradeBias.SHORT
                    intra_sig = IntradaySignal(
                        symbol=ss.symbol, token="",
                        setup_type=setup, bias=bias, side=ss.side,
                        entry_price=ss.entry, stop_loss=ss.stoploss, target=ss.target,
                        risk_reward=ss.risk_reward, confidence=ss.confidence,
                        reason=f"[SCR:{ss.strategy}] {' | '.join(ss.reasons[:2])}",
                        near_pivot=ss.strategy,
                    )
                    signals.append(intra_sig)
                except Exception as e:
                    logger.warning(f"Screener signal conversion failed: {e}")

        self._emit("EQUITY_SCAN", {
            "watchlist_size": len(self._equity_state.watchlist),
            "signals_found": len(signals),
            "screener_signals": len(screener_signals) if screener_signals else 0,
            "open_positions": self._equity_state.open_positions,
        })

        for sig in signals:
            if self._equity_state.open_positions >= self._equity_state.max_positions:
                break
            if self._dry_run:
                self._log(f"  [DRY] {sig.symbol} {sig.setup_type.value} {sig.bias.value} "
                          f"@ {sig.entry_price:.2f} RR:{sig.risk_reward:.1f}")
                self._emit("SIGNAL", {
                    "symbol": sig.symbol, "setup": sig.setup_type.value,
                    "bias": sig.bias.value, "entry": sig.entry_price,
                    "rr": sig.risk_reward, "dry_run": True,
                })
                continue

            result = monitor.process_signal(sig)
            action = result.get("action", "?")
            self._log(f"  {action}: {sig.symbol} — {result.get('reason', '')[:60]}")

            self._emit("TRADE" if action == "TRADED" else "SIGNAL_REJECTED", {
                "symbol": sig.symbol, "setup": sig.setup_type.value,
                "action": action, "side": "BUY" if sig.bias.value == "LONG" else "SELL",
                "entry": sig.entry_price, "sl": sig.stop_loss, "target": sig.target,
                "rr": sig.risk_reward, "reason": result.get("reason", "")[:100],
            })

            if action == "TRADED":
                self._equity_state.open_positions += 1

    def _run_screener_scan(self) -> list:
        """
        Run the screener engine on watchlist symbols.
        Returns list of screener Signal objects (not IntradaySignal).
        Lightweight — reuses cached candle data, no extra API calls.
        """
        try:
            from trading.screener.engine import ScreenerEngine
            from trading.screener.strategies import STRATEGIES
            from trading.services.ticker_service import ticker_service
            from trading.utils.time_utils import can_fetch_candles, cap_end_time

            # Use watchlist symbols if available, else top stocks
            symbols = [s.symbol for s in (self._equity_state.watchlist if self._equity_state else [])]
            if not symbols:
                return []

            # Initialize engine (lazy — reuse if possible)
            if not hasattr(self, '_screener_engine') or self._screener_engine is None:
                enabled = [s for s in STRATEGIES if s.enabled]
                self._screener_engine = ScreenerEngine(symbols=symbols[:10], strategies=enabled)

                # Bootstrap with candles (uses 1 batch call + candle fetches)
                def _fetch(symbol, tf, n_bars):
                    token = ticker_service.get_token(symbol)
                    if not token or not can_fetch_candles():
                        return []
                    interval = {"1m": "ONE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE"}.get(tf, "FIVE_MINUTE")
                    end = cap_end_time(date.today().isoformat())
                    return self._broker.fetch_candles(token, f"{date.today().isoformat()} 09:15", end, interval) or []

                self._screener_engine.bootstrap(fetch_candles_fn=_fetch)

            # Feed current prices (from cached equity snapshot — no extra API call)
            batch = []
            snapshot = getattr(self, '_last_equity_snapshot', None) or []
            for item in snapshot:
                if isinstance(item, dict) and item.get("symbol") in symbols:
                    batch.append({
                        "symbol": item["symbol"],
                        "ltp": item.get("ltp", 0),
                        "volume": item.get("volume", 0),
                    })

            if not batch:
                return []

            signals = []
            self._screener_engine.add_output_handler(lambda sig: signals.append(sig))
            self._screener_engine.on_batch_tick(batch)
            # Remove handler to prevent accumulation
            self._screener_engine._handlers = [h for h in self._screener_engine._handlers
                                                 if h.__name__ != '<lambda>']
            return signals

        except Exception as e:
            logger.warning(f"Screener scan failed (non-fatal): {e}")
            return []

    def _check_equity_exits(self, now: datetime):
        """Check SL/target hits + cache live prices for dashboard."""
        from trading.models import TradeJournal
        from trading.services.ticker_service import ticker_service

        # Refresh snapshot (1 batch API call — prices cached for dashboard)
        self._refresh_equity_snapshot()

        open_trades = list(TradeJournal.objects.filter(
            trade_date=date.today(),
            status__in=["EXECUTED", "PAPER"],
        ))
        if not open_trades:
            return

        # Build price lookup from cached snapshot (no extra API call)
        snap_prices = {p["symbol"]: p["ltp"] for p in self._last_equity_snapshot.get("positions", [])}
        if not snap_prices:
            return

        from trading.services.broker_service import BrokerService
        broker_svc = BrokerService()

        for trade in open_trades:
            ltp = snap_prices.get(trade.symbol, 0)
            if ltp <= 0:
                continue

            # ── Trailing stop management ──
            # Risk per share (R) = distance from entry to original SL
            risk_per_share = abs(trade.entry_price - trade.stop_loss)
            if risk_per_share > 0:
                if trade.side == "BUY":
                    profit_r = (ltp - trade.entry_price) / risk_per_share
                else:
                    profit_r = (trade.entry_price - ltp) / risk_per_share

                # Move SL to breakeven after 1R profit
                if profit_r >= 1.0:
                    new_sl = trade.entry_price  # breakeven
                    if trade.side == "BUY" and new_sl > trade.stop_loss:
                        trade.stop_loss = new_sl
                        trade.save(update_fields=["stop_loss"])
                    elif trade.side == "SELL" and new_sl < trade.stop_loss:
                        trade.stop_loss = new_sl
                        trade.save(update_fields=["stop_loss"])

                # Trail SL by 0.5R after 2R profit
                if profit_r >= 2.0:
                    trail_distance = risk_per_share * 0.5
                    if trade.side == "BUY":
                        trailing_sl = round(ltp - trail_distance, 2)
                        if trailing_sl > trade.stop_loss:
                            trade.stop_loss = trailing_sl
                            trade.save(update_fields=["stop_loss"])
                    else:
                        trailing_sl = round(ltp + trail_distance, 2)
                        if trailing_sl < trade.stop_loss:
                            trade.stop_loss = trailing_sl
                            trade.save(update_fields=["stop_loss"])

            hit_sl = (trade.side == "BUY" and ltp <= trade.stop_loss) or \
                     (trade.side == "SELL" and ltp >= trade.stop_loss)
            hit_tgt = (trade.side == "BUY" and ltp >= trade.target) or \
                      (trade.side == "SELL" and ltp <= trade.target)

            if hit_sl or hit_tgt:
                exit_side = "SELL" if trade.side == "BUY" else "BUY"
                reason = "SL HIT" if hit_sl else "TARGET HIT"

                if not self._dry_run:
                    broker_svc.place_order(
                        symbol=trade.symbol, side=exit_side,
                        quantity=trade.quantity, price=ltp,
                        order_type="MARKET", product_type="INTRADAY",
                    )
                    pnl = (ltp - trade.entry_price) * trade.quantity if trade.side == "BUY" \
                        else (trade.entry_price - ltp) * trade.quantity
                    trade.pnl = round(pnl, 2)
                    trade.fill_price = ltp
                    trade.status = "FILLED"
                    trade.save()
                    self._equity_state.open_positions = max(0, self._equity_state.open_positions - 1)
                    self._equity_state.daily_loss += max(0, -pnl)

                self._log(
                    f"  EXIT: {exit_side} {trade.quantity}x {trade.symbol} @ {ltp:.2f} "
                    f"| {reason} | P&L: {trade.pnl or 0:+,.0f} INR",
                    style="WARNING" if hit_sl else "SUCCESS",
                )
                self._emit("EXIT", {
                    "symbol": trade.symbol, "side": exit_side, "price": ltp,
                    "reason": reason, "pnl": trade.pnl or 0,
                })

    def _run_straddle_cycle(self, now: datetime):
        from trading.models import StraddlePosition

        active = list(StraddlePosition.objects.filter(status__in=["ACTIVE", "PARTIAL", "HEDGED"]))
        if not active:
            return

        self._log(f"--- Straddle check: {len(active)} position(s) ---")

        for pos in active:
            dte = (pos.expiry - date.today()).days

            # ── Check if this is a SPREAD (defined risk — no lifecycle management needed) ──
            is_spread = False
            if pos.management_log:
                first_log = pos.management_log[0] if pos.management_log else {}
                is_spread = first_log.get("spread_type") is not None

            if is_spread:
                # Spreads just need price update — risk is already capped
                from trading.options.data_service import OptionsDataService
                svc = OptionsDataService()
                ce_data = svc.fetch_option_ltp(pos.ce_symbol, pos.ce_token)
                pe_data = svc.fetch_option_ltp(pos.pe_symbol, pos.pe_token)
                pos.ce_current_price = ce_data.get("ltp", pos.ce_current_price)
                pos.pe_current_price = pe_data.get("ltp", pos.pe_current_price)

                # Compute spread P&L: (long_value - short_value - net_debit) × lots
                # CE fields = long leg, PE fields = short leg
                first_log = pos.management_log[0]
                net_debit = first_log.get("net_debit", 0)
                max_loss = first_log.get("max_loss", net_debit * pos.lot_size * pos.lots)
                max_profit = first_log.get("max_profit", (200 - net_debit) * pos.lot_size * pos.lots)
                long_ltp = pos.ce_current_price
                short_ltp = pos.pe_current_price
                spread_value = long_ltp - short_ltp
                raw_pnl = (spread_value - net_debit) * pos.lot_size * pos.lots
                # Clamp to defined risk bounds
                pos.current_pnl_inr = max(-max_loss, min(max_profit, raw_pnl))
                pos.save(update_fields=["ce_current_price", "pe_current_price",
                                         "current_pnl_inr", "last_updated"])

                self._log(f"  #{pos.id} SPREAD | long={long_ltp:.1f} short={short_ltp:.1f} | "
                          f"P&L: {pos.total_pnl:+,.0f} | {first_log.get('spread_type', 'SPREAD')}")

                # Close at expiry
                if dte <= 0:
                    pos.status = "CLOSED"
                    pos.action_taken = "CLOSE_BOTH"
                    pos.closed_at = datetime.now()
                    pos.save()
                    self._log(f"  #{pos.id} SPREAD EXPIRED | Final P&L: {pos.total_pnl:+,.0f}")

                self._emit("STRADDLE_CYCLE", {
                    "position_id": pos.id, "underlying": pos.underlying,
                    "strike": pos.display_strike, "action": "HOLD",
                    "pnl": pos.total_pnl, "reason": "Spread — defined risk, no lifecycle needed",
                })
                continue

            if self._dry_run:
                from dashboard_utils.data_layer import run_straddle_analysis
                from trading.options.data_service import OptionsDataService
                svc = OptionsDataService()
                result = run_straddle_analysis(svc, pos.id, include_candles=False)
                if "error" not in result:
                    pnl = result.get("net_pnl_inr", 0)
                    decay = result.get("premium_decayed_pct", 0)
                    underwater = result.get("is_underwater", False)
                    self._log(f"  [DRY] #{pos.id} {pos.underlying} {pos.strike} | "
                              f"P&L: {pnl:+,.0f} | Decay: {decay:.0f}% | UW: {underwater}")
                continue

            # ── Straddle lifecycle (only for actual straddles, not spreads) ──
            from trading.options.straddle.lifecycle import decide, count_shifts_today
            from trading.options.data_service import OptionsDataService

            svc = OptionsDataService()
            ce_data = svc.fetch_option_ltp(pos.ce_symbol, pos.ce_token)
            pe_data = svc.fetch_option_ltp(pos.pe_symbol, pos.pe_token)
            ce_ltp = ce_data.get("ltp", 0)
            pe_ltp = pe_data.get("ltp", 0)

            is_expiry_day = (pos.expiry - date.today()).days == 0
            shifts = count_shifts_today(pos.management_log)

            decision = decide(
                ce_sell=pos.ce_sell_price, pe_sell=pos.pe_sell_price,
                ce_ltp=ce_ltp, pe_ltp=pe_ltp,
                ce_strike=pos.ce_strike_actual, pe_strike=pos.pe_strike_actual,
                nifty_spot=svc.fetch_nifty_spot().get("ltp", 0),
                is_expiry_day=is_expiry_day,
                shifts_today=shifts,
            )

            action = decision.action
            pnl_pts = (pos.ce_sell_price + pos.pe_sell_price) - (ce_ltp + pe_ltp)
            pnl_inr = pnl_pts * pos.lot_size * pos.lots

            if action == "HOLD":
                # Update position prices in DB
                pos.ce_current_price = ce_ltp
                pos.pe_current_price = pe_ltp
                pos.current_pnl_inr = pnl_inr
                pos.save(update_fields=["ce_current_price", "pe_current_price",
                                         "current_pnl_inr", "last_updated"])

                self._log(f"  #{pos.id} {pos.underlying} {pos.display_strike} | HOLD | "
                          f"P&L: {pos.total_pnl:+,.0f} | {decision.reason[:60]}")

            elif action == "CLOSE_BOTH":
                # Direct execution — no LLM, no validator, lifecycle already decided
                from trading.services.broker_service import BrokerService
                broker_svc = BrokerService()
                qty = pos.lot_size * pos.lots

                # Close CE
                r1 = broker_svc.place_order(
                    symbol=pos.ce_symbol, side="BUY", quantity=qty,
                    price=ce_ltp, product_type="CARRYFORWARD", exchange="NFO",
                    symbol_token=pos.ce_token,
                )
                # Close PE
                r2 = broker_svc.place_order(
                    symbol=pos.pe_symbol, side="BUY", quantity=qty,
                    price=pe_ltp, product_type="CARRYFORWARD", exchange="NFO",
                    symbol_token=pos.pe_token,
                )

                # Update DB
                pos.ce_current_price = ce_ltp
                pos.pe_current_price = pe_ltp
                pos.current_pnl_inr = pnl_inr
                pos.status = "CLOSED"
                pos.action_taken = "CLOSE_BOTH"
                pos.closed_at = datetime.now()
                if pos.management_log is None:
                    pos.management_log = []
                pos.management_log.append({
                    "time": datetime.now().strftime("%H:%M"),
                    "action": "CLOSE_BOTH",
                    "nifty": svc.fetch_nifty_spot().get("ltp", 0),
                    "pnl_inr": pos.total_pnl,
                    "reason": decision.reason[:100],
                    "executed": True,
                })
                pos.save()

                self._log(f"  #{pos.id} CLOSED | CE@{ce_ltp:.1f} PE@{pe_ltp:.1f} | "
                          f"P&L: {pos.total_pnl:+,.0f} | {decision.reason[:50]}",
                          style="WARNING")

            elif action == "SHIFT_TO_ATM":
                # Close both legs + sell new ATM straddle
                from trading.services.broker_service import BrokerService
                from trading.options.data_service import find_atm_strike, find_option_token
                from trading.utils.expiry_utils import iso_to_angel
                broker_svc = BrokerService()
                qty = pos.lot_size * pos.lots
                new_strike = decision.new_strike

                self._log(f"  #{pos.id} SHIFT {pos.display_strike} → {new_strike} | {decision.reason[:50]}")

                # Step 1: Close both legs
                broker_svc.place_order(symbol=pos.ce_symbol, side="BUY", quantity=qty,
                    price=ce_ltp, product_type="CARRYFORWARD", exchange="NFO",
                    symbol_token=pos.ce_token)
                broker_svc.place_order(symbol=pos.pe_symbol, side="BUY", quantity=qty,
                    price=pe_ltp, product_type="CARRYFORWARD", exchange="NFO",
                    symbol_token=pos.pe_token)

                # Realize P&L from closing
                close_pnl = ((pos.ce_sell_price - ce_ltp) + (pos.pe_sell_price - pe_ltp)) * qty
                pos.realized_pnl += close_pnl

                # Step 2: Sell new ATM straddle
                exp_fmt = iso_to_angel(pos.expiry.isoformat())
                new_ce = find_option_token(pos.underlying, new_strike, exp_fmt, "CE") if exp_fmt else None
                new_pe = find_option_token(pos.underlying, new_strike, exp_fmt, "PE") if exp_fmt else None

                if new_ce and new_pe:
                    new_ce_sym, new_ce_tok = new_ce
                    new_pe_sym, new_pe_tok = new_pe
                    new_ce_ltp = svc.fetch_option_ltp(new_ce_sym, new_ce_tok).get("ltp", 0)
                    new_pe_ltp = svc.fetch_option_ltp(new_pe_sym, new_pe_tok).get("ltp", 0)

                    if new_ce_ltp > 0 and new_pe_ltp > 0:
                        broker_svc.place_order(symbol=new_ce_sym, side="SELL", quantity=qty,
                            price=new_ce_ltp, product_type="CARRYFORWARD", exchange="NFO",
                            symbol_token=new_ce_tok)
                        broker_svc.place_order(symbol=new_pe_sym, side="SELL", quantity=qty,
                            price=new_pe_ltp, product_type="CARRYFORWARD", exchange="NFO",
                            symbol_token=new_pe_tok)

                        # Update position to new legs
                        pos.ce_symbol = new_ce_sym
                        pos.ce_token = new_ce_tok
                        pos.ce_sell_price = new_ce_ltp
                        pos.ce_current_price = new_ce_ltp
                        pos.pe_symbol = new_pe_sym
                        pos.pe_token = new_pe_tok
                        pos.pe_sell_price = new_pe_ltp
                        pos.pe_current_price = new_pe_ltp
                        pos.strike = new_strike
                        pos.current_pnl_inr = 0  # fresh position

                        self._log(f"  #{pos.id} NEW straddle @{new_strike} | "
                                  f"CE={new_ce_ltp:.1f} PE={new_pe_ltp:.1f} | "
                                  f"Combined={new_ce_ltp+new_pe_ltp:.1f}")
                    else:
                        # Can't sell new — just close
                        pos.status = "CLOSED"
                        pos.action_taken = "CLOSE_BOTH"
                        pos.closed_at = datetime.now()
                        self._log(f"  #{pos.id} SHIFT failed (no premium) — closed instead")
                else:
                    pos.status = "CLOSED"
                    pos.action_taken = "CLOSE_BOTH"
                    pos.closed_at = datetime.now()
                    self._log(f"  #{pos.id} SHIFT failed (tokens not found) — closed instead")

                if pos.management_log is None:
                    pos.management_log = []
                pos.management_log.append({
                    "time": datetime.now().strftime("%H:%M"),
                    "action": action,
                    "nifty": svc.fetch_nifty_spot().get("ltp", 0),
                    "pnl_inr": pos.total_pnl,
                    "reason": decision.reason[:100],
                    "executed": True,
                })
                pos.save()

            pos.refresh_from_db()
            self._emit("STRADDLE_CYCLE", {
                "position_id": pos.id, "underlying": pos.underlying,
                "strike": pos.display_strike,
                "action": action, "pnl": pos.total_pnl,
                "unrealized": pos.current_pnl_inr, "realized": pos.realized_pnl,
                "reason": decision.reason[:80],
            })

    # ══════════════════════════════════════════════
    # PHASE: Closing (15:00 - 15:14)
    # ══════════════════════════════════════════════
    def _run_closing_phase(self):
        self._log("=" * 50, style="WARNING")
        self._log("PHASE: CLOSING — 3:00 PM", style="WARNING")
        self._log("=" * 50)

        # 1. Force-close all 0 DTE straddles
        from trading.models import StraddlePosition
        from trading.options.data_service import OptionsDataService
        from trading.services.broker_service import BrokerService

        svc = OptionsDataService()
        broker_svc = BrokerService()

        for pos in StraddlePosition.objects.filter(status__in=["ACTIVE", "PARTIAL", "HEDGED"]):
            dte = (pos.expiry - date.today()).days
            if dte > 0:
                self._log(f"  #{pos.id} DTE={dte} — keeping open (not expiry)")
                continue

            ce = svc.fetch_option_ltp(pos.ce_symbol, pos.ce_token)
            pe = svc.fetch_option_ltp(pos.pe_symbol, pos.pe_token)
            ce_ltp = ce.get("ltp", 0)
            pe_ltp = pe.get("ltp", 0)

            if not self._dry_run:
                qty = pos.lot_size * pos.lots
                broker_svc.place_order(symbol=pos.ce_symbol, side="BUY", quantity=qty,
                                       price=ce_ltp, product_type="CARRYFORWARD", exchange="NFO",
                                       symbol_token=pos.ce_token)
                broker_svc.place_order(symbol=pos.pe_symbol, side="BUY", quantity=qty,
                                       price=pe_ltp, product_type="CARRYFORWARD", exchange="NFO",
                                       symbol_token=pos.pe_token)

                pnl_pts = pos.combined_sell_pts - (ce_ltp + pe_ltp)
                pnl_inr = pnl_pts * pos.lot_size * pos.lots
                decay = (pnl_pts / pos.combined_sell_pts * 100) if pos.combined_sell_pts else 0

                pos.ce_current_price = ce_ltp
                pos.pe_current_price = pe_ltp
                pos.current_pnl_inr = pnl_inr
                pos.status = "CLOSED"
                pos.action_taken = "CLOSE_BOTH"
                pos.closed_at = datetime.now()
                if pos.management_log is None:
                    pos.management_log = []
                pos.management_log.append({
                    "time": datetime.now().strftime("%H:%M"),
                    "action": "CLOSE_BOTH",
                    "urgency": "IMMEDIATE",
                    "pnl_inr": pnl_inr,
                    "note": f"3:00 PM auto-close. Decay: {decay:.0f}%",
                    "executed": True,
                })
                pos.save()

            self._log(
                f"  CLOSED #{pos.id} {pos.underlying} {pos.strike} | "
                f"CE@{ce_ltp:.1f} PE@{pe_ltp:.1f} | P&L: {pos.current_pnl_inr:+,.0f} INR",
                style="SUCCESS" if pos.current_pnl_inr >= 0 else "ERROR",
            )

    # ══════════════════════════════════════════════
    # PHASE: Square Off (15:15)
    # ══════════════════════════════════════════════
    def _run_square_off(self):
        self._log("=" * 50, style="WARNING")
        self._log("PHASE: SQUARE OFF — 3:15 PM", style="WARNING")
        self._log("=" * 50)

        from trading.models import TradeJournal
        from trading.services.broker_service import BrokerService
        from trading.services.ticker_service import ticker_service

        trades = list(TradeJournal.objects.filter(
            trade_date=date.today(),
            status__in=["EXECUTED", "PAPER"],
        ))

        if not trades:
            self._log("  No open equity positions.")
            return

        # Batch fetch prices
        tokens, tok_map = [], {}
        for t in trades:
            tok = ticker_service.get_token(t.symbol)
            if tok:
                tokens.append(tok)
                tok_map[tok] = t

        prices = {}
        if tokens:
            for item in self._broker.market_data_batch({"NSE": tokens}, mode="LTP"):
                tok = str(item.get("symbolToken", ""))
                t = tok_map.get(tok)
                if t:
                    prices[t.symbol] = float(item.get("ltp", 0))

        broker_svc = BrokerService()
        total_pnl = 0

        for t in trades:
            ltp = prices.get(t.symbol, t.entry_price)
            exit_side = "SELL" if t.side == "BUY" else "BUY"

            if not self._dry_run:
                broker_svc.place_order(
                    symbol=t.symbol, side=exit_side, quantity=t.quantity,
                    price=ltp, order_type="MARKET", product_type="INTRADAY",
                )
                pnl = (ltp - t.entry_price) * t.quantity if t.side == "BUY" \
                    else (t.entry_price - ltp) * t.quantity
                t.pnl = round(pnl, 2)
                t.fill_price = ltp
                t.status = "FILLED"
                t.save()
                total_pnl += pnl

            self._log(f"  {exit_side} {t.quantity}x {t.symbol:12s} @ {ltp:.2f} | P&L: {t.pnl or 0:+,.0f}")

        self._log(f"  Equity squared off: {total_pnl:+,.0f} INR")

    # ══════════════════════════════════════════════
    # PHASE: Daily Review (3:20+)
    # ══════════════════════════════════════════════
    def _run_daily_review(self):
        self._log("=" * 50, style="SUCCESS")
        self._log("PHASE: DAILY REVIEW", style="SUCCESS")
        self._log("=" * 50)

        from trading.models import TradeJournal, StraddlePosition, WatchlistEntry

        trades = TradeJournal.objects.filter(trade_date=date.today())
        eq_pnl = sum(t.pnl or 0 for t in trades)
        eq_wins = trades.filter(pnl__gt=0).count()
        eq_losses = trades.filter(pnl__lt=0).count()

        straddles = StraddlePosition.objects.filter(trade_date=date.today())
        str_pnl = sum(p.total_pnl for p in straddles)

        total = eq_pnl + str_pnl

        self._log(f"  Equity:    {trades.count()} trades | W:{eq_wins} L:{eq_losses} | {eq_pnl:+,.0f} INR")
        self._log(f"  Straddle:  {straddles.count()} positions | {str_pnl:+,.0f} INR")
        self._log(f"  DAY TOTAL: {total:+,.0f} INR ({total/5000:+.2f}%)")

        # ── Strategy performance by setup type ──
        setup_stats = {}
        for t in trades:
            setup = t.reasoning.split("]")[0].replace("[", "").strip() if t.reasoning and "[" in t.reasoning else "UNKNOWN"
            if setup not in setup_stats:
                setup_stats[setup] = {"trades": 0, "wins": 0, "pnl": 0}
            setup_stats[setup]["trades"] += 1
            setup_stats[setup]["pnl"] += (t.pnl or 0)
            if (t.pnl or 0) > 0:
                setup_stats[setup]["wins"] += 1

        if setup_stats:
            self._log("  --- Strategy Performance ---")
            for setup, stats in sorted(setup_stats.items(), key=lambda x: -x[1]["pnl"]):
                wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] else 0
                self._log(f"    {setup:15s} T={stats['trades']} W={stats['wins']} "
                          f"WR={wr:.0f}% P&L={stats['pnl']:+,.0f}")

        # ── Watchlist conversion stats ──
        wl = WatchlistEntry.objects.filter(scan_date=date.today())
        if wl.exists():
            total_wl = wl.count()
            triggered = wl.filter(outcome="TRIGGERED").count()
            traded = wl.filter(outcome="TRADED").count()
            self._log(f"  --- Watchlist Conversion ---")
            self._log(f"    Scanned: {total_wl} | Triggered: {triggered} | Traded: {traded} | "
                      f"Conversion: {(traded/total_wl*100) if total_wl else 0:.0f}%")

        # ── Level quality scorecard (V2) ──
        trades_at_level = 0
        trades_total = trades.count()
        for t in trades:
            if t.reasoning and any(kw in t.reasoning for kw in
                    ["Cam_", "PDH", "PDL", "Round", "SWING", "VWAP", "ORB", "Level", "level"]):
                trades_at_level += 1
        if trades_total > 0:
            self._log(f"  --- Level Quality (V2) ---")
            self._log(f"    Trades at level: {trades_at_level}/{trades_total} "
                      f"({trades_at_level/trades_total*100:.0f}%)")
            # Target: 100% of trades should be at significant levels

        # ── Straddle performance detail ──
        for p in straddles:
            decay = ((p.combined_sell_pts - p.combined_current_pts) / p.combined_sell_pts * 100) \
                if p.combined_sell_pts else 0
            self._log(f"  --- Straddle #{p.id} ---")
            self._log(f"    {p.underlying} {p.strike} | {p.status} | Decay: {decay:.0f}% | "
                      f"P&L: {p.total_pnl:+,.0f} (realized: {p.realized_pnl:+,.0f}) | Actions: {len(p.management_log or [])}")

        # ── Auto-backtest: validate today's strategy against real data ──
        backtest_summary = {}
        try:
            from trading.services.intraday_backtester import run_intraday_backtest, BacktestConfig
            from trading.intraday.universe import get_universe

            today_str = date.today().isoformat()
            bt_symbols = get_universe(self._universe)[:10]
            bt_result = run_intraday_backtest(
                bt_symbols, today_str, today_str,
                BacktestConfig(min_confidence=0.65, mtf_enabled=True),
            )
            bs = bt_result["summary"]
            backtest_summary = {
                "trades": bs["total_trades"], "win_rate": bs["win_rate"],
                "pnl": bs["total_pnl"], "profit_factor": bs["profit_factor"],
            }
            self._log(f"  --- Backtest Validation (today's data) ---")
            self._log(f"    Trades: {bs['total_trades']} | WR: {bs['win_rate']:.0f}% | "
                      f"PF: {bs['profit_factor']:.2f} | P&L: {bs['total_pnl']:+,.0f}")

            # Flag if backtest disagrees with live
            if bs["total_trades"] > 0 and trades.count() > 0:
                bt_wr = bs["win_rate"]
                live_wr = (eq_wins / trades.count() * 100) if trades.count() else 0
                if abs(bt_wr - live_wr) > 20:
                    self._log(f"    WARNING: Backtest WR ({bt_wr:.0f}%) differs from live ({live_wr:.0f}%) by >20%",
                              style="WARNING")
        except Exception as e:
            self._log(f"  Backtest validation failed: {e}", style="WARNING")

        self._emit("DAY_REVIEW", {
            "equity_trades": trades.count(), "equity_wins": eq_wins, "equity_losses": eq_losses,
            "equity_pnl": eq_pnl, "straddle_positions": straddles.count(),
            "straddle_pnl": str_pnl, "total_pnl": total,
            "return_pct": round(total / 5000, 2),
            "setup_stats": setup_stats,
            "watchlist_scanned": wl.count() if wl.exists() else 0,
            "watchlist_traded": wl.filter(outcome="TRADED").count() if wl.exists() else 0,
            "backtest": backtest_summary,
        })

        # ── Update portfolio snapshot ──
        try:
            from trading.models import PortfolioSnapshot
            snap, _ = PortfolioSnapshot.objects.get_or_create(
                snapshot_date=date.today(),
                defaults={"capital": 500000, "available_cash": 500000},
            )
            snap.daily_pnl = total
            snap.total_pnl = (snap.total_pnl or 0) + total
            snap.daily_loss = sum(max(0, -(t.pnl or 0)) for t in trades)
            snap.open_positions = 0
            snap.save()
            self._log(f"  Portfolio updated: total P&L {snap.total_pnl:+,.0f} INR")
        except Exception as e:
            self._log(f"  Portfolio update failed: {e}", style="WARNING")

    # ══════════════════════════════════════════════
    # Startup Health Check
    # ══════════════════════════════════════════════
    def _startup_health_check(self):
        """Run at startup — clean zombies, validate universe, check readiness."""
        self._log("Running startup health check...", style="SUCCESS")

        from trading.models import TradeJournal, StraddlePosition

        # 1. Close zombie intraday positions from previous days
        zombies = TradeJournal.objects.filter(
            status__in=["EXECUTED", "PAPER"],
            trade_date__lt=date.today(),
        )
        if zombies.exists():
            count = zombies.count()
            zombies.update(status="CANCELLED", risk_reason="Auto-cancelled: overnight zombie")
            self._log(f"  Cleaned {count} zombie equity positions from previous days", style="WARNING")

        # 2. Close expired straddles that weren't properly closed
        expired = StraddlePosition.objects.filter(
            status__in=["ACTIVE", "PARTIAL", "HEDGED"],
            expiry__lt=date.today(),
        )
        if expired.exists():
            count = expired.count()
            expired.update(status="CLOSED", action_taken="CLOSE_BOTH")
            self._log(f"  Closed {count} expired straddle positions", style="WARNING")

        # 3. Validate ticker universe
        from trading.services.ticker_service import ticker_service
        from trading.intraday.universe import get_universe
        symbols = get_universe(self._universe)
        validity = ticker_service.validate_universe(symbols)
        invalid = [s for s, v in validity.items() if not v]
        if invalid:
            self._log(f"  WARNING: Invalid tickers in universe: {invalid}", style="WARNING")
        else:
            self._log(f"  Universe validated: {len(symbols)} tickers OK")

        # 4. Reconcile WatchlistEntry outcomes from TradeJournal
        from trading.models import WatchlistEntry, TradeJournal
        traded_symbols = TradeJournal.objects.filter(
            trade_date=date.today(), status__in=["EXECUTED", "PAPER", "FILLED"],
        ).values_list("symbol", flat=True).distinct()
        if traded_symbols:
            updated = WatchlistEntry.objects.filter(
                scan_date=date.today(), symbol__in=traded_symbols, outcome="WATCHING",
            ).update(outcome="TRADED")
            if updated:
                self._log(f"  Reconciled {updated} watchlist entries to TRADED")

        # 5. Check broker health
        self._log(f"  Broker: {self._broker.get_stats()}")

        self._emit("HEALTH_CHECK", {
            "zombies_cleaned": zombies.count() if hasattr(zombies, 'count') else 0,
            "expired_closed": expired.count() if hasattr(expired, 'count') else 0,
            "invalid_tickers": invalid,
            "broker_ok": self._broker.is_logged_in,
        })

    # ══════════════════════════════════════════════
    # Auto Straddle Registration
    # ══════════════════════════════════════════════
    def _maybe_register_straddle(self):
        """
        Adaptive options strategy — picks the right play for the market regime.

        Replaces fixed straddle-only with:
          VIX < 20       → Short Straddle (theta)
          VIX > 20 + gap → Bear/Bull Put/Call Spread (defined risk)
          VIX > 20 + flat→ Skip
          0 DTE + VIX<25 → Expiry-day theta
        """
        from trading.models import StraddlePosition
        from trading.options.data_service import OptionsDataService, find_option_token, find_atm_strike
        from trading.options.adaptive import AdaptiveOptionsEngine
        from trading.utils.expiry_utils import iso_to_angel, next_expiry_date

        if self._dry_run:
            return

        # Check if already have active position
        active = StraddlePosition.objects.filter(status__in=["ACTIVE", "PARTIAL", "HEDGED"])
        if active.exists():
            self._log(f"  Options: already active — skipping")
            return

        try:
            svc = OptionsDataService()
            nifty = svc.fetch_nifty_spot()
            vix_data = svc.fetch_vix()
            spot = nifty.get("ltp", 0)
            vix = vix_data.get("ltp", 0)
            prev_close = nifty.get("prev_close", 0)
            nifty_open = nifty.get("open", spot)

            if spot == 0 or vix == 0:
                self._log("  Options: no market data — skipping")
                return

            # Find next expiry
            expiry = next_expiry_date()
            dte = (expiry - date.today()).days if expiry else 5

            # Ask the adaptive engine
            engine = AdaptiveOptionsEngine()
            decision = engine.decide(
                nifty_spot=spot, vix=vix, dte=dte,
                nifty_prev_close=prev_close, nifty_open=nifty_open,
                expiry_date=expiry.isoformat() if expiry else "",
            )

            self._log(f"  Options: {decision.strategy} | {decision.reason[:60]}")
            self._emit("OPTIONS_DECISION", {
                "strategy": decision.strategy, "reason": decision.reason[:100],
                "vix": vix, "dte": dte, "spot": spot,
            })

            if decision.strategy == "SKIP":
                return

            if decision.strategy in ("STRADDLE", "0DTE_THETA"):
                # Register as straddle position (existing flow)
                self._register_straddle_from_decision(decision, svc, expiry)

            elif decision.strategy in ("BEAR_PUT_SPREAD", "BULL_CALL_SPREAD"):
                self._execute_spread_from_decision(decision, svc, expiry)

        except Exception as e:
            logger.error(f"Adaptive options failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # Legacy: check for individual underlyings below
        active = StraddlePosition.objects.filter(status__in=["ACTIVE", "PARTIAL", "HEDGED"])
        active_underlyings = set(active.values_list("underlying", flat=True))

        # Register for each underlying that doesn't have an active position
        straddle_configs = [
            {"underlying": "NIFTY", "lot_size": 75, "step": 50, "min_premium": 50},
            {"underlying": "BANKNIFTY", "lot_size": 15, "step": 100, "min_premium": 100},
        ]

        for cfg in straddle_configs:
            if cfg["underlying"] in active_underlyings:
                continue
            self._register_single_straddle(cfg)

    def _register_straddle_from_decision(self, decision, svc, expiry):
        """Register a straddle position from an adaptive decision."""
        from trading.options.data_service import find_option_token
        from trading.utils.expiry_utils import iso_to_angel
        from trading.models import StraddlePosition

        exp_fmt = iso_to_angel(expiry.isoformat()) if expiry else None
        if not exp_fmt:
            self._log("  Cannot register: no expiry format")
            return

        ce = find_option_token("NIFTY", decision.strike, exp_fmt, "CE")
        pe = find_option_token("NIFTY", decision.strike, exp_fmt, "PE")
        if not ce or not pe:
            self._log(f"  Cannot register: tokens not found for {decision.strike}")
            return

        ce_sym, ce_tok = ce
        pe_sym, pe_tok = pe
        ce_ltp = svc.fetch_option_ltp(ce_sym, ce_tok).get("ltp", 0)
        pe_ltp = svc.fetch_option_ltp(pe_sym, pe_tok).get("ltp", 0)

        if ce_ltp <= 0 or pe_ltp <= 0:
            self._log("  Cannot register: no premium data")
            return

        pos = StraddlePosition.objects.create(
            underlying="NIFTY", strike=decision.strike, expiry=expiry,
            lot_size=75, lots=decision.lots,
            ce_symbol=ce_sym, ce_token=ce_tok, ce_sell_price=ce_ltp,
            pe_symbol=pe_sym, pe_token=pe_tok, pe_sell_price=pe_ltp,
            trade_date=date.today(),
        )

        combined = ce_ltp + pe_ltp
        self._log(f"  REGISTERED #{pos.id} | {decision.strategy} @{decision.strike} | "
                  f"CE={ce_ltp:.1f} PE={pe_ltp:.1f} = {combined:.1f} pts")
        self._emit("STRADDLE_REGISTERED", {
            "position_id": pos.id, "strategy": decision.strategy,
            "strike": decision.strike, "combined": combined, "dte": (expiry - date.today()).days,
        })

    def _execute_spread_from_decision(self, decision, svc, expiry):
        """Execute a bear put or bull call spread from an adaptive decision."""
        from trading.options.data_service import find_option_token
        from trading.utils.expiry_utils import iso_to_angel
        from trading.services.broker_service import BrokerService
        from trading.models import StraddlePosition

        exp_fmt = iso_to_angel(expiry.isoformat()) if expiry else None
        if not exp_fmt:
            self._log("  Cannot execute spread: no expiry format")
            return

        qty = 75 * decision.lots
        broker_svc = BrokerService()

        if decision.strategy == "BEAR_PUT_SPREAD":
            # Buy ATM put (long_strike), Sell OTM put (short_strike)
            long_opt = find_option_token("NIFTY", decision.long_strike, exp_fmt, "PE")
            short_opt = find_option_token("NIFTY", decision.short_strike, exp_fmt, "PE")
            leg_type = "PE"
        else:  # BULL_CALL_SPREAD
            long_opt = find_option_token("NIFTY", decision.long_strike, exp_fmt, "CE")
            short_opt = find_option_token("NIFTY", decision.short_strike, exp_fmt, "CE")
            leg_type = "CE"

        if not long_opt or not short_opt:
            self._log(f"  Cannot execute spread: tokens not found")
            return

        long_sym, long_tok = long_opt
        short_sym, short_tok = short_opt
        long_ltp = svc.fetch_option_ltp(long_sym, long_tok).get("ltp", 0)
        short_ltp = svc.fetch_option_ltp(short_sym, short_tok).get("ltp", 0)

        if long_ltp <= 0 or short_ltp <= 0:
            self._log("  Cannot execute spread: no premium data")
            return

        net_debit = long_ltp - short_ltp  # pay for long, receive from short

        # Execute: buy long leg, sell short leg
        r1 = broker_svc.place_order(
            symbol=long_sym, side="BUY", quantity=qty,
            price=long_ltp, product_type="CARRYFORWARD", exchange="NFO",
            symbol_token=long_tok,
        )
        r2 = broker_svc.place_order(
            symbol=short_sym, side="SELL", quantity=qty,
            price=short_ltp, product_type="CARRYFORWARD", exchange="NFO",
            symbol_token=short_tok,
        )

        # Track as a StraddlePosition (reuse model, strategy field distinguishes)
        pos = StraddlePosition.objects.create(
            underlying="NIFTY",
            strike=decision.long_strike,
            expiry=expiry,
            lot_size=75, lots=decision.lots,
            # CE fields = LONG leg, PE fields = SHORT leg (regardless of option type)
            ce_symbol=long_sym,
            ce_token=long_tok,
            ce_sell_price=long_ltp,
            pe_symbol=short_sym,
            pe_token=short_tok,
            pe_sell_price=short_ltp,
            trade_date=date.today(),
            management_log=[{
                "time": datetime.now().strftime("%H:%M"),
                "action": decision.strategy,
                "nifty": svc.fetch_nifty_spot().get("ltp", 0),
                "pnl_inr": 0,
                "reason": decision.reason[:100],
                "executed": True,
                "spread_type": decision.strategy,
                "long_strike": decision.long_strike,
                "short_strike": decision.short_strike,
                "net_debit": net_debit,
                "max_loss": net_debit * qty,
                "max_profit": (decision.spread_width - net_debit) * qty,
            }],
        )

        self._log(f"  {decision.strategy} #{pos.id} | "
                  f"BUY {leg_type}@{decision.long_strike}={long_ltp:.1f} "
                  f"SELL {leg_type}@{decision.short_strike}={short_ltp:.1f} | "
                  f"Net debit: {net_debit:.1f} pts | "
                  f"Max loss: {net_debit * qty:,.0f} Max profit: {(decision.spread_width - net_debit) * qty:,.0f}",
                  style="SUCCESS")

        self._emit("OPTIONS_SPREAD", {
            "position_id": pos.id, "strategy": decision.strategy,
            "long_strike": decision.long_strike, "short_strike": decision.short_strike,
            "net_debit": net_debit, "max_loss": net_debit * qty,
            "max_profit": (decision.spread_width - net_debit) * qty,
        })

    def _register_single_straddle(self, cfg: dict):
        """Register a single ATM straddle for the given underlying config."""
        from trading.models import StraddlePosition
        from trading.options.data_service import OptionsDataService, find_option_token, find_atm_strike
        from trading.utils.expiry_utils import iso_to_angel

        underlying = cfg["underlying"]
        lot_size = cfg["lot_size"]
        step = cfg["step"]
        min_premium = cfg["min_premium"]

        try:
            svc = OptionsDataService()
            from trading.services.data_service import BrokerClient

            # Get spot price for this underlying
            if underlying == "NIFTY":
                spot_data = svc.fetch_nifty_spot()
            elif underlying == "BANKNIFTY":
                spot_data = svc.fetch_banknifty_spot()
            else:
                return

            vix = svc.fetch_vix()
            spot = spot_data.get("ltp", 0)
            vix_ltp = vix.get("ltp", 0)

            if spot <= 0 or vix_ltp < 12:
                return

            atm = find_atm_strike(spot, step)

            # Find next Tuesday expiry (NIFTY weekly)
            # BANKNIFTY also expires Tuesday as of 2026
            next_expiry = date.today()
            while next_expiry.weekday() != 1:
                next_expiry += timedelta(days=1)

            expiry_angel = iso_to_angel(next_expiry.isoformat())
            if not expiry_angel:
                return

            ce = find_option_token(underlying, atm, expiry_angel, "CE")
            pe = find_option_token(underlying, atm, expiry_angel, "PE")

            if not ce or not pe:
                self._log(f"  {underlying} straddle: tokens not found for {atm} {expiry_angel}", style="WARNING")
                return

            ce_data = svc.fetch_option_ltp(ce[0], ce[1])
            pe_data = svc.fetch_option_ltp(pe[0], pe[1])
            ce_ltp = ce_data.get("ltp", 0)
            pe_ltp = pe_data.get("ltp", 0)
            combined = ce_ltp + pe_ltp

            if combined < min_premium:
                self._log(f"  {underlying} straddle: premium {combined:.1f} < {min_premium}, skipping")
                return

            pos = StraddlePosition.objects.create(
                underlying=underlying, strike=atm, expiry=next_expiry,
                lot_size=lot_size, lots=1,
                ce_symbol=ce[0], ce_token=ce[1], ce_sell_price=ce_ltp,
                pe_symbol=pe[0], pe_token=pe[1], pe_sell_price=pe_ltp,
                trade_date=date.today(),
            )

            self._log(
                f"  AUTO-STRADDLE: #{pos.id} {underlying} {atm} [{next_expiry}] "
                f"CE@{ce_ltp:.1f} PE@{pe_ltp:.1f} = {combined:.1f} pts (₹{combined*lot_size:,.0f})",
                style="SUCCESS",
            )
            self._emit("STRADDLE_REGISTERED", {
                "position_id": pos.id, "strike": atm, "expiry": next_expiry.isoformat(),
                "ce_premium": ce_ltp, "pe_premium": pe_ltp, "combined": combined,
            })

        except Exception as e:
            self._log(f"  Auto-straddle failed: {e}", style="WARNING")

    # ══════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════
    def _refresh_equity_snapshot(self):
        """Fetch live prices for open equity and cache for heartbeat/dashboard."""
        from trading.models import TradeJournal
        from trading.services.ticker_service import ticker_service
        from trading.services.data_service import BrokerClient
        from trading.utils.pnl_utils import compute_equity_pnl

        open_trades = list(TradeJournal.objects.filter(
            trade_date=date.today(), status__in=["EXECUTED", "PAPER"]
        ))
        if not open_trades:
            self._last_equity_snapshot = {"positions": [], "unrealized": 0}
            return

        tokens, tok_map = [], {}
        for t in open_trades:
            tok = ticker_service.get_token(t.symbol)
            if tok:
                tokens.append(tok)
                tok_map[tok] = t

        if not tokens:
            self._log(f"Equity snapshot: {len(open_trades)} trades but no tokens found", style="WARNING")
            return

        try:
            b = BrokerClient.get_instance()
            b.ensure_login()
            fetched = b.market_data_batch({"NSE": tokens}, mode="LTP")
        except Exception as e:
            self._log(f"Equity snapshot batch fetch failed: {e}", style="WARNING")
            return

        positions = []
        unrealized = 0.0
        for item in fetched:
            tok = str(item.get("symbolToken", ""))
            t = tok_map.get(tok)
            if t:
                ltp = float(item.get("ltp", 0))
                pnl = compute_equity_pnl(t.side, t.entry_price, ltp, t.quantity)
                unrealized += pnl
                positions.append({
                    "symbol": t.symbol, "side": t.side, "qty": t.quantity,
                    "entry": t.entry_price, "ltp": ltp, "pnl": round(pnl, 0),
                })

        self._last_equity_snapshot = {"positions": positions, "unrealized": round(unrealized, 0)}
        self._log(f"Equity snapshot: {len(positions)} positions, P&L {unrealized:+,.0f}")

    def _sleep(self, seconds: int):
        end = time.monotonic() + seconds
        while self._running and time.monotonic() < end:
            time.sleep(min(1, end - time.monotonic()))

    def _shutdown(self, signum, frame):
        self._log("\nShutdown signal — finishing current cycle...", style="WARNING")
        self._running = False

    def _log(self, msg: str, style: str = None):
        ts = datetime.now().strftime("%H:%M:%S")
        text = f"[{ts}] {msg}"
        if style == "WARNING":
            self.stdout.write(self.style.WARNING(text))
        elif style == "ERROR":
            self.stdout.write(self.style.ERROR(text))
        elif style == "SUCCESS":
            self.stdout.write(self.style.SUCCESS(text))
        else:
            self.stdout.write(text)

    def _emit(self, event_type: str, data: dict = None):
        """
        Write a structured event to the shared JSONL log.
        Dashboard reads this file for live updates.
        """
        event = {
            "ts": datetime.now().isoformat(),
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": event_type,
            "data": data or {},
        }
        try:
            EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(EVENT_LOG_PATH, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass  # Never block trading for a log write failure
