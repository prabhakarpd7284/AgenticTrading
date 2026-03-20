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
                self._log(f"LLM premarket analysis: {analysis[:200]}...")
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

        # Sleep until next check (every 15 seconds)
        self._sleep(15)

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

        self._emit("EQUITY_SCAN", {
            "watchlist_size": len(self._equity_state.watchlist),
            "signals_found": len(signals),
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

            # Full workflow
            from trading.options.straddle.graph import run_straddle_workflow
            result = run_straddle_workflow(
                position_id=pos.id, underlying=pos.underlying,
                strike=pos.strike, expiry=pos.expiry.isoformat(),
                lot_size=pos.lot_size, lots=pos.lots,
                ce_symbol=pos.ce_symbol, ce_token=pos.ce_token,
                pe_symbol=pos.pe_symbol, pe_token=pos.pe_token,
                ce_sell_price=pos.ce_sell_price, pe_sell_price=pos.pe_sell_price,
            )

            action = (result.get("recommended_action") or {}).get("action", "N/A")
            pnl = (result.get("analysis") or {}).get("net_pnl_inr", 0)
            exec_r = result.get("execution_result") or {}

            self._log(f"  #{pos.id} {pos.underlying} {pos.strike} | {action} | "
                      f"P&L: {pnl:+,.0f} | Exec: {exec_r.get('actions_taken', [])}")
            # Get true P&L including realized roll losses
            pos.refresh_from_db()
            self._emit("STRADDLE_CYCLE", {
                "position_id": pos.id, "underlying": pos.underlying, "strike": pos.strike,
                "action": action, "pnl": pos.total_pnl,
                "unrealized": pos.current_pnl_inr, "realized": pos.realized_pnl,
                "executed": exec_r.get("actions_taken", []),
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
        Auto-register ATM straddles for NIFTY (and BANKNIFTY if no active position).
        Registers one per underlying. Max 2 straddles at a time.
        """
        from trading.models import StraddlePosition
        from trading.options.data_service import OptionsDataService, find_option_token, find_atm_strike
        from trading.utils.expiry_utils import iso_to_angel

        if self._dry_run:
            return

        # Check what's already active per underlying
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
