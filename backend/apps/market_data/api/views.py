from dataclasses import asdict
from decimal import Decimal

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.market_data.models import Symbol
from apps.market_data.services.data_port import DefaultMarketData
from apps.market_data.services.pulse_service import build_pulse
from apps.market_data.services.rotation_service import build_rotation
from apps.market_data.services.setup_service import build_setup
from apps.market_data.services.shortlist_service import build_shortlist
from apps.market_data.services.swing_scanner_service import build_swing_scanner
from apps.market_data.services.ok_backtest_service import build_ok_backtest
from apps.market_data.services.basket_service import build_basket_status
from apps.market_data.services.depth_imbalance import build_depth_imbalance
from apps.market_data.services.fii_dii_flow import build_fii_dii_flow
from apps.market_data.services.first_5min import build_first_5min
from apps.market_data.services.gap_fill import build_gap_fill
from apps.market_data.services.liquidity_map import build_liquidity_map
from apps.market_data.services.news_shock import (
    build_news_shock, pause_symbol, unpause_symbol,
)
from apps.market_data.services.orb_failure import build_orb_failure
from apps.market_data.services.orb_tracker import build_orb
from apps.market_data.services.second_5min import build_second_5min
from apps.market_data.services.intraday_sector_heatmap import build_intraday_sector_heatmap
from apps.market_data.services.lunchtime_reset import build_lunchtime_reset
from apps.market_data.services.market_health import build_market_health
from apps.market_data.services.stop_hunt import build_stop_hunt
from apps.market_data.services.tick_strip import build_tick_strip
from apps.market_data.services.sector_dispersion import build_sector_dispersion
from apps.market_data.services.sector_rrg import build_sector_rrg
from apps.market_data.services.stock_rrg import build_stock_rrg
from apps.market_data.services.tape_speed import build_tape_speed
from apps.market_data.services.vol_regime_strip import build_vol_regime
from apps.market_data.services.vwap_bands import build_vwap_bands


class SymbolSearchView(APIView):
    def get(self, request):
        q = request.query_params.get("q", "")
        rows = Symbol.objects.filter(tradingsymbol__icontains=q)[:25].values(
            "id", "exchange", "token", "tradingsymbol", "name", "lot_size"
        )
        return Response(list(rows))


class CandleView(APIView):
    def get(self, request):
        symbol = request.query_params.get("symbol", "")
        interval = request.query_params.get("interval", "5m")
        n = int(request.query_params.get("n", 200))
        port = DefaultMarketData(request.tenant.id)
        return Response(port.candles(symbol, interval, n))


class MarketPulseView(APIView):
    """GET /api/v1/market/pulse/
    The "What's Happening Today" briefing — Stages 1+2 of The Cascade.

    Returns a single JSON payload the frontend renders as a live
    macro/sector read so the operator can decide *whether* it's a day
    to trade before the agents even run.  Cached server-side for 30s.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        force = request.query_params.get("force") in ("1", "true", "yes")
        payload = build_pulse(force=force)
        return Response(asdict(payload))


class SectorRotationView(APIView):
    """GET /api/v1/market/rotation/
    Stage 3 — sector drill-in.

    Ranks NSE sector indices by % move, then surfaces the top 3 leaders and
    bottom 3 laggards per sector (from a curated liquid subset of NIFTY50).
    Cached 60s.  Use ``?force=1`` to bypass the cache during dev.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        force = request.query_params.get("force") in ("1", "true", "yes")
        payload = build_rotation(force=force)
        return Response(asdict(payload))


class ShortlistView(APIView):
    """GET /api/v1/market-data/shortlist/
    Stage 4 — narrow the hot sectors (Stage 3) down to 10-15 tradeable names.

    The response is already sorted by confluence score (0-100).  Soft screens
    (ATR tier, rel volume, 52w position, sector alignment) surface as
    ``reasons`` so the operator can audit *why* a name scored where it did.
    Candidates that fail a hard filter land in ``filtered_out`` — not silently
    dropped, so the desk can sanity-check the gates.
    Cached 300s.  Use ``?force=1`` to bypass the cache.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        force = request.query_params.get("force") in ("1", "true", "yes")
        payload = build_shortlist(force=force, rotation_force=force)
        return Response(asdict(payload))


class SetupPreviewView(APIView):
    """GET /api/v1/market-data/setup/?symbol=TCS&side=BUY
    Stage 5 — preview a deterministic plan + the full @RiskGuard breakdown.

    Synchronous (no Celery) so the desk can click a shortlist row and get an
    instant "would-this-clear?" verdict.  Reuses the production
    ``trading.services.risk_engine.validate_trade`` for the authoritative
    overall decision and runs each criterion independently so the UI shows
    *all ten gates*, not just the first failure.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        symbol = (request.query_params.get("symbol") or "").strip().upper()
        side = (request.query_params.get("side") or "BUY").strip().upper()
        if not symbol:
            return Response(
                {"detail": "Query param 'symbol' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if side not in ("BUY", "SELL"):
            return Response(
                {"detail": f"Invalid side '{side}'. Must be BUY or SELL."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Pull the desk's real capital + day P&L + open-position count from
        # the tenant's primary portfolio.  Falls back to env defaults if the
        # tenant hasn't created one yet (paper-mode greenfield).
        capital, daily_loss, open_positions = _portfolio_context(request)

        # Optional explicit overrides (useful for "what-if" exploration in
        # the UI without mutating the underlying portfolio).
        capital = _maybe_float(request.query_params.get("capital"), capital)
        daily_loss = _maybe_float(request.query_params.get("daily_loss"), daily_loss)
        open_positions = int(_maybe_float(
            request.query_params.get("open_positions"), open_positions))

        payload = build_setup(
            symbol=symbol,
            side=side,
            capital=capital,
            daily_loss=daily_loss,
            open_positions=open_positions,
            tenant_id=getattr(getattr(request, "tenant", None), "id", None),
        )
        return Response(asdict(payload))


class SwingScannerView(APIView):
    """GET /api/v1/market-data/swing-scanner/
    Oliver Kell Cycle of Price Action — daily/weekly swing scanner.

    Scans NIFTY 100 for 8 cycle phases (RE, WP, EC, BB, EX, WD, EC_BEAR,
    BB_BEAR) with multi-timeframe trend confirmation.  Returns stocks with
    active phases, summary metrics, and phase distribution.

    Cached 600s.  Use ``?force=1`` to bypass the cache.
    Optional: ``?symbols=RELIANCE,TCS`` to scan specific symbols.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        force = request.query_params.get("force") in ("1", "true", "yes")
        symbols_raw = request.query_params.get("symbols", "").strip()
        symbols = (
            [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
            if symbols_raw
            else None
        )
        payload = build_swing_scanner(force=force, symbols=symbols)
        return Response(asdict(payload))


class OKBacktestView(APIView):
    """GET /api/v1/market-data/ok-backtest/
    Oliver Kell cycle backtest — daily or intraday multi-TF.

    Query params:
        mode: "daily" (swing) or "intraday" (3m/5m/15m grid)
        from_date: Start date YYYY-MM-DD (default: 30 days ago)
        to_date: End date YYYY-MM-DD (default: today)
        symbols: Comma-separated (default: NIFTY 50)
        capital: Starting capital (default: 500000)
        force: "1" to bypass cache

    Synchronous — daily ~25s, intraday ~90s.  Cached 1 hour.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        mode = request.query_params.get("mode", "daily")
        force = request.query_params.get("force") in ("1", "true", "yes")
        from_date = request.query_params.get("from_date")
        to_date = request.query_params.get("to_date")
        capital = _maybe_float(request.query_params.get("capital"), None)
        symbols_raw = request.query_params.get("symbols", "").strip()
        symbols = (
            [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
            if symbols_raw
            else None
        )
        payload = build_ok_backtest(
            mode=mode, from_date=from_date, to_date=to_date,
            symbols=symbols, capital=capital, force=force,
        )
        return Response(asdict(payload))


class BasketView(APIView):
    """GET /api/v1/market-data/basket/
    Morning basket — mood assessment + equity/option signals.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        force = request.query_params.get("force") in ("1", "true", "yes")
        payload = build_basket_status(force=force)
        return Response(payload)


class PyramidView(APIView):
    """GET /api/v1/market-data/pyramid/
    Run pyramiding strategy simulation on option candles.

    Query params:
      strike (int, required), type (CE/PE), underlying (NIFTY/BANKNIFTY),
      expiry (DDMMMYY), date (YYYY-MM-DD), interval (FIVE_MINUTE),
      capital, risk_pct, profit_risk, max_pyramids, lot_size,
      dry_run (bool), telegram (bool)
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            return self._run(request)
        except ImportError as e:
            return Response(
                {"error": "dependency_missing", "detail": str(e)}, status=503
            )
        except Exception as e:
            import traceback
            return Response(
                {"error": "simulation_failed", "detail": str(e),
                 "trace": traceback.format_exc()[-500:]},
                status=500,
            )

    def _run(self, request):
        from trading.pyramid.strategy import (
            Candle, PyramidConfig, run_pyramid_with_chart_data,
        )

        strike = request.query_params.get("strike")
        if not strike:
            return Response({"error": "strike is required"}, status=400)
        strike = int(strike)

        opt_type = request.query_params.get("type", "CE").upper()
        underlying = request.query_params.get("underlying", "NIFTY").upper()
        interval = request.query_params.get("interval", "FIVE_MINUTE")
        dry_run = request.query_params.get("dry_run", "false").lower() == "true"
        send_tg = request.query_params.get("telegram", "false").lower() == "true"

        config = PyramidConfig(
            lot_size=int(request.query_params.get("lot_size", 25)),
            initial_capital=float(request.query_params.get("capital", 100000)),
            initial_risk_pct=float(request.query_params.get("risk_pct", 2.0)),
            profit_risk_pct=float(request.query_params.get("profit_risk", 0.80)),
            max_pyramids=int(request.query_params.get("max_pyramids", 5)),
        )

        # Resolve expiry
        expiry_str = request.query_params.get("expiry")
        if not expiry_str:
            from trading.utils.expiry_utils import next_expiry_date, iso_to_angel
            exp_date = next_expiry_date(underlying)
            expiry_str = iso_to_angel(exp_date.isoformat()) if exp_date else None
            if not expiry_str:
                return Response({"error": "Cannot determine expiry"}, status=400)

        # Resolve date
        candle_date = request.query_params.get("date")
        if not candle_date:
            from trading.utils.time_utils import get_candle_date_range
            candle_date = get_candle_date_range()[0].isoformat()

        symbol_label = f"{underlying} {strike} {opt_type} (exp {expiry_str})"

        if dry_run:
            candles = self._sample_candles()
            actual_date = candle_date
        else:
            candles = self._fetch_candles(
                underlying, strike, expiry_str, opt_type, candle_date, interval,
            )
            actual_date = candle_date
            if candles:
                ts = candles[0].timestamp
                actual_date = ts[:10] if len(ts) >= 10 else candle_date

        if not candles:
            return Response({
                "error": "No candle data",
                "detail": (
                    f"No option candles found for {underlying} {strike} {opt_type} "
                    f"(exp {expiry_str}) on {candle_date} or recent trading days. "
                    f"The option may not have been listed yet, or the strike is too far OTM."
                ),
            }, status=404)

        data = run_pyramid_with_chart_data(candles, symbol=symbol_label, config=config)
        data["config"] = {
            "strike": strike, "type": opt_type, "underlying": underlying,
            "expiry": expiry_str, "date": actual_date, "interval": interval,
            "capital": config.initial_capital, "risk_pct": config.initial_risk_pct,
            "profit_risk": config.profit_risk_pct, "max_pyramids": config.max_pyramids,
            "lot_size": config.lot_size, "dry_run": dry_run,
        }

        # Telegram report
        telegram_sent = False
        if send_tg:
            from trading.pyramid.telegram import send_pyramid_report
            telegram_sent = send_pyramid_report(data)
        data["telegram_sent"] = telegram_sent

        return Response(data)

    @staticmethod
    def _fetch_candles(underlying, strike, expiry_str, opt_type, candle_date, interval):
        """Fetch option candles, walking back up to 15 trading days.

        Old loop was 6 calendar days which often landed on a weekend twice
        and only effectively probed ~3 trading days — too narrow when the
        user picks a historical date via cockpit time-travel that predates
        the strike's listing.
        """
        from datetime import date as dt_date, timedelta
        from trading.options.data_service import find_option_token
        from trading.services.data_service import BrokerClient
        from trading.pyramid.strategy import Candle
        from trading.utils.time_utils import cap_end_time

        result = find_option_token(underlying, strike, expiry_str, opt_type)
        if not result:
            return []
        symbol, token = result
        broker = BrokerClient.get_instance()
        broker.ensure_login()

        d = dt_date.fromisoformat(candle_date)
        trading_days_tried = 0
        while trading_days_tried < 15:
            if d.weekday() >= 5:
                d -= timedelta(days=1)
                continue
            ds = d.isoformat()
            end_str = cap_end_time(ds)
            try:
                raw = broker.fetch_candles(
                    symbol_token=token, start=f"{ds} 09:15",
                    end=end_str, interval=interval, exchange="NFO",
                )
            except Exception:  # noqa: BLE001 — transient broker hiccup; try next day
                raw = None
            if raw and len(raw) > 5:
                return [Candle.from_raw(r) for r in raw]
            trading_days_tried += 1
            d -= timedelta(days=1)
        return []

    @staticmethod
    def _sample_candles():
        import random
        from datetime import datetime
        from trading.pyramid.strategy import Candle

        candles = []
        price = 180.0
        base_time = datetime(2026, 5, 5, 9, 15)
        random.seed(77)
        phases = {
            (0, 15): (0.1, 0.8), (15, 25): (0.8, 1.0), (25, 45): (1.2, 0.7),
            (45, 55): (0.6, 0.5), (55, 65): (1.5, 0.9), (65, 75): (-0.3, 1.2),
        }
        for i in range(75):
            total_min = 15 + i * 5
            ts = base_time.replace(hour=9 + total_min // 60, minute=total_min % 60)
            if ts.hour >= 15 and ts.minute > 30:
                break
            drift, vol = 0.1, 0.8
            for (s, e), (d, v) in phases.items():
                if s <= i < e:
                    drift, vol = d, v
                    break
            open_p = price
            close_p = open_p + drift + random.gauss(0, vol)
            high_p = max(open_p, close_p) + abs(random.gauss(0, vol * 0.6))
            low_p = min(open_p, close_p) - abs(random.gauss(0, vol * 0.5))
            candles.append(Candle(
                timestamp=ts.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                open=round(open_p, 2), high=round(high_p, 2),
                low=round(low_p, 2), close=round(close_p, 2),
                volume=random.randint(8000, 60000),
            ))
            price = close_p
        return candles


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _portfolio_context(request) -> tuple[float, float, int]:
    """Pull capital / day_pnl / open positions from the tenant's primary
    portfolio.  Defaults to 500k / 0 / 0 when nothing has been seeded — keeps
    the preview useful in dev / fresh tenants."""
    import os
    default_cap = float(os.getenv("DEFAULT_CAPITAL", "500000"))
    tenant = getattr(request, "tenant", None)
    if tenant is None:
        return default_cap, 0.0, 0
    try:
        from apps.portfolio.models import Portfolio, Position
    except Exception:  # noqa: BLE001
        return default_cap, 0.0, 0
    pf = (
        Portfolio.objects.filter(tenant=tenant).order_by("name").first()
    )
    if pf is None:
        return default_cap, 0.0, 0
    capital = float(pf.capital or default_cap) or default_cap
    # day_pnl is signed (negative = loss); the risk engine wants a positive
    # "loss-so-far" magnitude.
    day_pnl = float(pf.day_pnl or 0)
    daily_loss = abs(day_pnl) if day_pnl < 0 else 0.0
    open_count = Position.objects.filter(
        tenant=tenant, portfolio=pf, status=Position.Status.OPEN,
    ).count()
    return capital, daily_loss, open_count


def _maybe_float(raw, fallback):
    if raw is None or raw == "":
        return fallback
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


# Touch Decimal so the linter sees it intentional (used implicitly when
# Portfolio fields are coerced to float above).
_ = Decimal


class LiquidityMapView(APIView):
    """GET /api/v1/market-data/liquidity/

    Live bid/ask/spread + historical slippage per held or watchlisted symbol.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(build_liquidity_map(getattr(request, "tenant", None)))


class ORBView(APIView):
    """GET /api/v1/market-data/orb/

    Opening-range high/low + breakout state + retests per watchlist symbol.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(build_orb(getattr(request, "tenant", None)))


class VWAPBandsView(APIView):
    """GET /api/v1/market-data/vwap-bands/?symbol=HDFCBANK

    Intraday anchored VWAP + ±1σ / ±2σ bands. Returns full bar series so
    the React chart can plot the bands directly.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        sym = request.query_params.get("symbol") or ""
        return Response(build_vwap_bands(sym))


class First5MinView(APIView):
    """GET /api/v1/market-data/first-5min/

    Classifies every watchlist symbol's 09:15-09:20 IST 5-min bar.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(build_first_5min(getattr(request, "tenant", None)))


class ORBFailureView(APIView):
    """GET /api/v1/market-data/orb-failure/

    Surfaces watchlist symbols whose OR breakout has been retested >=2
    times within 30 min, with the empirical reversal probability so a
    fade trade can be sized.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(build_orb_failure(getattr(request, "tenant", None)))


class GapFillView(APIView):
    """GET /api/v1/market-data/gap-fill/"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(build_gap_fill(getattr(request, "tenant", None)))


class Second5MinView(APIView):
    """GET /api/v1/market-data/second-5min/"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(build_second_5min(getattr(request, "tenant", None)))


class VolRegimeView(APIView):
    """GET /api/v1/market-data/vol-regime/?symbol=HDFCBANK"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(build_vol_regime(request.query_params.get("symbol") or ""))


class SectorRRGView(APIView):
    """GET /api/v1/market-data/sector-rrg/?weekly=1"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        weekly = request.query_params.get("weekly", "").lower() in ("1", "true", "yes")
        return Response(build_sector_rrg(getattr(request, "tenant", None), weekly=weekly))


class SectorDispersionView(APIView):
    """GET /api/v1/market-data/sector-dispersion/"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(build_sector_dispersion(getattr(request, "tenant", None)))


class TapeSpeedView(APIView):
    """GET /api/v1/market-data/tape-speed/?symbol=HDFCBANK"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(build_tape_speed(request.query_params.get("symbol") or ""))


class NewsShockView(APIView):
    """GET /api/v1/market-data/news-shocks/"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(build_news_shock(getattr(request, "tenant", None)))


class NewsShockPauseView(APIView):
    """POST /api/v1/market-data/news-shocks/pause/  {symbol, minutes?, reason?}"""
    permission_classes = [IsAuthenticated]
    def post(self, request):
        body = request.data or {}
        try:
            minutes = int(body.get("minutes") or 15)
        except (TypeError, ValueError):
            minutes = 15
        return Response(pause_symbol(
            body.get("symbol") or "",
            minutes=minutes,
            reason=body.get("reason") or "manual",
        ))


class NewsShockUnpauseView(APIView):
    """POST /api/v1/market-data/news-shocks/unpause/  {symbol}"""
    permission_classes = [IsAuthenticated]
    def post(self, request):
        return Response(unpause_symbol((request.data or {}).get("symbol") or ""))


class FIIDIIFlowView(APIView):
    """GET /api/v1/market-data/fii-dii-flow/?days=30"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        try:
            days = int(request.query_params.get("days", 30) or 30)
        except ValueError:
            days = 30
        return Response(build_fii_dii_flow(getattr(request, "tenant", None), days=days))


class DepthImbalanceView(APIView):
    """GET /api/v1/market-data/depth-imbalance/"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(build_depth_imbalance(getattr(request, "tenant", None)))


class StockRRGView(APIView):
    """GET /api/v1/market-data/stock-rrg/?symbols=A,B,C"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        raw = request.query_params.get("symbols") or ""
        syms = [s.strip().upper() for s in raw.split(",") if s.strip()] if raw else None
        return Response(build_stock_rrg(symbols=syms))


class IntradaySectorHeatmapView(APIView):
    """GET /api/v1/market-data/intraday-sector-heatmap/"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(build_intraday_sector_heatmap(getattr(request, "tenant", None)))


class StopHuntView(APIView):
    """GET /api/v1/market-data/stop-hunt/"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(build_stop_hunt(getattr(request, "tenant", None)))


class MarketHealthView(APIView):
    """GET /api/v1/market-data/market-health/"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(build_market_health(getattr(request, "tenant", None)))


class LunchtimeResetView(APIView):
    """GET /api/v1/market-data/lunchtime-reset/?underlying=NIFTY"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(build_lunchtime_reset(
            getattr(request, "tenant", None),
            underlying=request.query_params.get("underlying") or "NIFTY",
        ))


class TickStripView(APIView):
    """GET /api/v1/market-data/tick-strip/"""
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(build_tick_strip(getattr(request, "tenant", None)))
