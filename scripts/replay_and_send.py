"""Replay today's 1m candles through screener, capture signals, send to Telegram with charts + P&L."""
import os, sys, json, time as _time
from pathlib import Path
from datetime import datetime

os.environ['DJANGO_SETTINGS_MODULE'] = 'config.settings'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django
django.setup()

from trading.screener.engine import ScreenerEngine
from trading.screener.strategies import STRATEGIES
from trading.services.data_service import BrokerClient, DataService
from trading.services.ticker_service import ticker_service
from trading.screener.telegram import TelegramAlertService
from dashboard_utils.market_scanner import SCREENER_UNIVERSE
from trading.utils.time_utils import last_trading_day, cap_end_time


def main():
    symbols = list(SCREENER_UNIVERSE)
    enabled = [s for s in STRATEGIES if s.enabled]

    engine = ScreenerEngine(symbols=symbols, strategies=enabled)
    engine.symbol_cooldown_bars = 0  # allow all signals for replay

    signals_found = []
    engine.add_output_handler(lambda sig: signals_found.append(sig))

    broker = BrokerClient.get_instance()
    broker.ensure_login()

    now = datetime.now()
    today = now.strftime('%Y-%m-%d')
    prev_day = last_trading_day(now).isoformat()

    # Step 1: Seed prev-day pivots from cache
    cache_file = Path('/tmp/screener_cache') / f'pivots_{prev_day}.json'
    if cache_file.exists():
        pivot_data = json.loads(cache_file.read_text())
        for sym in symbols:
            if sym in pivot_data:
                pd = pivot_data[sym]
                engine.stores[sym].prev_day_high = pd['high']
                engine.stores[sym].prev_day_low = pd['low']
                engine.stores[sym].prev_day_close = pd['close']
        print(f'Loaded {len(pivot_data)} pivots from cache')

    # Step 2: Seed prev-day 5m candles for indicator warmup (lookback needs history)
    candle_cache_file = Path('/tmp/screener_cache') / f'candles_5m_{prev_day}.json'
    cached_5m = {}
    if candle_cache_file.exists():
        try:
            cached_5m = json.loads(candle_cache_file.read_text())
            print(f'Loaded prev-day 5m from cache ({len(cached_5m)} symbols)')
        except Exception:
            pass

    if not cached_5m:
        # Fetch prev-day 5m for a subset (top 30) to avoid rate limits
        print(f'Fetching prev-day 5m candles...')
        for sym in symbols[:30]:
            token = ticker_service.get_token(sym)
            if not token:
                continue
            try:
                raw = broker.fetch_candles(token, f'{prev_day} 09:15', f'{prev_day} 15:30', 'FIVE_MINUTE')
                if raw:
                    cached_5m[sym] = raw
            except Exception:
                pass
        if cached_5m:
            try:
                candle_cache_file.parent.mkdir(parents=True, exist_ok=True)
                candle_cache_file.write_text(json.dumps(cached_5m))
            except Exception:
                pass
        print(f'  Got {len(cached_5m)} symbols')

    for sym in symbols:
        if sym in cached_5m:
            engine.stores[sym].seed_from_candles(cached_5m[sym], '5m')

    # Initialize indicators for seeded symbols
    for sym in symbols:
        for tf in ['5m', '15m']:
            if engine.stores[sym].bar_count(tf) > 0:
                engine.indicator_engine.update(sym, tf, engine.stores[sym])

    # Step 3: Replay today's 1m candles tick-by-tick (real timestamps)
    print(f'Fetching today 1m candles for {len(symbols)} symbols...')
    end_time = cap_end_time(today, now)
    seeded = 0
    total_bars = 0

    # Store raw 1m candles per symbol for chart rendering
    raw_1m_per_sym = {}

    for sym in symbols:
        token = ticker_service.get_token(sym)
        if not token:
            continue
        try:
            raw_1m = broker.fetch_candles(token, f'{today} 09:15', end_time, 'ONE_MINUTE')
            if raw_1m:
                raw_1m_per_sym[sym] = raw_1m
                for row in raw_1m:
                    ts_str = row[0] if isinstance(row[0], str) else str(row[0])
                    ts = datetime.fromisoformat(ts_str.replace('+05:30', ''))
                    close = float(row[4])
                    vol = int(row[5]) if len(row) > 5 else 0
                    engine.on_tick(sym, close, vol, ts)
                    total_bars += 1
                seeded += 1
        except Exception:
            pass

    print(f'Replayed {total_bars} bars across {seeded} symbols')
    print(f'Signals captured: {len(signals_found)}')

    # Dedup: best per symbol
    best = {}
    for sig in signals_found:
        if sig.symbol not in best or sig.confidence > best[sig.symbol].confidence:
            best[sig.symbol] = sig
    final = sorted(best.values(), key=lambda s: s.timestamp)

    # Step 4: Fetch current LTP
    print('Fetching current LTP...')
    ds = DataService()
    ltp_all = []
    for i in range(0, len(symbols), 50):
        ltp_all.extend(ds.fetch_batch_ltp(symbols[i:i + 50]))
    ltp_map = {x['symbol']: x['ltp'] for x in ltp_all}

    # Step 5: Build performance
    perf = []
    total_pnl = 0
    winners = 0

    for sig in final:
        cur = ltp_map.get(sig.symbol, 0)
        if cur <= 0:
            continue
        if sig.side == 'BUY':
            pnl_pts = cur - sig.entry
            hit_tgt = cur >= sig.target
            hit_sl = cur <= sig.stoploss
        else:
            pnl_pts = sig.entry - cur
            hit_tgt = cur <= sig.target
            hit_sl = cur >= sig.stoploss

        pnl_pct = pnl_pts / sig.entry * 100
        total_pnl += pnl_pts
        if pnl_pts > 0:
            winners += 1

        if hit_tgt:
            status = "\U0001f3af TGT"
        elif hit_sl:
            status = "\U0001f6d1 SL"
        elif pnl_pts > 0:
            status = "\u2705"
        else:
            status = "\u26a0\ufe0f"

        perf.append({
            'sig': sig, 'cur': cur, 'pnl_pts': pnl_pts, 'pnl_pct': pnl_pct,
            'status': status,
        })

    total_trades = len(perf)
    losers = total_trades - winners
    win_rate = winners / total_trades * 100 if total_trades else 0

    # Step 6: Send to Telegram
    tg = TelegramAlertService()
    if not tg.is_configured:
        print('Telegram not configured')
        return

    # Header
    header = (
        f"\U0001f4ca *Session Replay \u2014 {today}*\n"
        f"\U0001f554 {now.strftime('%H:%M')} | {seeded} symbols | {total_bars:,} bars\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\U0001f4c8 *Performance Snapshot*\n"
        f"Signals: *{total_trades}* | W: *{winners}* | L: *{losers}*\n"
        f"Win Rate: *{win_rate:.0f}%*\n"
        f"Total P&L: *{total_pnl:+.1f} pts*\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )
    tg._do_send(header, 'Markdown')
    _time.sleep(0.5)

    # Each signal with chart + P&L
    for p in perf:
        sig = p['sig']

        # Build caption (text under chart)
        pnl_emoji = "\U0001f7e2" if p['pnl_pts'] >= 0 else "\U0001f534"
        arrow = "\U0001f7e2" if sig.side == "BUY" else "\U0001f534"
        side_word = "LONG" if sig.side == "BUY" else "SHORT"
        strat_tag = sig.strategy.upper().replace(" ", "_")
        ts_str = sig.timestamp.strftime('%H:%M') if sig.timestamp else ""

        filled = round(sig.confidence * 8)
        conf_bar = "\u2588" * filled + "\u2591" * (8 - filled)
        rr_stars = "\u2b50" * min(int(sig.risk_reward), 5)

        caption = (
            f"{arrow} *{sig.symbol}* \u2014 {side_word}\n"
            f"`#{strat_tag}` \U0001f554 {ts_str}\n\n"
        )
        if sig.side == "BUY":
            caption += (
                f"\U0001f3af Tgt `{sig.target:,.1f}` (+{sig.target_points:.1f})\n"
                f"\u27a1\ufe0f Entry `{sig.entry:,.1f}`\n"
                f"\U0001f6d1 SL `{sig.stoploss:,.1f}` (-{sig.risk_points:.1f})\n\n"
            )
        else:
            caption += (
                f"\U0001f6d1 SL `{sig.stoploss:,.1f}` (-{sig.risk_points:.1f})\n"
                f"\u27a1\ufe0f Entry `{sig.entry:,.1f}`\n"
                f"\U0001f3af Tgt `{sig.target:,.1f}` (+{sig.target_points:.1f})\n\n"
            )
        caption += (
            f"R:R *{sig.risk_reward:.1f}x* {rr_stars} | Risk *{sig.risk_pct}%*\n"
            f"Conf `{conf_bar}` *{sig.confidence:.0%}*\n\n"
            f"{p['status']} CMP `{p['cur']:,.1f}` | {pnl_emoji} *{p['pnl_pts']:+.1f}pts* ({p['pnl_pct']:+.1f}%)"
        )

        # Generate chart from 5m bars in the engine store
        bars_5m = list(engine.stores[sig.symbol].bars.get('5m', []))
        chart_bytes = sig.render_chart_png(bars_5m)

        if chart_bytes:
            tg.send_chart(chart_bytes, caption=caption, parse_mode="Markdown")
        else:
            # Fallback: text-only
            tg._do_send(sig.format_telegram(), 'Markdown')

        _time.sleep(0.5)
        print(f"  {ts_str} {sig.side:4s} {sig.symbol:15s} {sig.strategy:25s} -> {p['status']} {p['pnl_pts']:+.1f}pts {'[chart]' if chart_bytes else '[text]'}")

    # Summary footer
    if perf:
        _time.sleep(0.3)
        lines = ["\U0001f4cb *Signal Summary*\n"]
        for p in sorted(perf, key=lambda x: x['pnl_pts'], reverse=True):
            sig = p['sig']
            arr = "\U0001f7e2" if p['pnl_pts'] >= 0 else "\U0001f534"
            ts_str = sig.timestamp.strftime('%H:%M')
            lines.append(
                f"{arr} `{ts_str}` *{sig.symbol}* {sig.side} "
                f"`{sig.entry:.0f}\u2192{p['cur']:.0f}` *{p['pnl_pts']:+.1f}* {p['status']}"
            )
        lines.append(f"\n*Total: {total_pnl:+.1f} pts | {win_rate:.0f}% win rate*")
        tg._do_send('\n'.join(lines), 'Markdown')

    print(f'\nDone - {len(perf)} signals sent to Telegram')


if __name__ == '__main__':
    main()
