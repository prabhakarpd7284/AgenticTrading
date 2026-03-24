"""
Agentic Trading — Command Center Dashboard.

The ONE screen a full-time trader opens at 9:00 AM and never leaves until 3:30 PM.
AI handles scanning, planning, risk-checking, and executing.
You monitor, override when needed, and go home.

Run:  streamlit run dashboard.py
"""
import os
import sys
import json
import time
from datetime import date, datetime, timedelta
from typing import Optional

from streamlit_autorefresh import st_autorefresh

# ── Django bootstrap ──
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django
django.setup()

import streamlit as st
import pandas as pd
from django.utils import timezone

from trading.models import (
    TradeJournal, StrategyDoc, PortfolioSnapshot, AuditLog,
    StraddlePosition, TraderNote, SystemControl, WatchlistEntry,
)
from trading.services.risk_engine import validate_trade
from trading.services.broker_service import BrokerService
from trading.options.data_service import OptionsDataService, find_option_token, find_atm_strike
from trading.options.straddle.analyzer import analyze_straddle

from dashboard_utils.data_layer import (
    get_market_pulse, compute_combined_pnl, get_risk_utilization,
    compute_risk_status, get_active_alerts, get_ai_activity_feed,
    get_active_positions, get_journal_analytics,
    is_ai_paused, pause_ai_trading, resume_ai_trading,
    is_market_open, TRADING_MODE,
)
from dashboard_utils.components import (
    render_tradingview_chart, render_tradingview_mini,
    render_risk_gauge, render_payoff_diagram, render_premium_decay_chart,
    render_symbol_pnl_chart, render_alerts,
)
from dashboard_utils.market_scanner import (
    NIFTY_50_SYMBOLS, SECTOR_MAP, fetch_nifty50_ltp, rank_opportunities,
)


# ── Page config ──
st.set_page_config(
    page_title="Agentic Trading — Command Center",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared services (singleton per session) ──
if "options_svc" not in st.session_state:
    st.session_state["options_svc"] = OptionsDataService()
_options_svc: OptionsDataService = st.session_state["options_svc"]


# ──────────────────────────────────────────────
# SIDEBAR — Mini Command Center + Navigation
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Agentic Trading")

    # Mode + Market status
    mode_color = "🟢" if TRADING_MODE == "paper" else "🔴"
    market_status = "🟢 OPEN" if is_market_open() else "🔴 CLOSED"
    st.markdown(f"**Mode:** {mode_color} `{TRADING_MODE.upper()}`  |  **Market:** {market_status}")

    # AI status
    if is_ai_paused():
        st.error("⏸ AI TRADING PAUSED")

    # Quick portfolio
    try:
        snap = PortfolioSnapshot.objects.latest()
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Capital", f"₹{snap.capital:,.0f}")
        c2.metric("Available", f"₹{snap.available_cash:,.0f}")

        pnl_data = compute_combined_pnl()
        c3, c4 = st.columns(2)
        c3.metric("Today P&L", f"₹{pnl_data['total_pnl']:+,.0f}")
        c4.metric("Positions", snap.open_positions)
    except PortfolioSnapshot.DoesNotExist:
        st.info("No portfolio. Go to Settings to initialize.")

    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "Command Center",
            "Intraday Agent",
            "Market Scanner",
            "Screener",
            "Trade Workflow",
            "Straddle Console",
            "Journal & Analytics",
            "Risk Control",
            "Backtest",
            "Settings",
        ],
        index=0,
    )


# ══════════════════════════════════════════════
# PAGE 1: COMMAND CENTER
# ══════════════════════════════════════════════
if page == "Command Center":
    st.header("Command Center")

    # ── Auto-refresh controls ──
    cc_r1, cc_r2, cc_r3 = st.columns([3, 1, 1])
    cc_r1.caption(f"{datetime.now().strftime('%A, %d %B %Y  %H:%M:%S')}  |  "
                  f"{'Market Open' if is_market_open() else 'Market Closed'}")
    cc_auto = cc_r2.checkbox("Auto-refresh", value=is_market_open(), key="cc_auto_refresh")
    cc_interval = cc_r3.selectbox("Every", [15, 30, 60, 120], index=1, key="cc_interval",
                                   format_func=lambda x: f"{x}s")

    if cc_auto:
        st_autorefresh(interval=cc_interval * 1000, key="cc_autorefresh")

    # ── Market Pulse ──
    try:
        pulse = get_market_pulse(_options_svc)

        nifty = pulse.get("nifty", {})
        banknifty = pulse.get("banknifty", {})
        vix = pulse.get("vix", {})

        m1, m2, m3, m4 = st.columns(4)

        nifty_chg = nifty.get("ltp", 0) - nifty.get("prev_close", 0)
        m1.metric("NIFTY 50", f"{nifty.get('ltp', 0):,.1f}", f"{nifty_chg:+.0f} pts")

        bn_chg = banknifty.get("ltp", 0) - banknifty.get("prev_close", 0)
        m2.metric("BANK NIFTY", f"{banknifty.get('ltp', 0):,.1f}", f"{bn_chg:+.0f} pts")

        vix_chg = vix.get("ltp", 0) - vix.get("prev_close", 0)
        m3.metric("INDIA VIX", f"{vix.get('ltp', 0):.2f}", f"{vix_chg:+.2f}")

        # VIX phase
        vix_ltp = vix.get("ltp", 0)
        vix_phase = "CALM" if vix_ltp < 15 else "ELEVATED" if vix_ltp < 22 else "SPIKE"
        m4.metric("VIX Phase", vix_phase)

    except Exception as e:
        st.warning(f"Market data unavailable: {e}")
        pulse = {}

    st.markdown("---")

    # ── Today's P&L + Risk Utilization ──
    pnl = compute_combined_pnl()
    risk = get_risk_utilization()
    risk_status = compute_risk_status(risk)

    p1, p2, p3, p4, p5 = st.columns(5)

    pnl_color = "normal" if pnl["total_pnl"] >= 0 else "inverse"
    p1.metric("Total P&L", f"₹{pnl['total_pnl']:+,.0f}",
              f"Eq: {pnl['equity_pnl']:+,.0f} | Opt: {pnl['options_pnl']:+,.0f}")

    risk_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
    p2.metric("Risk Status", f"{risk_emoji.get(risk_status, '⚪')} {risk_status}")
    p3.metric("Daily Loss", f"{risk['daily_loss_pct']:.0f}% / {risk['daily_loss_limit_pct']}%")
    p4.metric("Capital Used", f"{risk['capital_deployed_pct']:.1f}%")
    p5.metric("Positions", f"{risk['open_positions']} / {risk['max_open_positions']}")

    st.markdown("---")

    # ── Active Positions + Alerts side by side ──
    col_pos, col_alerts = st.columns([3, 2])

    with col_pos:
        st.subheader("Active Positions")
        positions = get_active_positions()

        if positions["equity"]:
            eq_rows = []
            for p in positions["equity"]:
                eq_rows.append({
                    "Symbol": p["symbol"],
                    "Side": p["side"],
                    "Entry": f"₹{p['entry_price']:.2f}",
                    "Qty": p["quantity"],
                    "P&L": f"₹{p['pnl']:+,.0f}" if p.get("pnl") else "open",
                    "Status": p["status"],
                })
            st.dataframe(pd.DataFrame(eq_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No equity positions today")

        if positions["options"]:
            opt_rows = []
            for p in positions["options"]:
                opt_rows.append({
                    "Position": f"{p['underlying']} {p['strike']}",
                    "Expiry": p["expiry"],
                    "DTE": p["dte"],
                    "Delta": f"{p['net_delta']:+.2f}",
                    "P&L": f"₹{p['pnl_inr']:+,.0f}",
                    "Status": p["status"],
                })
            st.dataframe(pd.DataFrame(opt_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No active straddle positions")

    with col_alerts:
        st.subheader("Alerts")
        alerts = get_active_alerts(pulse if pulse else None, risk=risk)
        render_alerts(alerts)

    st.markdown("---")

    # ── V2 System Status: Live Event Stream + Hourly Snapshots ──
    tab_events, tab_hourly, tab_options_decision = st.tabs([
        "Live Events", "Hourly Snapshots", "Options Decision"
    ])

    with tab_events:
        # Read event log for today's activity
        import json as _json
        try:
            _event_log = open("logs/trading_day_events.jsonl")
            events = [_json.loads(l) for l in _event_log if l.strip()]
            _event_log.close()
            non_hb = [e for e in events if e["type"] != "HEARTBEAT"]

            if non_hb:
                # Summary counts
                from collections import Counter as _Counter
                type_counts = _Counter(e["type"] for e in non_hb)
                ev_cols = st.columns(6)
                for i, (etype, count) in enumerate(type_counts.most_common(6)):
                    emoji = {"TRADE": "📈", "EXIT": "📉", "EQUITY_SCAN": "🔍",
                             "STRADDLE_CYCLE": "🔄", "OPTIONS_DECISION": "🎯",
                             "SIGNAL_REJECTED": "🚫", "HOURLY_SNAPSHOT": "📊",
                             "DAY_START": "🌅", "DAY_REVIEW": "📋"}.get(etype, "📋")
                    ev_cols[i % 6].metric(etype, f"{emoji} {count}")

                # Event table
                ev_rows = []
                for e in reversed(non_hb[-20:]):
                    d = e.get("data", {})
                    detail = ""
                    if e["type"] == "TRADE":
                        detail = f"{d.get('side','')} {d.get('symbol','')} @{d.get('entry',0):.0f} {d.get('setup','')}"
                    elif e["type"] == "EXIT":
                        detail = f"{d.get('symbol','')} {d.get('reason','')} P&L:{d.get('pnl',0):+,.0f}"
                    elif e["type"] == "EQUITY_SCAN":
                        detail = f"{d.get('signals_found',0)} signals | scr:{d.get('screener_signals',0)} | open:{d.get('open_positions',0)}"
                    elif e["type"] == "STRADDLE_CYCLE":
                        detail = f"#{d.get('position_id','')} {d.get('action','')} P&L:{d.get('pnl',0):+,.0f}"
                    elif e["type"] == "OPTIONS_DECISION":
                        detail = f"{d.get('strategy','')} VIX:{d.get('vix',0):.0f} DTE:{d.get('dte','')}"
                    elif e["type"] == "OPTIONS_SPREAD":
                        detail = f"{d.get('strategy','')} {d.get('long_strike','')}/{d.get('short_strike','')} max_loss:{d.get('max_loss',0):,.0f}"
                    elif e["type"] == "HOURLY_SNAPSHOT":
                        detail = f"NIFTY:{d.get('nifty',0):,.0f} Total:{d.get('total_pnl',0):+,.0f}"
                    elif e["type"] == "SIGNAL_REJECTED":
                        detail = f"{d.get('symbol','')} {d.get('setup','')} — {d.get('reason','')[:40]}"
                    else:
                        detail = str(d)[:60]

                    ev_rows.append({
                        "Time": e["time"],
                        "Event": e["type"],
                        "Detail": detail,
                    })
                st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No events yet today. Start run_trading_day.")
        except FileNotFoundError:
            st.info("No event log. Run: python manage.py run_trading_day")

    with tab_hourly:
        try:
            _event_log2 = open("logs/trading_day_events.jsonl")
            events2 = [_json.loads(l) for l in _event_log2 if l.strip()]
            _event_log2.close()
            snapshots = [e for e in events2 if e["type"] == "HOURLY_SNAPSHOT"]

            if snapshots:
                snap_rows = []
                for s in snapshots:
                    d = s["data"]
                    snap_rows.append({
                        "Time": d.get("time", s["time"]),
                        "NIFTY": f"{d.get('nifty',0):,.0f}",
                        "Eq Open": d.get("equity_open", 0),
                        "Eq Realized": f"₹{d.get('equity_realized',0):+,.0f}",
                        "Eq Unrealized": f"₹{d.get('equity_unrealized',0):+,.0f}",
                        "Opt Active": d.get("options_active", 0),
                        "Opt P&L": f"₹{d.get('options_realized',0)+d.get('options_unrealized',0):+,.0f}",
                        "Total": f"₹{d.get('total_pnl',0):+,.0f}",
                    })
                st.dataframe(pd.DataFrame(snap_rows), use_container_width=True, hide_index=True)

                # P&L progression chart
                if len(snap_rows) >= 2:
                    chart_data = [{"Time": s["data"].get("time", ""), "P&L": s["data"].get("total_pnl", 0)}
                                  for s in snapshots]
                    st.line_chart(pd.DataFrame(chart_data).set_index("Time"), height=200)
            else:
                st.info("No hourly snapshots yet. They post at 10:00, 11:00, ... 15:00.")
        except FileNotFoundError:
            st.info("No event log.")

    with tab_options_decision:
        st.subheader("Today's Adaptive Options Decision")
        try:
            _event_log3 = open("logs/trading_day_events.jsonl")
            events3 = [_json.loads(l) for l in _event_log3 if l.strip()]
            _event_log3.close()

            opt_decisions = [e for e in events3 if e["type"] == "OPTIONS_DECISION"]
            opt_spreads = [e for e in events3 if e["type"] == "OPTIONS_SPREAD"]
            opt_straddles = [e for e in events3 if e["type"] == "STRADDLE_REGISTERED"]

            if opt_decisions:
                for od in opt_decisions:
                    d = od["data"]
                    strategy = d.get("strategy", "?")
                    emoji = {"STRADDLE": "📊", "BEAR_PUT_SPREAD": "📉", "BULL_CALL_SPREAD": "📈",
                             "0DTE_THETA": "⏱", "SKIP": "⏭"}.get(strategy, "📋")
                    st.markdown(f"### {emoji} {strategy}")
                    oc1, oc2, oc3 = st.columns(3)
                    oc1.metric("VIX", f"{d.get('vix',0):.1f}")
                    oc2.metric("DTE", d.get("dte", "?"))
                    oc3.metric("NIFTY", f"{d.get('spot',0):,.0f}")
                    st.caption(d.get("reason", ""))

            if opt_spreads:
                for os_evt in opt_spreads:
                    d = os_evt["data"]
                    st.success(
                        f"Spread #{d.get('position_id','')} | "
                        f"{d.get('strategy','')} {d.get('long_strike','')}/{d.get('short_strike','')} | "
                        f"Max loss: ₹{d.get('max_loss',0):,.0f} | Max profit: ₹{d.get('max_profit',0):,.0f}"
                    )

            if opt_straddles:
                for os_evt in opt_straddles:
                    d = os_evt["data"]
                    st.info(
                        f"Straddle #{d.get('position_id','')} @{d.get('strike','')} | "
                        f"Premium: {d.get('combined',0):.0f} pts | DTE: {d.get('dte','')}"
                    )

            if not opt_decisions and not opt_spreads and not opt_straddles:
                st.info("No options decision yet. The adaptive engine runs at ~9:20 AM.")
        except FileNotFoundError:
            st.info("No event log.")

    st.markdown("---")

    # ── AI Activity Feed (legacy) ──
    with st.expander("Audit Log (last 15 entries)"):
        feed = get_ai_activity_feed(15)
        if feed:
            feed_rows = []
            for f in feed:
                type_emoji = {
                    "PLANNER_REQ": "📤", "PLANNER_RES": "📥", "PLANNER_ERR": "💥",
                    "RISK_APPROVE": "✅", "RISK_REJECT": "❌",
                    "EXECUTION": "⚡", "RECONCILE": "🔄",
                }.get(f["type"], "📋")
                feed_rows.append({
                    "Time": f["time"],
                    "Event": f"{type_emoji} {f['type']}",
                    "Symbol": f["symbol"],
                    "Detail": f["detail"],
                })
            st.dataframe(pd.DataFrame(feed_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No AI activity yet today.")



# ══════════════════════════════════════════════
# PAGE: INTRADAY AGENT
# ══════════════════════════════════════════════
elif page == "Intraday Agent":
    st.header("Intraday Agent")
    st.caption("Premarket scan → Monitor proven structures → Auto-trade. Goal: make money.")

    # ── Controls ──
    ctrl_cols = st.columns([2, 1, 1, 1])
    with ctrl_cols[0]:
        ia_date = st.date_input("Trading Date", value=date.today(), max_value=date.today())
    with ctrl_cols[1]:
        ia_universe = st.selectbox("Universe", ["high_volume", "nifty50", "banknifty"], index=0)
    with ctrl_cols[2]:
        ia_skip_llm = st.checkbox("Skip LLM", value=True, help="Fast mode: structure-only decisions")
    with ctrl_cols[3]:
        ia_max_pos = st.number_input("Max Positions", value=3, min_value=1, max_value=5, step=1)

    ia_date_str = ia_date.strftime("%Y-%m-%d")

    # ── Tabs ──
    tab_live, tab_scan, tab_watchlist, tab_signals, tab_trades = st.tabs([
        "Live Activity", "Run Scanner", "Today's Watchlist", "Live Signals", "Today's Trades"
    ])

    # ── TAB 0: Live Activity Feed ──
    with tab_live:
        st.subheader("AI Team Live Activity")
        st.caption("Real-time feed from `run_trading_day`. Start it with: `python manage.py run_trading_day`")

        import json as _json
        from pathlib import Path as _Path

        _event_log = _Path("logs/trading_day_events.jsonl")

        _la_r1, _la_r2 = st.columns([3, 1])
        _la_auto = _la_r2.checkbox("Auto-refresh", value=is_market_open(), key="la_auto")
        if _la_auto:
            st_autorefresh(interval=10000, key="la_autorefresh")

        if _event_log.exists() and _event_log.stat().st_size > 0:
            events = []
            try:
                for line in _event_log.read_text().strip().split("\n"):
                    if line.strip():
                        events.append(_json.loads(line))
            except Exception:
                pass

            if events:
                # Parse event categories
                day_start = next((e for e in events if e["type"] == "DAY_START"), None)
                heartbeats = [e for e in events if e["type"] == "HEARTBEAT"]
                last_hb = heartbeats[-1] if heartbeats else None
                day_review = next((e for e in reversed(events) if e["type"] == "DAY_REVIEW"), None)
                trades_taken = [e for e in events if e["type"] == "TRADE"]
                exits = [e for e in events if e["type"] == "EXIT"]
                straddle_cycles = [e for e in events if e["type"] == "STRADDLE_CYCLE"]
                straddle_regs = [e for e in events if e["type"] == "STRADDLE_REGISTERED"]
                equity_scans = [e for e in events if e["type"] == "EQUITY_SCAN"]

                # ── Session status bar ──
                if day_start:
                    d = day_start["data"]
                    hb_time = last_hb["time"] if last_hb else "?"
                    st.success(f"AI Team active since {day_start['time']} | "
                               f"Mode: {d.get('mode', '?').upper()} | Heartbeat: {hb_time}")

                # ── Top-level P&L metrics ──
                eq_unrealized = last_hb["data"].get("equity_unrealized", 0) if last_hb else 0
                eq_realized = last_hb["data"].get("equity_realized", 0) if last_hb else 0
                str_pnl = straddle_cycles[-1]["data"].get("pnl", 0) if straddle_cycles else 0
                total_est = eq_unrealized + eq_realized + str_pnl

                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Equity (open)", f"{eq_unrealized:+,.0f}", f"{last_hb['data'].get('open_equity', 0)} positions" if last_hb else "")
                p2.metric("Equity (closed)", f"{eq_realized:+,.0f}", f"{len(exits)} exits" if exits else "")
                p3.metric("Straddle", f"{str_pnl:+,.0f}", f"{len(straddle_cycles)} cycles")
                p4.metric("DAY TOTAL", f"{total_est:+,.0f}")

                st.markdown("---")

                # ── Equity ──
                eq_positions = last_hb["data"].get("equity_positions", []) if last_hb else []

                if trades_taken or eq_positions:
                    st.subheader(f"Equity ({len(trades_taken)} trades)")

                    # Live positions table (if heartbeat has prices)
                    if eq_positions:
                        eq_rows = []
                        for p in eq_positions:
                            pnl = p.get("pnl", 0)
                            pct = ((p["ltp"] - p["entry"]) / p["entry"] * 100) if p.get("entry") else 0
                            eq_rows.append({
                                "Symbol": p["symbol"], "Side": p["side"], "Qty": p["qty"],
                                "Entry": f"{p['entry']:.2f}", "LTP": f"{p['ltp']:.2f}",
                                "Change": f"{pct:+.1f}%", "P&L": f"{pnl:+,.0f}", "Status": "OPEN",
                            })
                        # Add closed exits
                        for e in exits:
                            d = e["data"]
                            eq_rows.append({
                                "Symbol": d.get("symbol", "?"), "Side": "-", "Qty": "-",
                                "Entry": "-", "LTP": f"{d.get('price', 0):.2f}",
                                "Change": "-", "P&L": f"{d.get('pnl', 0):+,.0f}",
                                "Status": d.get("reason", "CLOSED"),
                            })
                        st.dataframe(pd.DataFrame(eq_rows), use_container_width=True, hide_index=True)

                    # Always show trade entry history
                    with st.expander(f"Trade Entries ({len(trades_taken)})", expanded=not eq_positions):
                        trade_rows = []
                        for e in trades_taken:
                            d = e["data"]
                            trade_rows.append({
                                "Time": e["time"],
                                "Side": d.get("side", "?"),
                                "Symbol": d.get("symbol", "?"),
                                "Entry": f"{d.get('entry', 0):.2f}",
                                "SL": f"{d.get('sl', 0):.2f}",
                                "Target": f"{d.get('target', 0):.2f}",
                                "R:R": f"{d.get('rr', 0):.1f}",
                                "Setup": d.get("setup", "?"),
                            })
                        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

                # ── Exits ──
                if exits:
                    st.subheader("Closed Trades")
                    for e in exits:
                        d = e["data"]
                        color = "green" if d.get("pnl", 0) >= 0 else "red"
                        st.markdown(
                            f"{e['time']} | {d.get('symbol', '?')} **{d.get('reason', '?')}** "
                            f"@ {d.get('price', 0):.2f} | "
                            f"P&L: :{color}[**{d.get('pnl', 0):+,.0f} INR**]"
                        )

                # ── Straddle ──
                if straddle_cycles or straddle_regs:
                    st.subheader("Straddle")

                    # Registration info
                    for e in straddle_regs:
                        d = e["data"]
                        st.info(
                            f"{e['time']} | Registered #{d.get('position_id')} "
                            f"NIFTY {d.get('strike')} [{d.get('expiry')}] | "
                            f"CE@{d.get('ce_premium', 0):.1f} + PE@{d.get('pe_premium', 0):.1f} = "
                            f"{d.get('combined', 0):.1f} pts"
                        )

                    if straddle_cycles:
                        latest = straddle_cycles[-1]["data"]
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Position",
                                   f"#{latest.get('position_id','?')} {latest.get('underlying','?')} {latest.get('strike','?')}")
                        total_str_pnl = latest.get('pnl', 0)
                        realized = latest.get('realized', 0)
                        unrealized = latest.get('unrealized', total_str_pnl)
                        sc2.metric("Total P&L", f"{total_str_pnl:+,.0f} INR",
                                   f"open: {unrealized:+,.0f} | rolls: {realized:+,.0f}")
                        sc3.metric("Last Action", latest.get("action", "?"))

                        # Show action history (all cycles, not just last 5)
                        with st.expander(f"Straddle History ({len(straddle_cycles)} cycles)", expanded=True):
                            str_rows = []
                            for e in straddle_cycles:
                                d = e["data"]
                                pnl = d.get("pnl", 0)
                                executed = d.get("executed", [])
                                str_rows.append({
                                    "Time": e["time"],
                                    "Action": d.get("action", "?"),
                                    "Strike": d.get("strike", "?"),
                                    "P&L": f"{pnl:+,.0f}",
                                    "Executed": " | ".join(executed) if executed else "-",
                                })
                            st.dataframe(pd.DataFrame(str_rows), use_container_width=True, hide_index=True)

                # ── Watchlist ──
                wl_event = next((e for e in events if e["type"] == "WATCHLIST"), None)
                if wl_event:
                    with st.expander(f"Watchlist ({wl_event['data'].get('count', 0)} stocks)"):
                        wl_rows = []
                        for s in wl_event["data"].get("stocks", []):
                            wl_rows.append({
                                "Symbol": s["symbol"], "Score": s["score"],
                                "Bias": s["bias"],
                                "Setups": ", ".join(s.get("setups", [])) or "-",
                            })
                        if wl_rows:
                            st.dataframe(pd.DataFrame(wl_rows), use_container_width=True, hide_index=True)

                # ── Day Review (end of day) ──
                if day_review:
                    dr = day_review["data"]
                    st.subheader("Day Review")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Equity P&L", f"{dr.get('equity_pnl', 0):+,.0f}")
                    r2.metric("Straddle P&L", f"{dr.get('straddle_pnl', 0):+,.0f}")
                    r3.metric("Total P&L", f"{dr.get('total_pnl', 0):+,.0f}")
                    r4.metric("Return", f"{dr.get('return_pct', 0):+.2f}%")

                    setup_stats = dr.get("setup_stats", {})
                    if setup_stats:
                        st.markdown("**Strategy Performance:**")
                        perf_rows = []
                        for setup, stats in sorted(setup_stats.items(), key=lambda x: -x[1]["pnl"]):
                            wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] else 0
                            perf_rows.append({
                                "Setup": setup, "Trades": stats["trades"],
                                "Wins": stats["wins"], "Win Rate": f"{wr:.0f}%",
                                "P&L": f"{stats['pnl']:+,.0f}",
                            })
                        st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

                # ── Full Timeline ──
                with st.expander("Full Event Log"):
                    action_events = [e for e in events if e["type"] != "HEARTBEAT"]
                    icon_map = {
                        "DAY_START": "🟢", "HEALTH_CHECK": "🏥", "WATCHLIST": "📋",
                        "EQUITY_SCAN": "🔍", "SIGNAL": "⚡", "TRADE": "💰",
                        "SIGNAL_REJECTED": "❌", "EXIT": "🚪",
                        "STRADDLE_REGISTERED": "📝", "STRADDLE_CYCLE": "🔄",
                        "DAY_REVIEW": "📊",
                    }
                    for e in reversed(action_events[-50:]):
                        icon = icon_map.get(e["type"], "📝")
                        # Human-readable summary per event type
                        d = e["data"]
                        if e["type"] == "EQUITY_SCAN":
                            txt = f'scanned {d.get("watchlist_size",0)} | {d.get("signals_found",0)} signals | {d.get("open_positions",0)} open'
                        elif e["type"] == "TRADE":
                            txt = f'{d.get("side")} {d.get("symbol")} @ {d.get("entry",0):.2f} | {d.get("setup")} | RR {d.get("rr",0):.1f}'
                        elif e["type"] == "EXIT":
                            txt = f'{d.get("symbol")} {d.get("reason")} | P&L {d.get("pnl",0):+,.0f}'
                        elif e["type"] == "STRADDLE_CYCLE":
                            txt = f'#{d.get("position_id")} {d.get("action")} P&L {d.get("pnl",0):+,.0f}'
                        elif e["type"] == "STRADDLE_REGISTERED":
                            txt = f'#{d.get("position_id")} NIFTY {d.get("strike")} premium {d.get("combined",0):.0f}'
                        elif e["type"] == "WATCHLIST":
                            txt = f'{d.get("count")} stocks scored'
                        else:
                            txt = _json.dumps(d)[:100]
                        st.caption(f"{icon} {e['time']} | {e['type']} | {txt}")

            else:
                st.info("Event log is empty. Start `run_trading_day` to see live activity.")
        else:
            st.warning(
                "No live session detected. Start the AI team:\n\n"
                "```\npython manage.py run_trading_day\n```\n\n"
                "Or for analysis only: `python manage.py run_trading_day --dry-run`"
            )

    # ── TAB 1: Run Scanner ──
    with tab_scan:
        scan_cols = st.columns(2)
        with scan_cols[0]:
            run_scan = st.button("Run Premarket Scan", type="primary", use_container_width=True)
        with scan_cols[1]:
            run_full = st.button("Run Full Agent (Scan + Monitor + Trade)", use_container_width=True)

        if run_scan:
            with st.spinner(f"Scanning {ia_universe} universe for {ia_date_str}..."):
                try:
                    from trading.intraday.scanner import PremarketScanner
                    scanner = PremarketScanner(lookback_days=5, top_n=10)
                    watchlist = scanner.scan(universe=ia_universe, trading_date=ia_date_str)

                    if watchlist:
                        st.success(f"Found {len(watchlist)} stocks with setups")

                        # Save to DB
                        for s in watchlist:
                            WatchlistEntry.objects.update_or_create(
                                symbol=s.symbol,
                                scan_date=ia_date,
                                defaults={
                                    "score": s.score,
                                    "bias": s.bias.value,
                                    "setups": [st_type.value for st_type in s.setups],
                                    "prev_high": s.prev_high,
                                    "prev_low": s.prev_low,
                                    "prev_close": s.prev_close,
                                    "prev_atr": s.prev_atr,
                                    "reason": s.reason,
                                },
                            )

                        # Display results
                        scan_data = []
                        for i, s in enumerate(watchlist, 1):
                            scan_data.append({
                                "Rank": i,
                                "Symbol": s.symbol,
                                "Score": f"{s.score:.0f}",
                                "Bias": s.bias.value,
                                "Close": f"{s.prev_close:.2f}",
                                "PDH": f"{s.prev_high:.2f}",
                                "PDL": f"{s.prev_low:.2f}",
                                "ATR": f"{s.prev_atr:.2f}",
                                "NR": s.nr_days,
                                "Setups": ", ".join(st_type.value for st_type in s.setups),
                                "Reason": s.reason[:80],
                            })
                        st.dataframe(pd.DataFrame(scan_data), use_container_width=True, hide_index=True)
                    else:
                        st.warning("No stocks found with valid setups.")
                except Exception as e:
                    st.error(f"Scan failed: {e}")

        if run_full:
            with st.spinner(f"Running full intraday agent for {ia_date_str}..."):
                try:
                    from trading.intraday.agent import run_intraday_agent

                    state = run_intraday_agent(
                        trading_date=ia_date_str,
                        universe=ia_universe,
                        skip_llm=ia_skip_llm,
                        max_positions=ia_max_pos,
                    )

                    # Save watchlist
                    for s in state.watchlist:
                        WatchlistEntry.objects.update_or_create(
                            symbol=s.symbol,
                            scan_date=ia_date,
                            defaults={
                                "score": s.score,
                                "bias": s.bias.value,
                                "setups": [st_type.value for st_type in s.setups],
                                "prev_high": s.prev_high,
                                "prev_low": s.prev_low,
                                "prev_close": s.prev_close,
                                "prev_atr": s.prev_atr,
                                "reason": s.reason,
                            },
                        )

                    st.success(f"Agent complete! Watchlist: {len(state.watchlist)} | "
                               f"Trades: {len(state.trades_today)} | Phase: {state.phase.value}")

                    if state.premarket_analysis:
                        with st.expander("Premarket Analysis (LLM)", expanded=True):
                            st.text(state.premarket_analysis)

                    if state.trades_today:
                        st.subheader("Trades Taken")
                        trade_data = []
                        for t in state.trades_today:
                            trade_data.append({
                                "Time": t["time"],
                                "Symbol": t["symbol"],
                                "Side": t["side"],
                                "Qty": t["qty"],
                                "Entry": f"{t['entry']:.2f}",
                                "SL": f"{t['sl']:.2f}",
                                "Target": f"{t['target']:.2f}",
                                "Setup": t["setup"],
                                "Status": t["status"],
                            })
                        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
                    else:
                        st.info("No trades triggered in this session.")

                except Exception as e:
                    st.error(f"Agent failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # ── TAB 2: Today's Watchlist ──
    with tab_watchlist:
        entries = WatchlistEntry.objects.filter(scan_date=ia_date).order_by("-score")

        if entries.exists():
            st.metric("Stocks on Watchlist", entries.count())

            wl_data = []
            for e in entries:
                wl_data.append({
                    "Symbol": e.symbol,
                    "Score": f"{e.score:.0f}",
                    "Bias": e.bias,
                    "Close": f"{e.prev_close:.2f}",
                    "PDH": f"{e.prev_high:.2f}",
                    "PDL": f"{e.prev_low:.2f}",
                    "ATR": f"{e.prev_atr:.2f}",
                    "Setups": ", ".join(e.setups) if e.setups else "-",
                    "Outcome": e.outcome,
                    "Reason": e.reason[:80] if e.reason else "-",
                })
            st.dataframe(pd.DataFrame(wl_data), use_container_width=True, hide_index=True)

            # Score distribution chart
            scores = [e.score for e in entries]
            symbols = [e.symbol for e in entries]
            chart_df = pd.DataFrame({"Symbol": symbols, "Score": scores})
            st.bar_chart(chart_df.set_index("Symbol"))
        else:
            st.info(f"No watchlist for {ia_date_str}. Run the premarket scan first.")

    # ── TAB 3: Live Signals ──
    with tab_signals:
        st.info("Live signals appear here during market hours when price structures trigger on watchlist stocks.")

        # Show recent audit logs related to intraday trading
        recent_signals = AuditLog.objects.filter(
            event_type="EXECUTION",
            created_at__date=ia_date,
        ).order_by("-created_at")[:20]

        if recent_signals.exists():
            sig_data = []
            for s in recent_signals:
                sig_data.append({
                    "Time": s.created_at.strftime("%H:%M:%S"),
                    "Symbol": s.symbol,
                    "Details": s.prompt[:80] if s.prompt else "-",
                })
            st.dataframe(pd.DataFrame(sig_data), use_container_width=True, hide_index=True)
        else:
            st.caption("No signals recorded yet for today.")

        # Structure detector reference
        with st.expander("Price Structures Reference"):
            st.markdown("""
**ORB (Opening Range Breakout)** — First 15 min high/low. Trade breakout with volume. SL at opposite end.

**PDH/PDL Break** — Yesterday's high or low breaks with momentum. SL near the broken level.

**Gap and Go** — Stock gaps, holds after ORB, ride in gap direction. SL at ORB opposite end.

**Gap Fill** — Gap starts reverting toward previous close. Trade the fill.

**VWAP Reclaim** — Price crosses above VWAP from below with volume → Long.

**VWAP Reject** — Price tests VWAP from below, gets rejected → Short.
            """)

    # ── TAB 4: Today's Trades ──
    with tab_trades:
        today_trades = TradeJournal.objects.filter(trade_date=ia_date).order_by("created_at")

        if today_trades.exists():
            # Summary metrics
            total = today_trades.count()
            executed = today_trades.filter(status__in=["EXECUTED", "PAPER"]).count()
            rejected = today_trades.filter(status="REJECTED").count()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Trades", total)
            m2.metric("Executed", executed)
            m3.metric("Rejected", rejected)
            m4.metric("Hit Rate", f"{(executed/total*100):.0f}%" if total > 0 else "0%")

            # Trade table
            trade_rows = []
            for t in today_trades:
                trade_rows.append({
                    "Time": t.created_at.strftime("%H:%M:%S"),
                    "Symbol": t.symbol,
                    "Side": t.side,
                    "Qty": t.quantity,
                    "Entry": f"{t.entry_price:.2f}",
                    "SL": f"{t.stop_loss:.2f}",
                    "Target": f"{t.target:.2f}",
                    "R:R": f"{t.risk_reward_ratio:.1f}",
                    "Conf": f"{t.confidence:.2f}",
                    "Status": t.status,
                    "Reason": (t.reasoning or "")[:60],
                })
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
        else:
            st.info(f"No trades for {ia_date_str}.")


# ══════════════════════════════════════════════
# PAGE 2: MARKET SCANNER
# ══════════════════════════════════════════════
elif page == "Market Scanner":
    st.header("Market Scanner — NIFTY 50")
    st.caption("Scan → Analyze with indicators → Approve trade with AI team")

    from dashboard_utils.components import render_indicator_chart
    from dashboard_utils.market_scanner import SECTOR_MAP, fetch_nifty50_ltp, rank_opportunities
    from dashboard_utils.candle_cache import fetch_and_cache_candles, get_candles_for_timeframe

    # ── Scan controls ──
    sc1, sc2, sc3 = st.columns([2, 1, 1])
    with sc1:
        scan_scope = st.multiselect("Sectors", list(SECTOR_MAP.keys()), default=list(SECTOR_MAP.keys()))
    with sc2:
        top_n = st.selectbox("Show top", [10, 20, 30, 50], index=1)
    with sc3:
        scan_btn = st.button("Scan Now", type="primary", use_container_width=True)

    scan_symbols = list(set(s for sec in scan_scope for s in SECTOR_MAP.get(sec, [])))

    if scan_btn:
        with st.spinner(f"Scanning {len(scan_symbols)} stocks (1 batch call)..."):
            try:
                from trading.services.data_service import BrokerClient
                b = BrokerClient.get_instance()
                b.ensure_login()
                results = fetch_nifty50_ltp(b, scan_symbols)
                results = rank_opportunities(results)
                st.session_state["scanner_results"] = results
                st.session_state["scanner_ts"] = datetime.now()
            except Exception as e:
                st.error(f"Scan failed: {e}")

    results = st.session_state.get("scanner_results", [])
    scan_ts = st.session_state.get("scanner_ts")
    if scan_ts:
        st.caption(f"Last scan: {scan_ts.strftime('%H:%M:%S')}")

    if results:
        display = results[:top_n]

        # ── Sidebar stock list + main analysis area ──
        list_col, detail_col = st.columns([1, 3])

        with list_col:
            st.markdown("**Stocks**")
            for i, s in enumerate(display):
                chg = s.get("change_pct", 0)
                color = "#26a69a" if chg >= 0 else "#ef5350"
                label = f"{s['symbol']}"
                if st.button(
                    f"{s['symbol']:10s} ₹{s['ltp']:,.0f} {chg:+.1f}%",
                    key=f"stock_{i}",
                    use_container_width=True,
                ):
                    st.session_state["scanner_selected"] = s["symbol"]

        selected = st.session_state.get("scanner_selected", display[0]["symbol"] if display else None)

        with detail_col:
          if selected:
            stock = next((s for s in display if s["symbol"] == selected), None)
            if stock:
                st.subheader(f"{selected}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("LTP", f"₹{stock['ltp']:,.2f}")
                m2.metric("Change", f"{stock.get('change_pct',0):+.2f}%")
                m3.metric("Sector", stock.get("sector", "—"))
                m4.metric("Score", stock.get("opportunity_score", 0))

            # ── Fetch candles (cached, auto-detects after-hours) ──
            analysis_data = st.session_state.get(f"analysis_{selected}")

            ab1, ab2 = st.columns([1, 1])
            with ab1:
                analyze_btn = st.button(f"Analyze {selected}", type="primary", use_container_width=True)
            with ab2:
                refresh_btn = st.button("Refresh Data", use_container_width=True)

            if analyze_btn or refresh_btn:
                with st.spinner(f"Fetching candles for {selected} (2 API calls, cached)..."):
                    try:
                        from trading.utils.indicators import (
                            compute_indicator_confluence, atr as calc_atr,
                        )
                        from trading.services.ticker_service import ticker_service

                        cache = fetch_and_cache_candles(selected, force=refresh_btn)
                        if cache.get("error"):
                            st.error(cache["error"])
                        else:
                            candles = cache.get("intraday_5m", [])
                            pivots = cache.get("pivots", {})
                            prev_day = cache.get("prev_day")
                            token = ticker_service.get_token(selected)

                            # Compute indicators from intraday candles
                            closes = [c["close"] for c in candles] if candles else []
                            indicators = {}
                            if closes:
                                indicators = compute_indicator_confluence(
                                    closes, candles,
                                    prev_high=prev_day["high"] if prev_day else 0,
                                    prev_low=prev_day["low"] if prev_day else 0,
                                    prev_close=prev_day["close"] if prev_day else 0,
                                )

                            # Structure detection
                            signals = []
                            if candles and len(candles) >= 4 and prev_day:
                                from trading.intraday.structures import detect_all_structures
                                from trading.intraday.state import StockSetup, TradeBias
                                from trading.services.data_service import DataService
                                ds = DataService()
                                daily = cache.get("daily", [])
                                setup = StockSetup(
                                    symbol=selected, token=token or "", bias=TradeBias.NEUTRAL, score=50,
                                    prev_high=prev_day["high"], prev_low=prev_day["low"],
                                    prev_close=prev_day["close"],
                                    prev_atr=calc_atr(daily[-5:]) if len(daily) >= 5 else 0,
                                    pivot_s3=pivots.get("S3", 0), pivot_s4=pivots.get("S4", 0),
                                    pivot_r3=pivots.get("R3", 0), pivot_r4=pivots.get("R4", 0),
                                    pivot_p=pivots.get("P", 0),
                                    today_open=candles[0]["open"] if candles else 0,
                                    orb_high=max(c["high"] for c in candles[:3]) if len(candles) >= 3 else 0,
                                    orb_low=min(c["low"] for c in candles[:3]) if len(candles) >= 3 else 0,
                                )
                                avg_vol = sum(c.get("volume", 0) for c in candles) / len(candles)
                                signals = detect_all_structures(setup, candles, avg_vol)

                            analysis_data = {
                                "candles": candles, "pivots": pivots, "indicators": indicators,
                                "signals": signals, "prev_day": prev_day,
                                "intraday_date": cache.get("intraday_date", ""),
                            }
                            st.session_state[f"analysis_{selected}"] = analysis_data

                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            # Display analysis
            if analysis_data:
                candles = analysis_data.get("candles", [])
                pivots = analysis_data.get("pivots", {})
                indicators = analysis_data.get("indicators", {})
                signals = analysis_data.get("signals", [])
                intraday_date = analysis_data.get("intraday_date", "")

                # Indicator metrics
                ic1, ic2, ic3, ic4, ic5, ic6 = st.columns(6)
                ic1.metric("RSI", indicators.get("rsi", "—"))
                ic2.metric("MACD", indicators.get("macd", {}).get("histogram", "—") if isinstance(indicators.get("macd"), dict) else "—")
                ic3.metric("VWAP", f"{indicators.get('vwap', 0):,.0f}" if indicators.get("vwap") else "—")
                ic4.metric("BB Squeeze", "YES" if indicators.get("bb", {}).get("squeeze") else "No")
                ic5.metric("Confluence", f"{indicators.get('confluence_score', 0)}")
                ic6.metric("Bias", indicators.get("confluence_bias", "—"))

                if intraday_date:
                    from trading.utils.time_utils import is_market_open
                    status = "LIVE" if is_market_open() else "After-hours"
                    st.caption(f"Intraday data: {intraday_date} ({status})")

                # ── Timeframe tabs: 5m | Daily | Weekly | Monthly ──
                tf_tab1, tf_tab2, tf_tab3, tf_tab4 = st.tabs(["5m Intraday", "Daily", "Weekly", "Monthly"])

                best_signal = signals[0] if signals else None

                with tf_tab1:
                    render_indicator_chart(
                        candles, selected, pivots,
                        bb=indicators.get("bb"),
                        vwap_val=indicators.get("vwap", 0),
                        entry=best_signal.entry_price if best_signal else 0,
                        sl=best_signal.stop_loss if best_signal else 0,
                        target=best_signal.target if best_signal else 0,
                        timeframe="5m",
                    )

                with tf_tab2:
                    daily_candles = get_candles_for_timeframe(selected, "daily")
                    render_indicator_chart(
                        daily_candles, selected, timeframe="daily",
                    )

                with tf_tab3:
                    weekly_candles = get_candles_for_timeframe(selected, "weekly")
                    render_indicator_chart(
                        weekly_candles, selected, timeframe="weekly",
                    )

                with tf_tab4:
                    monthly_candles = get_candles_for_timeframe(selected, "monthly")
                    render_indicator_chart(
                        monthly_candles, selected, timeframe="monthly",
                    )

                # Camarilla levels
                if pivots:
                    with st.expander("Camarilla Pivots"):
                        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                        pc1.metric("S4", f"{pivots.get('S4',0):,.1f}")
                        pc2.metric("S3 (buy)", f"{pivots.get('S3',0):,.1f}")
                        pc3.metric("Pivot", f"{pivots.get('P',0):,.1f}")
                        pc4.metric("R3 (sell)", f"{pivots.get('R3',0):,.1f}")
                        pc5.metric("R4", f"{pivots.get('R4',0):,.1f}")

                # ── AI Signal + Human Approval ──
                st.markdown("---")
                if signals:
                    sig = signals[0]
                    st.subheader("AI Signal Detected")

                    sg1, sg2, sg3, sg4 = st.columns(4)
                    sg1.metric("Setup", sig.setup_type.value)
                    sg2.metric("Bias", sig.bias.value)
                    sg3.metric("Confidence", f"{sig.confidence:.0%}")
                    sg4.metric("R:R", f"{sig.risk_reward:.1f}")

                    sg5, sg6, sg7 = st.columns(3)
                    sg5.metric("Entry", f"₹{sig.entry_price:,.2f}")
                    sg6.metric("Stop Loss", f"₹{sig.stop_loss:,.2f}")
                    sg7.metric("Target", f"₹{sig.target:,.2f}")

                    st.caption(sig.reason)

                    # Human-in-the-loop: approve or reject
                    st.markdown("#### Execute Trade?")
                    approve_cols = st.columns([1, 1, 2])

                    with approve_cols[0]:
                        if st.button("APPROVE & EXECUTE", type="primary", use_container_width=True,
                                     key=f"approve_{selected}"):
                            with st.spinner("Executing..."):
                                from trading.intraday.monitor import IntradayMonitor
                                from trading.intraday.state import IntradayState, Phase
                                state = IntradayState(
                                    trading_date=date.today().strftime("%Y-%m-%d"),
                                    capital=500000, max_positions=3, phase=Phase.ACTIVE,
                                )
                                monitor = IntradayMonitor(state)
                                result = monitor.process_signal(sig)
                                if result.get("action") == "TRADED":
                                    st.success(f"TRADED: {sig.side} {selected} @ {sig.entry_price:.2f}")
                                elif result.get("action") == "REJECTED":
                                    st.error(f"REJECTED by @RiskGuard: {result.get('reason', '')}")
                                else:
                                    st.warning(f"Result: {result}")

                    with approve_cols[1]:
                        if st.button("SKIP", use_container_width=True, key=f"skip_{selected}"):
                            st.info("Signal skipped.")
                else:
                    st.info(f"No price structure detected for {selected} right now. "
                            f"The AI team checks every 5 minutes — structures form throughout the day.")
    else:
        st.info("Click **Scan Now** to discover opportunities.")


# ══════════════════════════════════════════════
# PAGE: SCREENER
# ══════════════════════════════════════════════
elif page == "Screener":
    st.header("Live Screener")
    st.caption("Real-time opportunity detection across NIFTY 50 | 6 strategies | Multi-TF")

    from trading.screener.strategies import STRATEGIES
    from trading.screener.signals import Signal

    tab_live, tab_backtest, tab_strategies_tab = st.tabs([
        "Live Signals", "Backtest", "Strategies"
    ])

    # ── Live Signals Tab ──
    with tab_live:
        lc1, lc2, lc3 = st.columns([2, 1, 1])

        with lc1:
            screener_symbols = st.text_input(
                "Symbols (comma separated)",
                value="RELIANCE,HDFCBANK,TCS,INFY,SBIN,AXISBANK,ICICIBANK,KOTAKBANK,BAJFINANCE,ITC",
                key="screener_syms",
            ).strip().upper().split(",")
            screener_symbols = [s.strip() for s in screener_symbols if s.strip()]

        with lc2:
            screener_strategies = st.multiselect(
                "Strategies",
                [s.name for s in STRATEGIES if s.enabled],
                default=[s.name for s in STRATEGIES if s.enabled],
                key="screener_strats",
            )

        with lc3:
            scan_btn = st.button("Scan Now", type="primary", use_container_width=True, key="screener_scan")

        if scan_btn and screener_symbols:
            with st.spinner(f"Running screener on {len(screener_symbols)} symbols..."):
                try:
                    from trading.screener.engine import ScreenerEngine
                    from trading.services.data_service import BrokerClient
                    from trading.services.ticker_service import ticker_service

                    # Build engine with selected strategies
                    selected = [s for s in STRATEGIES if s.name in screener_strategies and s.enabled]
                    engine = ScreenerEngine(symbols=screener_symbols, strategies=selected)

                    # Collect signals
                    signals_found = []
                    engine.add_output_handler(lambda sig: signals_found.append(sig))

                    # Bootstrap with historical candles
                    b = BrokerClient.get_instance()
                    b.ensure_login()

                    def fetch_fn(symbol, tf, n_bars):
                        token = ticker_service.get_token(symbol)
                        if not token:
                            return []
                        from trading.utils.time_utils import can_fetch_candles, cap_end_time
                        if not can_fetch_candles():
                            return []
                        interval = {"1m": "ONE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE"}.get(tf, "FIVE_MINUTE")
                        end = cap_end_time(date.today().isoformat())
                        raw = b.fetch_candles(token, f"{date.today().isoformat()} 09:15", end, interval)
                        return raw or []

                    engine.bootstrap(fetch_candles_fn=fetch_fn)

                    # Run batch tick with current prices
                    batch = []
                    from trading.services.data_service import DataService
                    ds = DataService()
                    ltp_data = ds.fetch_batch_ltp(screener_symbols)
                    for item in ltp_data:
                        batch.append({
                            "symbol": item["symbol"],
                            "ltp": item["ltp"],
                            "volume": item.get("volume", 0),
                        })
                    if batch:
                        engine.on_batch_tick(batch)

                    st.session_state["screener_signals"] = signals_found
                    st.session_state["screener_ts"] = datetime.now()
                    st.session_state["screener_engine_stats"] = {
                        "symbols": len(screener_symbols),
                        "strategies": len(selected),
                        "bars_total": sum(len(s.bars.get("5m", [])) for s in engine.stores.values()),
                    }

                except Exception as e:
                    st.error(f"Screener failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Display signals
        signals_found = st.session_state.get("screener_signals", [])
        scan_ts = st.session_state.get("screener_ts")
        stats = st.session_state.get("screener_engine_stats", {})

        if scan_ts:
            st.caption(f"Last scan: {scan_ts.strftime('%H:%M:%S')} | "
                       f"{stats.get('symbols', 0)} symbols | "
                       f"{stats.get('strategies', 0)} strategies | "
                       f"{stats.get('bars_total', 0)} bars loaded")

        if signals_found:
            st.success(f"{len(signals_found)} signal(s) detected")

            for sig in signals_found:
                with st.container():
                    arrow = "🟢" if sig.side == "BUY" else "🔴"
                    st.markdown(f"### {arrow} {sig.symbol} — {sig.strategy}")

                    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                    sc1.metric("Side", sig.side)
                    sc2.metric("Entry", f"₹{sig.entry:,.2f}")
                    sc3.metric("Stop Loss", f"₹{sig.stoploss:,.2f}")
                    sc4.metric("Target", f"₹{sig.target:,.2f}")
                    sc5.metric("R:R", f"{sig.risk_reward:.1f}")

                    sc6, sc7 = st.columns(2)
                    sc6.metric("Confidence", f"{sig.confidence:.0%}")
                    sc7.metric("Risk", f"{sig.risk_points:.1f} pts ({sig.risk_pct}%)")

                    st.caption(f"Reasons: {' | '.join(sig.reasons)}")

                    # Human-in-the-loop: approve trade
                    acol1, acol2 = st.columns([1, 1])
                    with acol1:
                        if st.button(f"APPROVE {sig.symbol} {sig.side}", type="primary",
                                     key=f"approve_scr_{sig.symbol}_{sig.strategy}"):
                            st.info(f"Approved — route to execution pipeline")
                    with acol2:
                        if st.button(f"Skip", key=f"skip_scr_{sig.symbol}_{sig.strategy}"):
                            st.info("Skipped")

                    st.markdown("---")
        elif scan_ts:
            st.info("No signals detected in this scan. Market may be quiet or conditions not met.")

    # ── Backtest Tab ──
    with tab_backtest:
        st.subheader("Screener Backtest")

        bc1, bc2, bc3 = st.columns(3)
        bt_from = bc1.date_input("From", value=date.today() - timedelta(days=5), key="scr_bt_from")
        bt_to = bc2.date_input("To", value=date.today() - timedelta(days=1), key="scr_bt_to")
        bt_syms = bc3.text_input("Symbols", value="RELIANCE,HDFCBANK,TCS,INFY,SBIN", key="scr_bt_syms")

        if st.button("Run Backtest", type="primary", key="scr_bt_run"):
            with st.spinner("Running screener backtest..."):
                try:
                    from trading.screener.backtest import run_backtest as screener_backtest
                    symbols = [s.strip() for s in bt_syms.split(",") if s.strip()]
                    result = screener_backtest(
                        symbols=symbols,
                        from_date=bt_from.isoformat(),
                        to_date=bt_to.isoformat(),
                    )

                    # Summary
                    sm1, sm2, sm3, sm4 = st.columns(4)
                    sm1.metric("Signals", result.total_signals)
                    sm2.metric("Trades", len(result.trades))
                    sm3.metric("Win Rate", f"{result.win_rate:.0f}%")
                    sm4.metric("Profit Factor", f"{result.profit_factor:.2f}")

                    sm5, sm6, sm7, sm8 = st.columns(4)
                    sm5.metric("Total P&L", f"{result.total_pnl:+,.0f} pts")
                    sm6.metric("Avg R:R", f"{result.avg_rr_achieved:.1f}")
                    sm7.metric("Max DD", f"{result.max_drawdown:+,.0f} pts")
                    sm8.metric("Best Day", f"{result.best_day_pnl:+,.0f}" if hasattr(result, 'best_day_pnl') else "—")

                    # Per strategy
                    if result.per_strategy:
                        st.subheader("By Strategy")
                        strat_rows = []
                        for name, stats in result.per_strategy.items():
                            strat_rows.append({
                                "Strategy": name,
                                "Trades": stats.get("trades", 0),
                                "Wins": stats.get("wins", 0),
                                "WR%": f"{stats.get('win_rate', 0):.0f}%",
                                "P&L": f"{stats.get('pnl', 0):+,.0f}",
                                "PF": f"{stats.get('profit_factor', 0):.2f}",
                            })
                        st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

                    # Trade log
                    if result.trades:
                        st.subheader("Trade Log")
                        trade_rows = []
                        for t in result.trades:
                            trade_rows.append({
                                "Time": t.signal.timestamp.strftime("%m-%d %H:%M") if t.signal else "—",
                                "Symbol": t.signal.symbol if t.signal else "—",
                                "Strategy": t.signal.strategy if t.signal else "—",
                                "Side": t.signal.side if t.signal else "—",
                                "Entry": f"{t.entry_price:.2f}" if hasattr(t, 'entry_price') else "—",
                                "Exit": f"{t.exit_price:.2f}",
                                "P&L": f"{t.pnl:+,.1f}",
                                "Reason": t.exit_reason,
                            })
                        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Backtest failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # ── Strategies Tab ──
    with tab_strategies_tab:
        st.subheader("Strategy Definitions")
        for s in STRATEGIES:
            status = "🟢 Enabled" if s.enabled else "🔴 Disabled"
            with st.expander(f"{status} — {s.name} ({s.side})"):
                st.markdown(f"**{s.description}**")
                st.markdown(f"**Side:** {s.side} | **Min R:R:** {s.min_rr} | **Cooldown:** {s.cooldown_bars} bars")
                if s.active_window:
                    st.markdown(f"**Active window:** {s.active_window[0].strftime('%H:%M')} — {s.active_window[1].strftime('%H:%M')}")
                st.markdown("**Conditions:**")
                for c in s.conditions:
                    st.markdown(f"- {c.description}")
                st.markdown(f"**Entry:** {s.entry_rule.method} | **SL:** {s.stoploss_rule.method} ({s.stoploss_rule.params}) | "
                            f"**Target:** {s.target_rule.method} ({s.target_rule.params})")


# ══════════════════════════════════════════════
# PAGE 3: TRADE WORKFLOW
# ══════════════════════════════════════════════
elif page == "Trade Workflow":
    st.header("Execute Trade Workflow")
    st.caption("Orchestrator → Data → RAG → Planner → Risk → Execute → Journal")

    # Pre-fill from scanner if applicable
    prefill = st.session_state.pop("scanner_to_trade", "")

    col_input, col_config = st.columns([2, 1])

    with col_input:
        default_intent = f"Plan a trade for {prefill}" if prefill else ""
        intent = st.text_area(
            "Trading Intent",
            value=default_intent,
            placeholder="e.g. Plan a BUY trade for MFSL based on today's intraday structure",
            height=100,
        )

    with col_config:
        symbol = st.text_input("Symbol (optional)", value=prefill, placeholder="MFSL").strip().upper()
        run_full = st.checkbox("Run full workflow (includes execution)", value=True)

    # TradingView chart for the symbol
    if symbol:
        with st.expander(f"📊 {symbol} Chart", expanded=False):
            render_tradingview_chart(symbol, height=400)

    if st.button("Run Workflow", type="primary", use_container_width=True):
        if not intent.strip():
            st.warning("Enter a trading intent first.")
        else:
            progress = st.progress(0)
            status_area = st.empty()

            status_area.info("📊 Fetching market data...")
            progress.progress(10)

            from trading.graph.trading_graph import (
                fetch_data_node, retrieve_context_node,
                planner_node, risk_node, execute_node, journal_node,
            )

            state: dict = {
                "user_intent": intent,
                "symbol": symbol,
                "market_data": None, "market_data_raw": None,
                "rag_context": None, "trade_plan": None, "planner_raw": None,
                "risk_approved": None, "risk_result": None,
                "execution_result": None, "journal_id": None, "error": None,
            }

            # Fetch data
            result = fetch_data_node(state)
            state.update(result)
            progress.progress(20)

            if state.get("error") and not state.get("market_data_raw"):
                st.error(f"Data fetch failed: {state['error']}")
                st.stop()

            # RAG
            status_area.info("🧠 Retrieving context (trades + strategies)...")
            result = retrieve_context_node(state)
            state.update(result)
            progress.progress(35)

            with st.expander("RAG Context", expanded=False):
                st.text(state.get("rag_context", "No context"))

            # Planner
            status_area.info("🤖 Running planner (Claude)...")
            result = planner_node(state)
            state.update(result)
            progress.progress(55)

            plan = state.get("trade_plan")
            if plan:
                st.markdown("### Trade Plan")
                plan_cols = st.columns(4)
                plan_cols[0].metric("Symbol", plan["symbol"])
                plan_cols[1].metric("Side", plan["side"])
                plan_cols[2].metric("Entry", f"₹{plan['entry_price']:.2f}")
                plan_cols[3].metric("Confidence", f"{plan['confidence']:.0%}")

                detail_cols = st.columns(3)
                detail_cols[0].metric("Stop Loss", f"₹{plan['stop_loss']:.2f}")
                detail_cols[1].metric("Target", f"₹{plan['target']:.2f}")
                detail_cols[2].metric("Quantity", plan["quantity"])

                st.markdown(f"**Reasoning:** {plan['reasoning']}")
            elif state.get("error"):
                st.error(f"Planner error: {state['error']}")
            else:
                st.warning("Planner did not produce a trade plan.")

            # Risk
            status_area.info("🛡 Running risk validation...")
            result = risk_node(state)
            state.update(result)
            progress.progress(70)

            risk = state.get("risk_result", {})
            approved = state.get("risk_approved", False)

            if approved:
                st.success(f"🛡 Risk: **APPROVED** — {risk.get('reason', '')}")
                if risk.get("risk_amount"):
                    r_cols = st.columns(2)
                    r_cols[0].metric("Risk Amount", f"₹{risk['risk_amount']:.0f}")
                    r_cols[1].metric("Risk % Capital", f"{risk.get('risk_pct_of_capital', 0):.1f}%")
            else:
                st.error(f"🛡 Risk: **REJECTED** — {risk.get('reason', 'No plan')}")

            # Execute
            if approved and run_full and plan:
                status_area.info(f"⚡ Executing ({TRADING_MODE} mode)...")
                result = execute_node(state)
                state.update(result)
                progress.progress(85)

                exec_r = state.get("execution_result", {})
                if exec_r.get("success"):
                    st.success(
                        f"✅ **{exec_r.get('mode', 'paper').upper()} FILL:** "
                        f"{exec_r.get('fill_quantity', 0)}x @ ₹{exec_r.get('fill_price', 0):.2f} "
                        f"(Order: {exec_r.get('order_id', 'N/A')})"
                    )
                else:
                    st.error(f"Execution failed: {exec_r.get('message', 'Unknown')}")
            elif not approved:
                status_area.warning("Skipped execution — risk rejected.")

            # Journal
            status_area.info("📓 Saving to journal...")
            result = journal_node(state)
            state.update(result)
            progress.progress(100)

            jid = state.get("journal_id")
            if jid:
                st.success(f"📓 Journal entry **#{jid}** saved.")

            status_area.success("Workflow complete.")

            with st.expander("Full State (debug)", expanded=False):
                debug_state = {}
                for k, v in state.items():
                    try:
                        json.dumps(v)
                        debug_state[k] = v
                    except (TypeError, ValueError):
                        debug_state[k] = str(v)
                st.json(debug_state)


# ══════════════════════════════════════════════
# PAGE 4: STRADDLE CONSOLE
# ══════════════════════════════════════════════
elif page == "Straddle Console":
    _svc = _options_svc

    def _is_market_hours() -> bool:
        return is_market_open()

    st.header("Short Straddle Console")
    mode_badge = "🔴 LIVE" if TRADING_MODE == "live" else "🟢 PAPER"
    st.caption(
        f"Register → Monitor → Analyze → Execute  |  "
        f"@OptionsStrategist + @RiskGuard  |  {mode_badge}  |  "
        f"{'🟢 Market Open' if _is_market_hours() else '🔴 Market Closed'}"
    )

    _active_positions = list(StraddlePosition.objects.filter(
        status__in=[StraddlePosition.Status.ACTIVE, StraddlePosition.Status.HEDGED,
                    StraddlePosition.Status.PARTIAL]
    ).order_by("-opened_at"))

    tab_monitor, tab_analyze, tab_payoff, tab_register, tab_history = st.tabs([
        "📊 Monitor", "🤖 Analyze & Act", "📈 Payoff & Greeks", "➕ Register", "📜 History"
    ])

    # ────────────────────────────────
    # TAB: Monitor
    # ────────────────────────────────
    with tab_monitor:
        if not _active_positions:
            st.info("No active straddle positions. Register one in the ➕ tab.")
        else:
            pos_options = {
                f"#{p.id} | {p.underlying} {p.display_strike} [{p.expiry}] {p.status}": p
                for p in _active_positions
            }
            selected_label = st.selectbox(
                "Select Position", list(pos_options.keys()), key="monitor_pos_select"
            )
            pos = pos_options[selected_label]

            # Refresh controls
            cache_key = f"monitor_snap_{pos.id}"
            ts_key = f"monitor_snap_ts_{pos.id}"
            snap_ts: Optional[datetime] = st.session_state.get(ts_key)
            stale_secs = int((datetime.now() - snap_ts).total_seconds()) if snap_ts else 999
            stale_warn = stale_secs > 300

            ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
            refresh = ctrl1.button("🔄 Refresh Now", use_container_width=True, key="monitor_refresh")
            auto_refresh = ctrl2.checkbox("Auto-refresh", value=_is_market_hours(), key="monitor_auto_refresh")
            refresh_interval = ctrl3.selectbox(
                "Every", [15, 30, 60, 120], index=1, key="monitor_interval",
                format_func=lambda x: f"{x}s",
            )

            if auto_refresh:
                st_autorefresh(interval=refresh_interval * 1000, key="monitor_autorefresh")

            if snap_ts:
                ts_str = snap_ts.strftime("%H:%M:%S")
                if stale_warn:
                    st.warning(f"Data is {stale_secs//60}m {stale_secs%60}s old (last: {ts_str}). Refresh.")
                else:
                    st.caption(f"Last updated: {ts_str}  ({stale_secs}s ago)")

            need_fetch = (
                refresh
                or cache_key not in st.session_state
                or (auto_refresh and stale_secs >= refresh_interval)
            )

            if need_fetch:
                with st.spinner("Fetching live prices..."):
                    try:
                        snap_data = _svc.fetch_straddle_snapshot(
                            ce_symbol=pos.ce_symbol, ce_token=pos.ce_token,
                            pe_symbol=pos.pe_symbol, pe_token=pos.pe_token,
                            date_str=date.today().isoformat(),
                            include_candles=False,
                        )
                        st.session_state[cache_key] = snap_data
                        st.session_state[ts_key] = datetime.now()
                        snap_ts = st.session_state[ts_key]
                        stale_secs = 0
                    except Exception as e:
                        st.error(f"Live data fetch failed: {e}")

            snap_data = st.session_state.get(cache_key, {})
            if not snap_data:
                st.warning("No market data. Click Refresh.")
            else:
                try:
                    nifty = snap_data.get("nifty", {})
                    vix = snap_data.get("vix", {})
                    ce = snap_data.get("ce", {})
                    pe = snap_data.get("pe", {})

                    analysis = analyze_straddle(
                        underlying=pos.underlying, strike=pos.ce_strike_actual,
                        expiry=pos.expiry.isoformat(), lot_size=pos.lot_size, lots=pos.lots,
                        ce_sell_price=pos.ce_sell_price, pe_sell_price=pos.pe_sell_price,
                        ce_ltp=ce.get("ltp", 0), pe_ltp=pe.get("ltp", 0),
                        nifty_spot=nifty.get("ltp", 0), nifty_prev_close=nifty.get("prev_close", 0),
                        vix_current=vix.get("ltp", 0), vix_prev_close=vix.get("prev_close", 0),
                        candles=[],
                    )

                    # Track premium for decay chart
                    decay_key = f"premium_decay_{pos.id}"
                    if decay_key not in st.session_state:
                        st.session_state[decay_key] = []
                    st.session_state[decay_key].append({
                        "time": datetime.now().strftime("%H:%M"),
                        "premium": analysis.combined_current,
                    })
                    # Keep last 100 data points
                    st.session_state[decay_key] = st.session_state[decay_key][-100:]

                    # Critical alerts
                    if analysis.is_underwater:
                        st.error("🚨 POSITION UNDERWATER — Combined premium > sold. CLOSE immediately.")
                    elif analysis.stop_triggered:
                        st.error("🚨 STOP TRIGGERED — Premium at 1.5x sold. Review now.")
                    if analysis.expiry_tomorrow:
                        st.warning("⚠️ EXPIRY TOMORROW — Gamma is extreme. Prefer closing over holding.")

                    # Market
                    st.subheader("Market")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    nifty_chg = nifty.get("ltp", 0) - nifty.get("prev_close", 0)
                    m1.metric("NIFTY", f"{nifty.get('ltp', 0):,.0f}", f"{nifty_chg:+.0f}")
                    m2.metric("High / Low", f"{nifty.get('high', 0):,.0f} / {nifty.get('low', 0):,.0f}")
                    vix_chg = vix.get("ltp", 0) - vix.get("prev_close", 0)
                    m3.metric("VIX", f"{vix.get('ltp', 0):.2f}", f"{vix_chg:+.2f}")
                    m4.metric("VIX Phase", analysis.vix_phase)
                    m5.metric("Market", analysis.market_phase)

                    # Legs
                    st.subheader("Legs")
                    c1, c2, c3 = st.columns(3)
                    ce_pnl = pos.ce_sell_price - ce.get("ltp", 0)
                    pe_pnl = pos.pe_sell_price - pe.get("ltp", 0)
                    with c1:
                        st.caption(pos.ce_symbol)
                        st.metric("CE LTP", f"₹{ce.get('ltp', 0):.2f}", f"sold {pos.ce_sell_price:.2f}")
                        st.metric("CE P&L", f"{ce_pnl:+.2f} pts",
                                  delta_color="normal" if ce_pnl >= 0 else "inverse")
                    with c2:
                        st.caption(pos.pe_symbol)
                        st.metric("PE LTP", f"₹{pe.get('ltp', 0):.2f}", f"sold {pos.pe_sell_price:.2f}")
                        st.metric("PE P&L", f"{pe_pnl:+.2f} pts",
                                  delta_color="normal" if pe_pnl >= 0 else "inverse")
                    with c3:
                        st.caption("Combined")
                        st.metric("Net P&L", f"₹{analysis.net_pnl_inr:+,.0f}",
                                  f"{analysis.net_pnl_pts:+.2f} pts")
                        st.metric("Decay / Delta",
                                  f"{analysis.premium_decayed_pct:.1f}% / {analysis.net_delta:+.2f}",
                                  analysis.delta_bias)

                    # Health
                    st.subheader("Health")
                    h1, h2, h3, h4 = st.columns(4)
                    h1.metric("Tested Leg", analysis.nearest_itm_leg)
                    h2.metric("PE ITM by", f"{analysis.pe_itm_by:.0f} pts")
                    h3.metric("DTE", analysis.days_to_expiry)
                    h4.metric("Stop", "🚨 YES" if analysis.stop_triggered else "✅ No")

                    with st.expander("📊 Expiry Scenarios"):
                        rows = [
                            {"Scenario": s.label, "NIFTY": f"{s.nifty_level:,.0f}",
                             "P&L (INR)": f"{s.net_pnl_inr:+,.0f}"}
                            for s in analysis.scenarios
                        ]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Analysis error: {e}")
                    import traceback
                    st.code(traceback.format_exc(), language="text")


    # ────────────────────────────────
    # TAB: Analyze & Act
    # ────────────────────────────────
    with tab_analyze:
        if not _active_positions:
            st.info("No active positions. Register one first.")
        else:
            pos_opts = {
                f"#{p.id} | {p.underlying} {p.display_strike} [{p.expiry}]": p
                for p in _active_positions
            }
            sel_label = st.selectbox(
                "Select Position", list(pos_opts.keys()), key="analyze_pos_select"
            )
            pos = pos_opts[sel_label]
            st.caption(
                f"{pos.underlying} {pos.display_strike}  |  Expiry {pos.expiry}  |  "
                f"{pos.lot_size} x {pos.lots} lot(s)  |  "
                f"CE sold @ {pos.ce_sell_price}  |  PE sold @ {pos.pe_sell_price}"
            )

            if st.button(
                "🤖 Run Full Analysis + LLM Recommendation",
                type="primary", use_container_width=True,
            ):
                st.session_state.pop(f"straddle_state_{pos.id}", None)
                progress = st.progress(0)
                status_ph = st.empty()

                from trading.options.straddle.graph import (
                    fetch_market_data_node, analyze_position_node,
                    generate_action_node, validate_action_node,
                )

                state = {
                    "position_id": pos.id, "underlying": pos.underlying,
                    "strike": pos.strike, "expiry": pos.expiry.isoformat(),
                    "lot_size": pos.lot_size, "lots": pos.lots,
                    "ce_symbol": pos.ce_symbol, "ce_token": pos.ce_token,
                    "pe_symbol": pos.pe_symbol, "pe_token": pos.pe_token,
                    "ce_sell_price": pos.ce_sell_price, "pe_sell_price": pos.pe_sell_price,
                    "nifty_candles": None, "market_snapshot": None, "analysis": None,
                    "recommended_action": None, "planner_raw": None,
                    "action_approved": None, "validation_result": None,
                    "execution_result": None, "journal_id": None, "error": None,
                }

                status_ph.info("📡 Fetching live data + 5-min candles...")
                state.update(fetch_market_data_node(state))
                progress.progress(25)

                if state.get("error"):
                    st.error(state["error"])
                    st.stop()

                status_ph.info("🔢 Analyzing position — P&L, delta, scenarios...")
                state.update(analyze_position_node(state))
                progress.progress(50)

                status_ph.info("🤖 @OptionsStrategist thinking (Claude)...")
                state.update(generate_action_node(state))
                progress.progress(75)

                status_ph.info("🛡 @RiskGuard validating...")
                state.update(validate_action_node(state))
                progress.progress(100)
                status_ph.success("Analysis complete.")

                st.session_state[f"straddle_state_{pos.id}"] = state
                if state.get("market_snapshot"):
                    st.session_state[f"monitor_snap_{pos.id}"] = state["market_snapshot"]
                    st.session_state[f"monitor_snap_ts_{pos.id}"] = datetime.now()

            # Display persisted results
            saved_state = st.session_state.get(f"straddle_state_{pos.id}")
            if saved_state:
                analysis_d = saved_state.get("analysis") or {}
                action = saved_state.get("recommended_action") or {}
                validation = saved_state.get("validation_result") or {}

                if analysis_d:
                    st.code(analysis_d.get("summary_text", ""), language=None)

                if action:
                    st.subheader("LLM Recommendation")
                    ac1, ac2, ac3 = st.columns(3)
                    ac1.metric("Action", action.get("action", "?"))
                    ac2.metric("Urgency", action.get("urgency", "?"))
                    ac3.metric("Confidence", f"{action.get('confidence', 0):.0%}")

                    ac4, ac5 = st.columns(2)
                    ac4.metric("CE Leg", action.get("ce_action", "?"))
                    ac5.metric("PE Leg", action.get("pe_action", "?"))

                    if action.get("roll_to_strike"):
                        st.info(f"Roll to strike: **{action['roll_to_strike']}** (close tested leg, sell new one at this strike)")
                    if action.get("pe_stop_loss"):
                        st.info(f"PE Stop: close if PE > ₹{action['pe_stop_loss']:.2f}")
                    if action.get("pe_target"):
                        st.info(f"PE Target: close if PE < ₹{action['pe_target']:.2f}")
                    if action.get("hedge_lots", 0) > 0:
                        st.info(f"Hedge: {action['hedge_side']} {action['hedge_lots']}L NIFTY Futures")

                    st.markdown(f"**Reasoning:** {action.get('reasoning', '')}")
                    st.caption(f"Key risk: {action.get('key_risk', '')}")

                approved = validation.get("approved", False)
                if validation:
                    if approved:
                        st.success(f"🛡 @RiskGuard: **APPROVED** — {validation.get('reason', '')}")
                    else:
                        st.error(f"🛡 @RiskGuard: **REJECTED** — {validation.get('reason', '')}")
                    if validation.get("override_action"):
                        st.warning(f"Action overridden to: {validation['override_action']}")

                # Adjustment plan context
                if analysis_d:
                    combined_sold = analysis_d.get("combined_sold", 1)
                    combined_current = analysis_d.get("combined_current", 0)
                    severity = combined_current / combined_sold if combined_sold > 0 else 0
                    tested_leg = analysis_d.get("nearest_itm_leg", "?")
                    is_uw = analysis_d.get("is_underwater", False)
                    pnl_inr = analysis_d.get("net_pnl_inr", 0)
                    decay_pct = analysis_d.get("premium_decayed_pct", 0)

                    with st.expander("Adjustment Playbook (what happens if position goes underwater)"):
                        st.markdown(f"""
**Current state:** {'UNDERWATER' if is_uw else 'PROFITABLE'} | Severity: **{severity:.2f}x** sold premium | Tested leg: **{tested_leg}** | Decay: {decay_pct:.1f}%

| If severity reaches... | Action | What happens |
|---|---|---|
| {'**→**' if severity <= 1.3 and is_uw else ''} ≤1.3x | **ROLL {tested_leg}** | Close tested leg, sell new one at current ATM. Collect fresh premium. |
| {'**→**' if 1.3 < severity <= 1.5 and is_uw else ''} 1.3x–1.5x | **CLOSE {tested_leg} only** | Cut the loser, keep winning leg for theta decay to ~0. |
| {'**→**' if severity > 1.5 else ''} >1.5x | **CLOSE BOTH** (hard stop) | Full exit. Position blown. |

**0 DTE theta protection:** Before 2:30 PM, if position is profitable and <70% decayed, the validator blocks any premature CLOSE_BOTH from the LLM.
                        """)

                        if not is_uw:
                            nifty_spot = analysis_d.get("nifty_spot", 0)
                            lower_be = analysis_d.get("strike", 0) - combined_sold
                            upper_be = analysis_d.get("strike", 0) + combined_sold
                            st.caption(
                                f"Breakeven range: {lower_be:,.0f} — {upper_be:,.0f} "
                                f"(NIFTY at {nifty_spot:,.0f}, {abs(nifty_spot - analysis_d.get('strike', 0)):.0f} pts from strike)"
                            )

                # Execute
                if approved and action.get("action") not in ("HOLD", "MONITOR"):
                    st.markdown("---")

                    can_execute = True
                    if TRADING_MODE == "live":
                        confirmed = st.checkbox(
                            f"I confirm: send LIVE order — "
                            f"{action.get('action')} on {pos.underlying} {pos.display_strike}",
                            key=f"exec_confirm_{pos.id}",
                        )
                        if not confirmed:
                            st.warning("Check the box above to enable Execute in LIVE mode.")
                            can_execute = False

                    if can_execute and st.button(
                        f"⚡ Execute: {action.get('action')} ({TRADING_MODE.upper()} mode)",
                        type="primary", use_container_width=True,
                        key=f"exec_btn_{pos.id}",
                    ):
                        from trading.options.straddle.graph import (
                            execute_action_node, journal_action_node,
                        )
                        with st.spinner("Executing orders..."):
                            exec_state = dict(saved_state)
                            exec_state.update(execute_action_node(exec_state))
                            exec_state.update(journal_action_node(exec_state))

                        exec_r = exec_state.get("execution_result") or {}
                        if exec_r.get("success"):
                            actions_list = exec_r.get("actions_taken", [])
                            st.success(
                                f"✅ [{exec_r.get('mode', 'paper').upper()}] "
                                f"{' | '.join(actions_list)}"
                            )
                            for order in exec_r.get("orders", []):
                                if order.get("order_id"):
                                    st.caption(f"Order ID: {order['order_id']}  "
                                               f"Fill: {order.get('fill_price', 0):.2f}")
                            # Show roll/reenter details
                            for act in actions_list:
                                if "SOLD_NEW" in act:
                                    st.info(f"New leg opened: {act}")
                            for k in [f"straddle_state_{pos.id}", f"monitor_snap_{pos.id}",
                                      f"monitor_snap_ts_{pos.id}"]:
                                st.session_state.pop(k, None)
                        else:
                            st.error(f"Execution FAILED: {exec_r.get('message', 'unknown error')}")
                        st.rerun()

    # ────────────────────────────────
    # TAB: Payoff & Greeks
    # ────────────────────────────────
    with tab_payoff:
        if not _active_positions:
            st.info("No active positions.")
        else:
            pos_opts_pf = {
                f"#{p.id} | {p.underlying} {p.display_strike} [{p.expiry}]": p
                for p in _active_positions
            }
            sel_pf = st.selectbox("Select Position", list(pos_opts_pf.keys()), key="payoff_pos")
            pos = pos_opts_pf[sel_pf]

            # Get latest analysis data
            snap_data = st.session_state.get(f"monitor_snap_{pos.id}", {})
            nifty_spot = snap_data.get("nifty", {}).get("ltp", 0) if snap_data else 0

            if nifty_spot == 0:
                st.warning("Fetch live data first (go to Monitor tab and refresh).")
                nifty_spot = pos.ce_strike_actual  # fallback

            st.subheader("Payoff Diagram")
            # Use midpoint of CE/PE strikes for asymmetric positions
            diagram_strike = (pos.ce_strike_actual + pos.pe_strike_actual) // 2
            render_payoff_diagram(
                strike=diagram_strike,
                ce_sell=pos.ce_sell_price,
                pe_sell=pos.pe_sell_price,
                lot_size=pos.lot_size,
                lots=pos.lots,
                nifty_spot=nifty_spot,
            )

            # Greeks summary
            st.subheader("Greeks (Approximate)")
            if snap_data:
                ce_data = snap_data.get("ce", {})
                pe_data = snap_data.get("pe", {})
                vix_data = snap_data.get("vix", {})

                analysis = analyze_straddle(
                    underlying=pos.underlying, strike=pos.ce_strike_actual,
                    expiry=pos.expiry.isoformat(), lot_size=pos.lot_size, lots=pos.lots,
                    ce_sell_price=pos.ce_sell_price, pe_sell_price=pos.pe_sell_price,
                    ce_ltp=ce_data.get("ltp", 0), pe_ltp=pe_data.get("ltp", 0),
                    nifty_spot=nifty_spot, nifty_prev_close=snap_data.get("nifty", {}).get("prev_close", 0),
                    vix_current=vix_data.get("ltp", 0), vix_prev_close=vix_data.get("prev_close", 0),
                    candles=[],
                )

                g1, g2, g3, g4 = st.columns(4)
                g1.metric("CE Delta (short)", f"{analysis.ce_delta:+.3f}")
                g2.metric("PE Delta (short)", f"{analysis.pe_delta:+.3f}")
                g3.metric("Net Delta", f"{analysis.net_delta:+.3f}")
                g4.metric("Delta Bias", analysis.delta_bias)

                g5, g6, g7, g8 = st.columns(4)
                impact = abs(analysis.net_delta) * 100 * pos.lot_size * pos.lots
                g5.metric("100pt Impact", f"₹{impact:,.0f}")
                g6.metric("Premium Decay", f"{analysis.premium_decayed_pct:.1f}%")
                g7.metric("DTE", analysis.days_to_expiry)
                g8.metric("VIX", f"{vix_data.get('ltp', 0):.2f}")

            # Premium Decay Chart
            st.subheader("Premium Decay Tracker")
            decay_key = f"premium_decay_{pos.id}"
            decay_data = st.session_state.get(decay_key, [])
            render_premium_decay_chart(decay_data)

    # ────────────────────────────────
    # TAB: Register
    # ────────────────────────────────
    with tab_register:
        st.subheader("Register a New Straddle Position")
        st.caption("Enter the details of the straddle you have already sold.")

        # Token Lookup — auto-detect ATM strike from live NIFTY
        _nifty_for_atm = _options_svc.fetch_nifty_spot()
        _atm_default = find_atm_strike(_nifty_for_atm.get("ltp", 23200)) if _nifty_for_atm.get("ltp") else 23200

        st.markdown("#### 🔍 Token Lookup")
        tl1, tl2 = st.columns(2)
        tl_und = tl1.text_input("Underlying", value="NIFTY", key="tl_und")
        tl_strike = int(tl2.number_input("Strike", value=_atm_default, step=50, key="tl_strike"))
        if _nifty_for_atm.get("ltp"):
            tl2.caption(f"NIFTY: {_nifty_for_atm['ltp']:,.1f} → ATM: {_atm_default}")

        _exp_wday = 1  # Tuesday
        _today = date.today()
        _days_to = (_exp_wday - _today.weekday()) % 7
        _first = _today + timedelta(days=_days_to)
        _exp_opts = []
        for _i in range(5):
            _d = _first + timedelta(weeks=_i)
            _angel = _d.strftime("%d%b%y").upper()
            _prefix = "This" if _i == 0 else "Next" if _i == 1 else ""
            _label = f"{(_prefix + ' ') if _prefix else ''}{_d.strftime('%d %b')} ({_angel})"
            _exp_opts.append((_label, _angel))

        tl3, tl4, tl5 = st.columns([2, 2, 1])
        _exp_labels = ["— pick week —"] + [lbl for lbl, _ in _exp_opts]
        _exp_pick = tl3.selectbox("Expiry (quick-pick)", _exp_labels, key="tl_expiry_pick")
        if _exp_pick != "— pick week —":
            st.session_state["tl_expiry"] = next(af for lbl, af in _exp_opts if lbl == _exp_pick)
        tl_expiry = tl4.text_input("Expiry (or type)", placeholder="10MAR26", key="tl_expiry")
        tl_type = tl5.selectbox("Type", ["Both", "CE", "PE"], key="tl_type")

        if st.button("🔍 Find & Auto-fill Tokens", key="token_lookup", use_container_width=True):
            if not tl_expiry.strip():
                st.warning("Enter an expiry (e.g. 10MAR26)")
            else:
                with st.spinner("Searching NFO master..."):
                    types = ["CE", "PE"] if tl_type == "Both" else [tl_type]
                    found_any = False
                    for ot in types:
                        result = find_option_token(tl_und, tl_strike, tl_expiry.strip().upper(), ot)
                        if result:
                            sym, tok = result
                            st.session_state[f"reg_{ot.lower()}_symbol"] = sym
                            st.session_state[f"reg_{ot.lower()}_token"] = tok
                            st.success(f"**{ot}** -> `{sym}`  |  token: `{tok}`")
                            found_any = True
                        else:
                            st.warning(f"Not found: {tl_und} {tl_strike} {tl_expiry} {ot}")
                    if found_any:
                        st.session_state["reg_underlying"] = tl_und
                        st.session_state["reg_strike"] = tl_strike

        st.markdown("---")
        st.markdown("#### Register Position")

        _reg_und = st.session_state.get("reg_underlying", "NIFTY")
        _reg_strike = st.session_state.get("reg_strike", 24200)
        _ce_symbol = st.session_state.get("reg_ce_symbol", "")
        _ce_token = st.session_state.get("reg_ce_token", "")
        _pe_symbol = st.session_state.get("reg_pe_symbol", "")
        _pe_token = st.session_state.get("reg_pe_token", "")
        _default_lot_size = 75 if "NIFTY" in _reg_und.upper() and "BANK" not in _reg_und.upper() else \
                            15 if "BANK" in _reg_und.upper() else 25

        with st.form("register_straddle"):
            r1, r2, r3 = st.columns(3)
            underlying = r1.text_input("Underlying", value=_reg_und).upper()
            strike = r2.number_input("Strike", value=_reg_strike, step=50, min_value=10000)
            expiry = r3.date_input("Expiry Date", value=date.today())

            r4, r5 = st.columns(2)
            lots = r4.number_input("Lots", value=1, min_value=1, max_value=100)
            lot_size = r5.number_input("Lot Size", value=_default_lot_size, min_value=1)

            st.markdown("**CE Leg (short call)**")
            c1, c2, c3 = st.columns(3)
            ce_symbol = c1.text_input("CE Symbol", value=_ce_symbol)
            ce_token_val = c2.text_input("CE Token", value=_ce_token)
            ce_sell_price = c3.number_input("CE Sell Price (pts)", value=0.0, min_value=0.0, step=0.05)

            st.markdown("**PE Leg (short put)**")
            p1, p2, p3 = st.columns(3)
            pe_symbol = p1.text_input("PE Symbol", value=_pe_symbol)
            pe_token_val = p2.text_input("PE Token", value=_pe_token)
            pe_sell_price = p3.number_input("PE Sell Price (pts)", value=0.0, min_value=0.0, step=0.05)

            combined = ce_sell_price + pe_sell_price
            if combined > 0:
                st.info(
                    f"Combined premium: **{combined:.2f} pts  "
                    f"= ₹{combined * lot_size * lots:,.0f}**"
                )

            submitted = st.form_submit_button("Register Position", type="primary")
            if submitted:
                if not ce_symbol or not pe_symbol or not ce_token_val or not pe_token_val:
                    st.error("CE and PE symbol + token required.")
                elif combined == 0:
                    st.error("Enter CE and PE sell prices.")
                else:
                    new_pos = StraddlePosition.objects.create(
                        underlying=underlying, strike=strike, expiry=expiry,
                        lot_size=lot_size, lots=lots,
                        ce_symbol=ce_symbol, ce_token=ce_token_val,
                        ce_sell_price=ce_sell_price,
                        pe_symbol=pe_symbol, pe_token=pe_token_val,
                        pe_sell_price=pe_sell_price,
                        trade_date=date.today(),
                    )
                    for k in ["reg_underlying", "reg_strike", "reg_ce_symbol",
                              "reg_ce_token", "reg_pe_symbol", "reg_pe_token"]:
                        st.session_state.pop(k, None)
                    st.success(
                        f"✅ Registered #{new_pos.id} | {underlying} {strike} [{expiry}]  "
                        f"| Premium: {combined:.2f} pts = ₹{new_pos.total_premium_sold:,.0f}"
                    )
                    st.rerun()

    # ────────────────────────────────
    # TAB: History
    # ────────────────────────────────
    with tab_history:
        all_positions = list(StraddlePosition.objects.order_by("-opened_at")[:50])
        if not all_positions:
            st.info("No straddle positions recorded yet.")
        else:
            # ── Day-wise P&L Performance ──
            st.subheader("Day-wise P&L")
            from collections import defaultdict
            daily_pnl = defaultdict(lambda: {"total": 0, "realized": 0, "unrealized": 0, "trades": 0, "wins": 0})
            for p in all_positions:
                d = p.trade_date.isoformat()
                daily_pnl[d]["total"] += p.total_pnl
                daily_pnl[d]["realized"] += p.realized_pnl
                daily_pnl[d]["unrealized"] += p.current_pnl_inr
                daily_pnl[d]["trades"] += 1
                if p.total_pnl > 0:
                    daily_pnl[d]["wins"] += 1

            day_rows = []
            cumulative = 0
            for d in sorted(daily_pnl.keys()):
                v = daily_pnl[d]
                cumulative += v["total"]
                wr = (v["wins"] / v["trades"] * 100) if v["trades"] else 0
                day_rows.append({
                    "Date": d,
                    "Trades": v["trades"],
                    "Wins": v["wins"],
                    "WR%": f"{wr:.0f}%",
                    "Realized": f"{v['realized']:+,.0f}",
                    "Unrealized": f"{v['unrealized']:+,.0f}",
                    "Day P&L": f"{v['total']:+,.0f}",
                    "Cumulative": f"{cumulative:+,.0f}",
                })
            st.dataframe(pd.DataFrame(day_rows), use_container_width=True, hide_index=True)

            # Cumulative P&L chart
            if len(day_rows) >= 2:
                cum_values = []
                running = 0
                for d in sorted(daily_pnl.keys()):
                    running += daily_pnl[d]["total"]
                    cum_values.append({"Date": d, "P&L (INR)": running})
                st.line_chart(pd.DataFrame(cum_values).set_index("Date"), height=200)

            # ── All Positions Table ──
            st.subheader("All Positions")
            rows = []
            for p in all_positions:
                rows.append({
                    "ID": p.id, "Underlying": p.underlying, "Strike": p.display_strike,
                    "Expiry": str(p.expiry), "Lots": p.lots,
                    "CE Sold": p.ce_sell_price, "PE Sold": p.pe_sell_price,
                    "Status": p.status, "Action": p.action_taken,
                    "Realized": f"{p.realized_pnl:+,.0f}",
                    "Unrealized": f"{p.current_pnl_inr:+,.0f}",
                    "Total P&L": f"{p.total_pnl:+,.0f}",
                    "Opened": p.opened_at.strftime("%Y-%m-%d %H:%M"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── Trade Ledger + Management Log ──
            st.subheader("Position Detail")
            pos_ids = {f"#{p.id} {p.underlying} {p.display_strike} [{p.status}]": p for p in all_positions}
            sel = st.selectbox("Select position", list(pos_ids.keys()), key="hist_pos")
            sel_pos = pos_ids[sel]

            if sel_pos.management_log:
                log = sel_pos.management_log

                # ── Trade Ledger (actions with executions) ──
                trades = [e for e in log if e.get("executed") and e.get("exec")]
                if trades:
                    st.markdown("**Trade Ledger**")
                    total_collected = 0
                    total_paid = 0
                    ledger_rows = []
                    for i, e in enumerate(trades, 1):
                        collected = e.get("premium_collected", 0)
                        paid = e.get("premium_paid", 0)
                        total_collected += collected
                        total_paid += paid
                        net = collected - paid
                        exec_str = " | ".join(e.get("exec", []))

                        ledger_rows.append({
                            "#": i,
                            "Time": e.get("time", "?"),
                            "Action": e.get("action", "?"),
                            "NIFTY": f"{e.get('nifty', 0):,.0f}",
                            "Executions": exec_str,
                            "Collected": f"₹{collected:,.0f}" if collected else "—",
                            "Paid": f"₹{paid:,.0f}" if paid else "—",
                            "Realized": f"₹{e.get('realized_pnl', 0):+,.0f}",
                            "Total P&L": f"₹{e.get('pnl_inr', 0):+,.0f}",
                        })
                    st.dataframe(pd.DataFrame(ledger_rows), use_container_width=True, hide_index=True)

                    # Summary stats
                    lc1, lc2, lc3, lc4 = st.columns(4)
                    lc1.metric("Trades", len(trades))
                    lc2.metric("Premium Collected", f"₹{total_collected:,.0f}")
                    lc3.metric("Premium Paid", f"₹{total_paid:,.0f}")
                    lc4.metric("Realized P&L", f"₹{sel_pos.realized_pnl:+,.0f}")

                # ── Straddle Range History ──
                range_entries = [e for e in log if e.get("nifty", 0) > 0]
                if range_entries:
                    st.markdown("**Position Range Over Time**")
                    range_rows = []
                    for e in range_entries:
                        ce_s = e.get("ce_sell", 0)
                        pe_s = e.get("pe_sell", 0)
                        ce_st = e.get("ce_strike", sel_pos.strike)
                        pe_st = e.get("pe_strike", sel_pos.strike)
                        combined = ce_s + pe_s
                        if combined > 0:
                            range_rows.append({
                                "Time": e.get("time", "?"),
                                "Action": e.get("action", "?"),
                                "NIFTY": f"{e.get('nifty', 0):,.0f}",
                                "CE@": ce_st,
                                "PE@": pe_st,
                                "CE Sell": f"{ce_s:.0f}" if ce_s else "—",
                                "PE Sell": f"{pe_s:.0f}" if pe_s else "—",
                                "Combined": f"{combined:.0f}",
                                "Range": f"{pe_st - combined:.0f} — {ce_st + combined:.0f}" if combined else "—",
                            })
                    if range_rows:
                        st.dataframe(pd.DataFrame(range_rows), use_container_width=True, hide_index=True)

                # ── Full Management Log ──
                with st.expander("Full Management Log"):
                    log_rows = [{
                        "Time": e.get("time", "?"),
                        "Action": e.get("action", "?"),
                        "NIFTY": f"{e.get('nifty', 0):,.0f}",
                        "P&L": f"₹{e.get('pnl_inr', 0):+,.0f}",
                        "Exec": "Y" if e.get("executed") else "—",
                        "Note": e.get("note", "")[:60],
                    } for e in log]
                    st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No management actions recorded yet.")


# ══════════════════════════════════════════════
# PAGE 5: JOURNAL & ANALYTICS
# ══════════════════════════════════════════════
elif page == "Journal & Analytics":
    st.header("Trade Journal & Analytics")

    tab_journal, tab_analytics, tab_notes = st.tabs(["📓 Journal", "📊 Analytics", "📝 Trader Notes"])

    # ── Journal Tab ──
    with tab_journal:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            sym_filter = st.text_input("Filter by symbol", "").strip().upper()
        with col_f2:
            status_filter = st.selectbox(
                "Status", ["ALL"] + [s[0] for s in TradeJournal.Status.choices],
            )
        with col_f3:
            days_back = st.slider("Days back", 1, 90, 30)

        qs = TradeJournal.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=days_back)
        ).order_by("-created_at")

        if sym_filter:
            qs = qs.filter(symbol__icontains=sym_filter)
        if status_filter != "ALL":
            qs = qs.filter(status=status_filter)

        entries = list(qs[:100])

        if not entries:
            st.info("No journal entries match your filters.")
        else:
            total = len(entries)
            wins = sum(1 for e in entries if e.pnl and e.pnl > 0)
            losses = sum(1 for e in entries if e.pnl and e.pnl < 0)
            total_pnl = sum(e.pnl for e in entries if e.pnl is not None)
            closed = wins + losses

            m_cols = st.columns(5)
            m_cols[0].metric("Total Trades", total)
            m_cols[1].metric("Wins", wins)
            m_cols[2].metric("Losses", losses)
            m_cols[3].metric("Win Rate", f"{wins/closed*100:.0f}%" if closed else "N/A")
            m_cols[4].metric("Total P&L", f"₹{total_pnl:+,.0f}")

            rows = []
            for e in entries:
                rows.append({
                    "ID": e.id,
                    "Date": e.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Symbol": e.symbol, "Side": e.side, "Qty": e.quantity,
                    "Entry": e.entry_price, "SL": e.stop_loss, "Target": e.target,
                    "Status": e.status, "Risk": "✅" if e.risk_approved else "❌",
                    "Confidence": f"{e.confidence:.0%}",
                    "P&L": f"₹{e.pnl:+,.0f}" if e.pnl is not None else "—",
                    "R:R": f"{e.risk_reward_ratio:.1f}",
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with st.expander("Trade Reasoning Details"):
                for e in entries[:10]:
                    st.markdown(f"**#{e.id} {e.side} {e.symbol}** ({e.status}): {e.reasoning}")
                    if e.risk_reason:
                        st.caption(f"Risk: {e.risk_reason}")
                    st.markdown("---")

    # ── Analytics Tab ──
    with tab_analytics:
        analytics = get_journal_analytics(30)

        if analytics["trades"] == 0:
            st.info("No closed trades in the last 30 days for analytics.")
        else:
            st.subheader("Performance Summary (30 days)")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Closed Trades", analytics["trades"])
            a2.metric("Max Win Streak", analytics["max_win_streak"])
            a3.metric("Max Loss Streak", analytics["max_loss_streak"])
            a4.metric("Current Streak", f"{analytics['current_streak']:+d}")

            # Per-symbol P&L chart
            st.subheader("P&L by Symbol")
            render_symbol_pnl_chart(analytics["symbols"])

            # Confidence calibration
            if analytics["calibration"]:
                st.subheader("Confidence Calibration")
                st.caption("Does higher confidence actually mean higher win rate?")
                cal_rows = [{
                    "Confidence": c["range"],
                    "Trades": c["count"],
                    "Win Rate": f"{c['win_rate']:.0f}%",
                } for c in analytics["calibration"]]
                st.dataframe(pd.DataFrame(cal_rows), use_container_width=True, hide_index=True)

    # ── Trader Notes Tab ──
    with tab_notes:
        st.subheader("Trader Notes")
        st.caption("Persistent per-ticker notes. Survives sessions.")

        note_symbol = st.text_input("Symbol", placeholder="HDFCBANK", key="note_sym").strip().upper()

        if note_symbol:
            try:
                existing = TraderNote.objects.get(symbol=note_symbol)
                current_note = existing.note
            except TraderNote.DoesNotExist:
                current_note = ""

            note_text = st.text_area(
                f"Notes for {note_symbol}",
                value=current_note,
                height=200,
                key=f"note_text_{note_symbol}",
            )

            if st.button("Save Note", key="save_note"):
                TraderNote.objects.update_or_create(
                    symbol=note_symbol,
                    defaults={"note": note_text},
                )
                st.success(f"Note saved for {note_symbol}")

        # Show all existing notes
        all_notes = TraderNote.objects.exclude(note="").order_by("symbol")
        if all_notes.exists():
            st.markdown("---")
            st.subheader("All Notes")
            for n in all_notes:
                with st.expander(f"{n.symbol} (updated {n.updated_at.strftime('%Y-%m-%d %H:%M')})"):
                    st.markdown(n.note)


# ══════════════════════════════════════════════
# PAGE 6: RISK CONTROL
# ══════════════════════════════════════════════
elif page == "Risk Control":
    st.header("Risk Control Panel")

    # ── Risk Status Banner ──
    risk = get_risk_utilization()
    risk_status = compute_risk_status(risk)

    status_colors = {"GREEN": "success", "YELLOW": "warning", "RED": "error"}
    status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}
    getattr(st, status_colors.get(risk_status, "info"))(
        f"## {status_emoji.get(risk_status, '⚪')} RISK STATUS: {risk_status}"
    )

    # ── Risk Gauges ──
    st.subheader("Risk Utilization")
    g1, g2, g3, g4 = st.columns(4)

    with g1:
        render_risk_gauge(
            "Daily Loss", risk["daily_loss"],
            risk["max_daily_loss"], " INR"
        )
    with g2:
        render_risk_gauge(
            "Capital Deployed", risk["capital_deployed"],
            risk["capital"], " INR"
        )
    with g3:
        render_risk_gauge(
            "Open Positions", float(risk["open_positions"]),
            float(risk["max_open_positions"]), ""
        )
    with g4:
        st.markdown("**Underwater Options**")
        if risk["underwater_options"] > 0:
            st.error(f"🚨 {risk['underwater_options']} position(s)")
        else:
            st.success("✅ None")
        st.caption(f"{risk['active_straddles']} active straddle(s)")

    st.markdown("---")

    # ── Emergency Controls ──
    st.subheader("Emergency Controls")

    ec1, ec2 = st.columns(2)

    with ec1:
        paused = is_ai_paused()
        if paused:
            st.error("AI Trading is **PAUSED**")
            if st.button("▶️ Resume AI Trading", use_container_width=True):
                resume_ai_trading()
                st.success("AI trading resumed.")
                st.rerun()
        else:
            st.success("AI Trading is **ACTIVE**")
            if st.button("⏸ Pause AI Trading", type="primary", use_container_width=True):
                pause_ai_trading()
                st.warning("AI trading paused.")
                st.rerun()

    with ec2:
        st.warning("**Force Close All** — Emergency only")
        if st.checkbox("I understand this will close ALL positions", key="force_close_confirm"):
            if st.button("🔴 FORCE CLOSE ALL POSITIONS", type="primary", use_container_width=True):
                # Close all active straddles
                closed_count = 0
                for pos in StraddlePosition.objects.filter(status="ACTIVE"):
                    pos.status = StraddlePosition.Status.CLOSED
                    pos.action_taken = StraddlePosition.ActionTaken.CLOSE_BOTH
                    pos.closed_at = datetime.now()
                    pos.save()
                    closed_count += 1

                # Mark today's equity trades as closed
                eq_count = TradeJournal.objects.filter(
                    status__in=["EXECUTED", "FILLED", "PAPER"],
                    trade_date=date.today(),
                ).update(status="CANCELLED")

                st.error(f"Force closed {closed_count} straddle(s) and {eq_count} equity position(s).")
                st.rerun()

    st.markdown("---")

    # ── Audit Log Stream ──
    st.subheader("Live Audit Stream")

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        event_filter = st.selectbox(
            "Event Type", ["ALL"] + [e[0] for e in AuditLog.EventType.choices],
        )
    with col_a2:
        audit_sym = st.text_input("Symbol filter", "").strip().upper()
    with col_a3:
        audit_days = st.slider("Days back", 1, 30, 7, key="audit_days")

    audit_qs = AuditLog.objects.filter(
        created_at__gte=timezone.now() - timedelta(days=audit_days)
    ).order_by("-created_at")

    if event_filter != "ALL":
        audit_qs = audit_qs.filter(event_type=event_filter)
    if audit_sym:
        audit_qs = audit_qs.filter(symbol__icontains=audit_sym)

    audit_entries = list(audit_qs[:50])

    if not audit_entries:
        st.info("No audit entries match your filters.")
    else:
        st.markdown(f"**{len(audit_entries)} entries** (last {audit_days} days)")

        for entry in audit_entries:
            icon = {
                "PLANNER_REQ": "📤", "PLANNER_RES": "📥", "PLANNER_ERR": "💥",
                "RISK_APPROVE": "✅", "RISK_REJECT": "❌",
                "EXECUTION": "⚡", "RECONCILE": "🔄",
            }.get(entry.event_type, "📋")

            ts = entry.created_at.strftime("%Y-%m-%d %H:%M:%S")
            header = f"{icon} **{entry.event_type}** | {entry.symbol or '—'} | {ts}"
            if entry.latency_ms:
                header += f" | {entry.latency_ms}ms"

            with st.expander(header, expanded=False):
                if entry.prompt:
                    st.markdown("**Prompt:**")
                    st.code(entry.prompt[:2000], language="text")
                if entry.response:
                    st.markdown("**Response:**")
                    st.code(entry.response[:2000], language="text")
                if entry.risk_details:
                    st.markdown("**Risk Details:**")
                    st.json(entry.risk_details)
                if entry.execution_details:
                    st.markdown("**Execution Details:**")
                    st.json(entry.execution_details)
                if entry.model_name:
                    st.caption(f"Model: {entry.model_name}")


# ══════════════════════════════════════════════
# PAGE 7: BACKTEST
# ══════════════════════════════════════════════
elif page == "Backtest":
    st.header("Backtest Engine")
    st.caption("Replay historical candles through planner + risk engine.")

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        bt_symbol = st.text_input("Symbol", value="MFSL", max_chars=20).upper()
    with col_b2:
        bt_capital = st.number_input("Initial Capital (INR)", value=500000, step=50000, min_value=10000)
    with col_b3:
        bt_source = st.radio("Data Source", ["Fetch from Broker", "Upload CSV", "Sample Data"])

    uploaded_file = None
    if bt_source == "Fetch from Broker":
        date_cols = st.columns(2)
        with date_cols[0]:
            bt_from_date = st.date_input("From Date", value=date.today() - timedelta(days=10), max_value=date.today())
        with date_cols[1]:
            bt_to_date = st.date_input("To Date", value=date.today() - timedelta(days=1), max_value=date.today())
        st.info(f"Will fetch daily OHLCV for **{bt_symbol}** from Angel One ({bt_from_date} -> {bt_to_date})")
    elif bt_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload OHLCV CSV", type=["csv"])
    else:
        st.info("Will generate 5 synthetic candles for quick testing.")

    if st.button("Run Backtest", type="primary", use_container_width=True):
        from trading.services.backtester import run_backtest

        candles = []

        if bt_source == "Fetch from Broker":
            with st.spinner(f"Fetching historical data for {bt_symbol}..."):
                try:
                    from trading.services.data_service import DataService
                    ds = DataService()
                    candles = ds.fetch_historical(
                        symbol=bt_symbol,
                        from_date=bt_from_date.strftime("%Y-%m-%d"),
                        to_date=bt_to_date.strftime("%Y-%m-%d"),
                        interval="FIVE_MINUTE",
                    )
                    if not candles:
                        st.warning(f"No data for {bt_symbol}.")
                        candles = None
                    else:
                        st.success(f"Fetched {len(candles)} trading days")
                except Exception as e:
                    st.error(f"Fetch failed: {e}")
                    candles = None
        elif bt_source == "Upload CSV" and uploaded_file is not None:
            import csv, io
            content = uploaded_file.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                candles.append({
                    "date": row["date"], "open": float(row["open"]),
                    "high": float(row["high"]), "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(float(row.get("volume", 0))),
                })
        elif bt_source == "Sample Data":
            import random
            random.seed(42)
            base = 1500.0
            for i in range(5):
                o = base + random.uniform(-20, 20)
                h = o + random.uniform(5, 30)
                l = o - random.uniform(5, 30)
                c = random.uniform(l, h)
                candles.append({
                    "date": f"2026-02-{20+i:02d}",
                    "open": round(o, 2), "high": round(h, 2),
                    "low": round(l, 2), "close": round(c, 2),
                    "volume": random.randint(50000, 200000),
                })
                base = c
        else:
            candles = None

        if candles:
            with st.spinner(f"Running backtest on {len(candles)} candles..."):
                try:
                    result = run_backtest(symbol=bt_symbol, candles=candles, initial_capital=bt_capital)
                    s = result["summary"]

                    st.subheader("Results")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total P&L", f"{s['total_pnl']:+,.0f} INR")
                    m2.metric("Win Rate", f"{s['win_rate']}%")
                    m3.metric("Return", f"{s['return_pct']:+.1f}%")
                    m4.metric("Max Drawdown", f"{s['max_drawdown_pct']:.1f}%")

                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("Trades Executed", s["trades_executed"])
                    m6.metric("Wins / Losses", f"{s['wins']}W / {s['losses']}L")
                    m7.metric("Skipped", s["trades_skipped"])
                    m8.metric("Final Capital", f"{s['final_capital']:,.0f}")

                    st.subheader("Trade Log")
                    trade_rows = []
                    for t in result["trades"]:
                        if t["action"] == "TRADED":
                            sim = t["simulation"]
                            trade_rows.append({
                                "Date": t["candle"], "Side": sim["side"],
                                "Entry": sim["entry"], "SL": sim["sl"],
                                "Target": sim["target"], "Qty": sim["qty"],
                                "Exit": sim["exit_price"], "Outcome": sim["outcome"],
                                "P&L": t["pnl"], "Capital": t["capital_after"],
                            })
                        else:
                            trade_rows.append({
                                "Date": t["candle"], "Side": "—",
                                "Entry": 0, "SL": 0, "Target": 0, "Qty": 0, "Exit": 0,
                                "Outcome": t["action"], "P&L": 0, "Capital": 0,
                            })

                    if trade_rows:
                        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

                    equity_data = [bt_capital]
                    for t in result["trades"]:
                        if t["action"] == "TRADED":
                            equity_data.append(t["capital_after"])
                    if len(equity_data) > 1:
                        st.subheader("Equity Curve")
                        st.line_chart(equity_data)

                except Exception as e:
                    st.error(f"Backtest failed: {e}")


# ══════════════════════════════════════════════
# PAGE 8: SETTINGS
# ══════════════════════════════════════════════
elif page == "Settings":
    st.header("System Settings")

    tab_config, tab_strategies, tab_portfolio, tab_risk_test = st.tabs([
        "⚙️ Config", "📋 Strategies", "💰 Portfolio", "🧪 Risk Test"
    ])

    with tab_config:
        st.subheader("Current Configuration")
        config_data = {
            "Trading Mode": TRADING_MODE.upper(),
            "LLM Model": os.getenv("LLM_MODEL", "not set"),
            "Anthropic Key": "✅ Set" if os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-") else "❌ Not set",
            "Max Risk/Trade": f"{os.getenv('MAX_RISK_PER_TRADE_PCT', '1.0')}%",
            "Max Daily Loss": f"{os.getenv('MAX_DAILY_LOSS_PCT', '3.0')}%",
            "Max Position Size": f"{os.getenv('MAX_POSITION_SIZE_PCT', '10.0')}%",
            "Min Confidence": "0.6",
            "Max Open Positions": "3",
            "Min R:R Ratio": "1.5",
            "Default Capital": f"₹{os.getenv('DEFAULT_CAPITAL', '500000')}",
            "Database": os.getenv("DB_ENGINE", "sqlite3"),
        }
        for k, v in config_data.items():
            st.markdown(f"**{k}:** `{v}`")

    with tab_strategies:
        st.subheader("Strategy Documents")
        st.caption("Injected as RAG context into the planner agent.")

        strategies = StrategyDoc.objects.all()
        if not strategies.exists():
            st.info("No strategies. Seed defaults below.")
        else:
            for s in strategies:
                active_badge = "🟢" if s.is_active else "⚪"
                with st.expander(f"{active_badge} [{s.category}] {s.title}"):
                    st.markdown(s.content)
                    col_a, col_b = st.columns(2)
                    if col_a.button("Deactivate" if s.is_active else "Activate", key=f"toggle_{s.id}"):
                        s.is_active = not s.is_active
                        s.save()
                        st.rerun()
                    if col_b.button("Delete", key=f"del_{s.id}"):
                        s.delete()
                        st.rerun()

        st.markdown("---")
        with st.form("add_strategy"):
            title = st.text_input("Title")
            category = st.selectbox("Category", [c[0] for c in StrategyDoc.Category.choices])
            content = st.text_area("Content", height=120)
            submitted = st.form_submit_button("Add Strategy")
            if submitted and title and content:
                StrategyDoc.objects.create(title=title, category=category, content=content)
                st.success(f"Added: [{category}] {title}")
                st.rerun()

        if st.button("Seed Default Strategies"):
            from trading.management.commands.run_trading_agent import Command
            cmd = Command()
            cmd.stdout = type("FakeOut", (), {"write": lambda self, x: None})()
            cmd._seed_strategies()
            st.success("Default strategies seeded.")
            st.rerun()

    with tab_portfolio:
        st.subheader("Portfolio Management")
        try:
            snap = PortfolioSnapshot.objects.latest()
            st.markdown(f"**Current Snapshot — {snap.snapshot_date}**")

            p_cols = st.columns(3)
            p_cols[0].metric("Total Capital", f"₹{snap.capital:,.0f}")
            p_cols[1].metric("Invested", f"₹{snap.invested:,.0f}")
            p_cols[2].metric("Available Cash", f"₹{snap.available_cash:,.0f}")

            p_cols2 = st.columns(4)
            p_cols2[0].metric("Today's P&L", f"₹{snap.daily_pnl:+,.0f}")
            p_cols2[1].metric("Total P&L", f"₹{snap.total_pnl:+,.0f}")
            p_cols2[2].metric("Today's Losses", f"₹{snap.daily_loss:,.0f}")
            p_cols2[3].metric("Open Positions", snap.open_positions)
        except PortfolioSnapshot.DoesNotExist:
            st.warning("No portfolio snapshot exists.")

        st.markdown("---")
        with st.form("init_portfolio"):
            capital = st.number_input("Capital (INR)", min_value=10000, value=500000, step=50000)
            if st.form_submit_button("Create Snapshot"):
                PortfolioSnapshot.objects.create(
                    capital=capital, invested=0, available_cash=capital,
                    daily_pnl=0, total_pnl=0, daily_loss=0,
                    open_positions=0, open_positions_count=0,
                    snapshot_date=date.today(),
                )
                st.success(f"Portfolio initialized: ₹{capital:,.0f}")
                st.rerun()

        st.markdown("---")
        st.subheader("Snapshot History")
        snapshots = PortfolioSnapshot.objects.order_by("-snapshot_date")[:30]
        if snapshots:
            snap_rows = [{
                "Date": s.snapshot_date.isoformat(),
                "Capital": f"₹{s.capital:,.0f}",
                "Daily P&L": f"₹{s.daily_pnl:+,.0f}",
                "Total P&L": f"₹{s.total_pnl:+,.0f}",
                "Positions": s.open_positions,
            } for s in snapshots]
            st.dataframe(pd.DataFrame(snap_rows), use_container_width=True, hide_index=True)

    with tab_risk_test:
        st.subheader("Risk Engine Test")
        st.caption("Validate a hypothetical trade against the risk engine.")

        with st.form("risk_test"):
            r_cols = st.columns(2)
            test_symbol = r_cols[0].text_input("Symbol", "MFSL")
            test_side = r_cols[1].selectbox("Side", ["BUY", "SELL"])

            r_cols2 = st.columns(3)
            test_entry = r_cols2[0].number_input("Entry", value=1000.0, step=1.0)
            test_sl = r_cols2[1].number_input("Stop Loss", value=990.0, step=1.0)
            test_target = r_cols2[2].number_input("Target", value=1025.0, step=1.0)

            r_cols3 = st.columns(2)
            test_qty = r_cols3[0].number_input("Quantity", value=50, min_value=1)
            test_conf = r_cols3[1].slider("Confidence", 0.0, 1.0, 0.7, 0.05)

            r_cols4 = st.columns(3)
            test_capital = r_cols4[0].number_input("Capital", value=500000, step=50000)
            test_daily_loss = r_cols4[1].number_input("Daily Loss (INR)", value=0, step=1000, min_value=0)
            test_open_pos = r_cols4[2].number_input("Open Positions", value=0, step=1, min_value=0, max_value=10)

            if st.form_submit_button("Validate"):
                plan = {
                    "symbol": test_symbol, "side": test_side,
                    "entry_price": test_entry, "stop_loss": test_sl,
                    "target": test_target, "quantity": test_qty,
                    "confidence": test_conf,
                }
                approved, reason, details = validate_trade(
                    plan, capital=test_capital,
                    daily_loss=float(test_daily_loss),
                    open_positions=int(test_open_pos),
                )

                if approved:
                    st.success(f"✅ APPROVED: {reason}")
                else:
                    st.error(f"❌ REJECTED: {reason}")

                if details:
                    d_cols = st.columns(3)
                    d_cols[0].metric("Risk Amount", f"₹{details.get('risk_amount', 0):.0f}")
                    d_cols[1].metric("Risk % Capital", f"{details.get('risk_pct_of_capital', 0):.1f}%")
                    d_cols[2].metric("R:R Ratio", f"{details.get('risk_reward_ratio', 0):.1f}")
