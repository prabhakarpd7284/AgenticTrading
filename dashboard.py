"""
Agentic Trading Dashboard — Streamlit UI.

Run:  streamlit run dashboard.py

Connects to the trading graph, shows live workflow,
displays journal, portfolio, strategies, and audit log.
"""
import os
import sys
import json
from datetime import date, datetime, timedelta
from django.utils import timezone

# ── Django bootstrap ──
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import django
django.setup()

import streamlit as st
import pandas as pd

from trading.models import TradeJournal, StrategyDoc, PortfolioSnapshot, AuditLog
from trading.services.risk_engine import validate_trade
from trading.services.broker_service import BrokerService

# ── Page config ──
st.set_page_config(
    page_title="Agentic Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TRADING_MODE = os.getenv("TRADING_MODE", "paper")

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Agentic Trading")

    mode_color = "🟢" if TRADING_MODE == "paper" else "🔴"
    st.markdown(f"**Mode:** {mode_color} `{TRADING_MODE.upper()}`")

    # Portfolio summary
    try:
        snap = PortfolioSnapshot.objects.latest()
        st.markdown("---")
        st.markdown("**Portfolio**")
        col1, col2 = st.columns(2)
        col1.metric("Capital", f"₹{snap.capital:,.0f}")
        col2.metric("Available", f"₹{snap.available_cash:,.0f}")
        col1.metric("Today P&L", f"₹{snap.daily_pnl:+,.0f}")
        col2.metric("Positions", snap.open_positions)
    except Exception:
        st.info("No portfolio. Initialize below.")

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Trade", "Straddle", "Journal", "Strategies", "Portfolio", "Backtest", "Audit Log", "Settings"],
        index=0,
    )


# ──────────────────────────────────────────────
# PAGE: Trade
# ──────────────────────────────────────────────
if page == "Trade":
    st.header("Execute Trade Workflow")
    st.caption("Orchestrator → Data → RAG → Planner → Risk → Execute → Journal")

    col_input, col_config = st.columns([2, 1])

    with col_input:
        intent = st.text_area(
            "Trading Intent",
            placeholder="e.g. Plan a BUY trade for MFSL based on today's intraday structure",
            height=100,
        )

    with col_config:
        symbol = st.text_input("Symbol (optional)", placeholder="MFSL").strip().upper()
        run_full = st.checkbox("Run full workflow (includes execution)", value=True)

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
                "market_data": None,
                "market_data_raw": None,
                "rag_context": None,
                "trade_plan": None,
                "planner_raw": None,
                "risk_approved": None,
                "risk_result": None,
                "execution_result": None,
                "journal_id": None,
                "error": None,
            }

            # Fetch data
            result = fetch_data_node(state)
            state.update(result)
            progress.progress(20)

            if state.get("error") and not state.get("market_data_raw"):
                st.error(f"Data fetch failed: {state['error']}")
                st.stop()

            market_meta = state.get("market_data") or {}
            if isinstance(market_meta, dict) and market_meta.get("error"):
                st.warning("Market data unavailable — planner will use RAG context only.")

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
                st.markdown("### 📋 Trade Plan")
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


# ──────────────────────────────────────────────
# PAGE: Straddle
# ──────────────────────────────────────────────
elif page == "Straddle":
    from trading.models import StraddlePosition

    st.header("Short Straddle Manager")
    st.caption("Register → Monitor → Analyze → Execute  |  Powered by @OptionsStrategist + @RiskGuard")

    tab_monitor, tab_analyze, tab_register, tab_history = st.tabs([
        "📊 Monitor", "🤖 Analyze & Act", "➕ Register Position", "📜 History"
    ])

    # ────────────────────────────────
    # TAB: Monitor
    # ────────────────────────────────
    with tab_monitor:
        active_positions = list(StraddlePosition.objects.filter(
            status__in=[StraddlePosition.Status.ACTIVE, StraddlePosition.Status.HEDGED,
                        StraddlePosition.Status.PARTIAL]
        ).order_by("-opened_at"))

        if not active_positions:
            st.info("No active straddle positions. Register one in the ➕ tab.")
        else:
            pos_options = {
                f"#{p.id} | {p.underlying} {p.strike} [{p.expiry}] {p.status}": p
                for p in active_positions
            }
            selected_label = st.selectbox("Select Position", list(pos_options.keys()))
            pos = pos_options[selected_label]

            if st.button("🔄 Refresh Live Data", use_container_width=True):
                st.rerun()

            with st.spinner("Fetching live market data..."):
                try:
                    from trading.options.data_service import OptionsDataService
                    from trading.options.straddle.analyzer import analyze_straddle

                    svc = OptionsDataService()
                    snap = svc.fetch_straddle_snapshot(
                        ce_symbol=pos.ce_symbol, ce_token=pos.ce_token,
                        pe_symbol=pos.pe_symbol, pe_token=pos.pe_token,
                        date_str=date.today().isoformat(),
                    )

                    nifty = snap.get("nifty", {})
                    vix   = snap.get("vix",   {})
                    ce    = snap.get("ce",    {})
                    pe    = snap.get("pe",    {})

                    analysis = analyze_straddle(
                        underlying=pos.underlying, strike=pos.strike,
                        expiry=pos.expiry.isoformat(), lot_size=pos.lot_size, lots=pos.lots,
                        ce_sell_price=pos.ce_sell_price, pe_sell_price=pos.pe_sell_price,
                        ce_ltp=ce.get("ltp", 0), pe_ltp=pe.get("ltp", 0),
                        nifty_spot=nifty.get("ltp", 0), nifty_prev_close=nifty.get("prev_close", 0),
                        vix_current=vix.get("ltp", 0), vix_prev_close=vix.get("prev_close", 0),
                        candles=snap.get("candles", []),
                    )

                    # ── Top alerts ──
                    if analysis.is_underwater:
                        st.error("🚨 POSITION UNDERWATER — Combined premium > sold. Hard stop triggered.")
                    if analysis.expiry_tomorrow:
                        st.warning("⚠️ EXPIRY TOMORROW — Gamma is extreme. Prefer closing over holding.")

                    # ── NIFTY + VIX row ──
                    st.subheader("Live Market")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    nifty_chg = nifty.get("ltp", 0) - nifty.get("prev_close", 0)
                    m1.metric("NIFTY Spot",  f"₹{nifty.get('ltp', 0):,.2f}",
                              f"{nifty_chg:+.0f} pts")
                    m2.metric("Day High",    f"₹{nifty.get('high', 0):,.2f}")
                    m3.metric("Day Low",     f"₹{nifty.get('low', 0):,.2f}")
                    vix_chg = vix.get("ltp", 0) - vix.get("prev_close", 0)
                    m4.metric("India VIX",   f"{vix.get('ltp', 0):.2f}",
                              f"{vix_chg:+.2f}  ({analysis.vix_phase})")
                    m5.metric("Market Phase", analysis.market_phase)

                    # ── Option legs ──
                    st.subheader("Option Legs")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"**{pos.ce_symbol}** (short call)")
                        ce_pnl = pos.ce_sell_price - ce.get("ltp", 0)
                        st.metric("CE LTP",  f"₹{ce.get('ltp', 0):.2f}", f"Sold @ {pos.ce_sell_price:.2f}")
                        st.metric("CE P&L",  f"{ce_pnl:+.2f} pts",
                                  delta_color="normal" if ce_pnl >= 0 else "inverse")
                    with c2:
                        st.markdown(f"**{pos.pe_symbol}** (short put)")
                        pe_pnl = pos.pe_sell_price - pe.get("ltp", 0)
                        st.metric("PE LTP",  f"₹{pe.get('ltp', 0):.2f}", f"Sold @ {pos.pe_sell_price:.2f}")
                        st.metric("PE P&L",  f"{pe_pnl:+.2f} pts",
                                  delta_color="normal" if pe_pnl >= 0 else "inverse")
                    with c3:
                        st.markdown("**Combined**")
                        st.metric("Net P&L",
                                  f"₹{analysis.net_pnl_inr:+,.0f}",
                                  f"{analysis.net_pnl_pts:+.2f} pts ({analysis.premium_decayed_pct:.1f}% decayed)")
                        st.metric("Net Delta", f"{analysis.net_delta:+.2f}",
                                  analysis.delta_bias)

                    # ── Moneyness + DTE ──
                    st.subheader("Position Health")
                    h1, h2, h3, h4 = st.columns(4)
                    h1.metric("Tested Leg",   analysis.nearest_itm_leg)
                    h2.metric("PE ITM by",    f"{analysis.pe_itm_by:.0f} pts")
                    h3.metric("Days to Expiry", analysis.days_to_expiry)
                    h4.metric("Stop Triggered", "YES 🚨" if analysis.stop_triggered else "No ✅")

                    # ── Expiry scenarios ──
                    with st.expander("📊 Expiry Scenarios"):
                        rows = [
                            {"Scenario": s.label, "NIFTY": f"{s.nifty_level:,.0f}",
                             "CE expiry": s.ce_expiry_value, "PE expiry": s.pe_expiry_value,
                             "P&L (INR)": f"{s.net_pnl_inr:+,.0f}"}
                            for s in analysis.scenarios
                        ]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Failed to fetch live data: {e}")

    # ────────────────────────────────
    # TAB: Analyze & Act
    # ────────────────────────────────
    with tab_analyze:
        active_pos = list(StraddlePosition.objects.filter(
            status__in=[StraddlePosition.Status.ACTIVE, StraddlePosition.Status.HEDGED,
                        StraddlePosition.Status.PARTIAL]
        ).order_by("-opened_at"))

        if not active_pos:
            st.info("No active positions. Register one first.")
        else:
            pos_opts = {f"#{p.id} | {p.underlying} {p.strike} [{p.expiry}]": p for p in active_pos}
            sel_label = st.selectbox("Select Position", list(pos_opts.keys()), key="analyze_pos_select")
            pos = pos_opts[sel_label]

            st.info(f"**{pos}**  |  Expiry: {pos.expiry}  |  Lot size: {pos.lot_size} × {pos.lots} lot(s)")

            if st.button("🤖 Run Full Analysis + LLM Recommendation", type="primary", use_container_width=True):
                progress = st.progress(0)
                status   = st.empty()

                from trading.options.straddle.graph import (
                    fetch_market_data_node, analyze_position_node,
                    generate_action_node, validate_action_node,
                )

                state = {
                    "position_id":   pos.id,
                    "underlying":    pos.underlying,
                    "strike":        pos.strike,
                    "expiry":        pos.expiry.isoformat(),
                    "lot_size":      pos.lot_size,
                    "lots":          pos.lots,
                    "ce_symbol":     pos.ce_symbol,
                    "ce_token":      pos.ce_token,
                    "pe_symbol":     pos.pe_symbol,
                    "pe_token":      pos.pe_token,
                    "ce_sell_price": pos.ce_sell_price,
                    "pe_sell_price": pos.pe_sell_price,
                    "nifty_candles": None, "market_snapshot": None, "analysis": None,
                    "recommended_action": None, "planner_raw": None,
                    "action_approved": None, "validation_result": None,
                    "execution_result": None, "journal_id": None, "error": None,
                }

                status.info("📡 Fetching live market data from Angel One...")
                state.update(fetch_market_data_node(state))
                progress.progress(25)

                if state.get("error"):
                    st.error(state["error"])
                    st.stop()

                status.info("🔢 Analyzing position (P&L, delta, scenarios)...")
                state.update(analyze_position_node(state))
                progress.progress(50)

                analysis = state.get("analysis", {})
                if analysis:
                    st.code(analysis.get("summary_text", ""), language=None)

                status.info("🤖 Running @OptionsStrategist (Claude)...")
                state.update(generate_action_node(state))
                progress.progress(75)

                status.info("🛡 Validating with @RiskGuard...")
                state.update(validate_action_node(state))
                progress.progress(90)

                action     = state.get("recommended_action", {}) or {}
                validation = state.get("validation_result", {}) or {}

                st.subheader("LLM Recommendation")
                if action:
                    a_color = {
                        "HOLD": "blue", "CLOSE_BOTH": "red", "CLOSE_CE": "orange",
                        "CLOSE_PE": "orange", "HEDGE_FUTURES": "violet", "ROLL": "green",
                    }.get(action.get("action", "HOLD"), "blue")

                    ac1, ac2, ac3 = st.columns(3)
                    ac1.metric("Action",    action.get("action", "?"))
                    ac2.metric("Urgency",   action.get("urgency", "?"))
                    ac3.metric("Confidence", f"{action.get('confidence', 0):.0%}")

                    ac4, ac5 = st.columns(2)
                    ac4.metric("CE Leg", action.get("ce_action", "?"))
                    ac5.metric("PE Leg", action.get("pe_action", "?"))

                    if action.get("pe_stop_loss"):
                        st.info(f"PE Stop: Close if PE > ₹{action['pe_stop_loss']:.2f}")
                    if action.get("pe_target"):
                        st.info(f"PE Target: Close if PE < ₹{action['pe_target']:.2f}")
                    if action.get("hedge_lots", 0) > 0:
                        st.info(f"Hedge: {action['hedge_side']} {action['hedge_lots']}L NIFTY Futures")

                    st.markdown(f"**Reasoning:** {action.get('reasoning', '')}")
                    st.caption(f"Key risk: {action.get('key_risk', '')}")

                approved = validation.get("approved", False)
                if approved:
                    st.success(f"🛡 @RiskGuard: **APPROVED** — {validation.get('reason', '')}")
                else:
                    st.error(f"🛡 @RiskGuard: **REJECTED** — {validation.get('reason', '')}")
                if validation.get("override_action"):
                    st.warning(f"Action overridden to: {validation['override_action']}")

                progress.progress(100)
                status.success("Analysis complete.")

                # Execute button appears only after analysis
                if approved and action.get("action") not in ("HOLD",):
                    st.markdown("---")
                    if st.button(
                        f"⚡ Execute: {action.get('action')} ({TRADING_MODE.upper()} mode)",
                        type="primary", use_container_width=True
                    ):
                        from trading.options.straddle.graph import (
                            execute_action_node, journal_action_node
                        )
                        state.update(execute_action_node(state))
                        state.update(journal_action_node(state))
                        exec_r = state.get("execution_result", {})
                        if exec_r.get("success"):
                            st.success(
                                f"✅ [{exec_r.get('mode','paper').upper()}] "
                                f"{' | '.join(exec_r.get('actions_taken', []))}"
                            )
                        else:
                            st.error(f"Execution failed: {exec_r.get('message', '')}")
                        st.rerun()

    # ────────────────────────────────
    # TAB: Register
    # ────────────────────────────────
    with tab_register:
        st.subheader("Register a New Straddle Position")
        st.caption("Fill in the details of the straddle you have already sold.")

        with st.form("register_straddle"):
            r1, r2, r3 = st.columns(3)
            underlying = r1.text_input("Underlying", value="NIFTY").upper()
            strike     = r2.number_input("Strike", value=24200, step=50, min_value=10000)
            expiry     = r3.date_input("Expiry Date", value=date.today())

            r4, r5 = st.columns(2)
            lots     = r4.number_input("Lots", value=1, min_value=1, max_value=100)
            lot_size = r5.number_input("Lot Size", value=65, min_value=1)

            st.markdown("**CE Leg (short call)**")
            c1, c2, c3 = st.columns(3)
            ce_symbol    = c1.text_input("CE Symbol", placeholder="NIFTY10MAR2624200CE")
            ce_token     = c2.text_input("CE Token",  placeholder="45482")
            ce_sell_price = c3.number_input("CE Sell Price (pts)", value=0.0, min_value=0.0, step=0.05)

            st.markdown("**PE Leg (short put)**")
            p1, p2, p3 = st.columns(3)
            pe_symbol    = p1.text_input("PE Symbol", placeholder="NIFTY10MAR2624200PE")
            pe_token     = p2.text_input("PE Token",  placeholder="45483")
            pe_sell_price = p3.number_input("PE Sell Price (pts)", value=0.0, min_value=0.0, step=0.05)

            combined = ce_sell_price + pe_sell_price
            if combined > 0:
                st.info(f"Combined premium: **{combined:.2f} pts = ₹{combined * lot_size * lots:,.0f}**")

            submitted = st.form_submit_button("Register Position", type="primary")
            if submitted:
                if not ce_symbol or not pe_symbol or not ce_token or not pe_token:
                    st.error("CE and PE symbol + token are required.")
                elif combined == 0:
                    st.error("Enter the CE and PE sell prices.")
                else:
                    pos = StraddlePosition.objects.create(
                        underlying=underlying, strike=strike, expiry=expiry,
                        lot_size=lot_size, lots=lots,
                        ce_symbol=ce_symbol, ce_token=ce_token, ce_sell_price=ce_sell_price,
                        pe_symbol=pe_symbol, pe_token=pe_token, pe_sell_price=pe_sell_price,
                        trade_date=date.today(),
                    )
                    st.success(
                        f"✅ Registered: **#{pos.id}** | {underlying} {strike} [{expiry}]  "
                        f"| Combined: {combined:.2f} pts = ₹{pos.total_premium_sold:,.0f}  "
                        f"→ Go to 📊 Monitor tab"
                    )
                    st.rerun()

        st.markdown("---")
        st.subheader("Token Lookup Helper")
        st.caption("Find Angel One NFO tokens for a NIFTY option.")

        tl1, tl2, tl3, tl4 = st.columns(4)
        tl_und    = tl1.text_input("Underlying", value="NIFTY", key="tl_und")
        tl_strike = tl2.number_input("Strike", value=24200, step=50, key="tl_strike")
        tl_expiry = tl3.text_input("Expiry (e.g. 10MAR26)", placeholder="10MAR26", key="tl_expiry")
        tl_type   = tl4.selectbox("Type", ["CE", "PE", "Both"], key="tl_type")

        if st.button("Find Tokens", key="token_lookup"):
            from trading.options.data_service import find_option_token
            types = ["CE", "PE"] if tl_type == "Both" else [tl_type]
            for ot in types:
                result = find_option_token(tl_und, tl_strike, tl_expiry, ot)
                if result:
                    sym, tok = result
                    st.success(f"**{ot}** → Symbol: `{sym}` | Token: `{tok}`")
                else:
                    st.warning(f"Token not found for {tl_und} {tl_strike} {tl_expiry} {ot}")

    # ────────────────────────────────
    # TAB: History
    # ────────────────────────────────
    with tab_history:
        st.subheader("All Straddle Positions")

        all_positions = list(StraddlePosition.objects.order_by("-opened_at")[:50])
        if not all_positions:
            st.info("No straddle positions recorded yet.")
        else:
            rows = []
            for p in all_positions:
                rows.append({
                    "ID":         p.id,
                    "Underlying": p.underlying,
                    "Strike":     p.strike,
                    "Expiry":     str(p.expiry),
                    "Lots":       p.lots,
                    "CE Sold @":  p.ce_sell_price,
                    "PE Sold @":  p.pe_sell_price,
                    "Combined":   f"{p.combined_sell_pts:.2f}",
                    "Status":     p.status,
                    "Action":     p.action_taken,
                    "P&L (INR)":  f"{p.current_pnl_inr:+,.0f}",
                    "Opened":     p.opened_at.strftime("%Y-%m-%d %H:%M"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Management log for selected position
            st.subheader("Management Log")
            pos_ids = {f"#{p.id} {p.underlying} {p.strike} [{p.status}]": p for p in all_positions}
            sel = st.selectbox("Select position for log", list(pos_ids.keys()), key="hist_pos")
            sel_pos = pos_ids[sel]

            if sel_pos.management_log:
                log_rows = [{
                    "Time":    e.get("time", "?"),
                    "Action":  e.get("action", "?"),
                    "Urgency": e.get("urgency", "?"),
                    "NIFTY":   f"{e.get('nifty', 0):,.0f}",
                    "P&L":     f"{e.get('pnl_inr', 0):+,.0f}",
                    "Executed": "✅" if e.get("executed") else "—",
                    "Note":    e.get("note", "")[:60],
                } for e in sel_pos.management_log]
                st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No management actions recorded yet for this position.")


# ──────────────────────────────────────────────
# PAGE: Journal
# ──────────────────────────────────────────────
elif page == "Journal":
    st.header("Trade Journal")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sym_filter = st.text_input("Filter by symbol", "").strip().upper()
    with col_f2:
        status_filter = st.selectbox(
            "Status",
            ["ALL"] + [s[0] for s in TradeJournal.Status.choices],
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
                "Symbol": e.symbol,
                "Side": e.side,
                "Qty": e.quantity,
                "Entry": e.entry_price,
                "SL": e.stop_loss,
                "Target": e.target,
                "Status": e.status,
                "Risk": "✅" if e.risk_approved else "❌",
                "Confidence": f"{e.confidence:.0%}",
                "P&L": f"₹{e.pnl:+,.0f}" if e.pnl is not None else "—",
                "R:R": f"{e.risk_reward_ratio:.1f}",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("Trade Reasoning Details"):
            for e in entries[:10]:
                st.markdown(
                    f"**#{e.id} {e.side} {e.symbol}** ({e.status}): "
                    f"{e.reasoning}"
                )
                if e.risk_reason:
                    st.caption(f"Risk: {e.risk_reason}")
                st.markdown("---")


# ──────────────────────────────────────────────
# PAGE: Strategies
# ──────────────────────────────────────────────
elif page == "Strategies":
    st.header("Strategy Documents")
    st.caption("These are injected as RAG context into the planner agent.")

    strategies = StrategyDoc.objects.all()

    if not strategies.exists():
        st.info("No strategies yet. Seed defaults or add one below.")
    else:
        for s in strategies:
            active_badge = "🟢" if s.is_active else "⚪"
            with st.expander(f"{active_badge} [{s.category}] {s.title}"):
                st.markdown(s.content)
                col_a, col_b = st.columns(2)
                if col_a.button(
                    "Deactivate" if s.is_active else "Activate",
                    key=f"toggle_{s.id}",
                ):
                    s.is_active = not s.is_active
                    s.save()
                    st.rerun()
                if col_b.button("Delete", key=f"del_{s.id}"):
                    s.delete()
                    st.rerun()

    st.markdown("---")
    st.subheader("Add Strategy")
    with st.form("add_strategy"):
        title = st.text_input("Title")
        category = st.selectbox("Category", [c[0] for c in StrategyDoc.Category.choices])
        content = st.text_area("Content", height=120)
        submitted = st.form_submit_button("Add Strategy")

        if submitted and title and content:
            StrategyDoc.objects.create(title=title, category=category, content=content)
            st.success(f"Added: [{category}] {title}")
            st.rerun()

    st.markdown("---")
    if st.button("Seed Default Strategies"):
        from trading.management.commands.run_trading_agent import Command
        cmd = Command()
        cmd.stdout = type("FakeOut", (), {"write": lambda self, x: None})()
        cmd._seed_strategies()
        st.success("Default strategies seeded.")
        st.rerun()


# ──────────────────────────────────────────────
# PAGE: Portfolio
# ──────────────────────────────────────────────
elif page == "Portfolio":
    st.header("Portfolio Management")

    try:
        snap = PortfolioSnapshot.objects.latest()
        st.subheader(f"Current Snapshot — {snap.snapshot_date}")

        p_cols = st.columns(3)
        p_cols[0].metric("Total Capital", f"₹{snap.capital:,.0f}")
        p_cols[1].metric("Invested", f"₹{snap.invested:,.0f}")
        p_cols[2].metric("Available Cash", f"₹{snap.available_cash:,.0f}")

        p_cols2 = st.columns(4)
        p_cols2[0].metric("Today's P&L", f"₹{snap.daily_pnl:+,.0f}")
        p_cols2[1].metric("Total P&L", f"₹{snap.total_pnl:+,.0f}")
        p_cols2[2].metric("Today's Losses", f"₹{snap.daily_loss:,.0f}")
        p_cols2[3].metric("Open Positions", snap.open_positions)

    except Exception:
        st.warning("No portfolio snapshot exists.")

    st.markdown("---")
    st.subheader("Initialize Portfolio")
    with st.form("init_portfolio"):
        capital = st.number_input("Capital (INR)", min_value=10000, value=500000, step=50000)
        if st.form_submit_button("Create Snapshot"):
            PortfolioSnapshot.objects.create(
                capital=capital,
                invested=0,
                available_cash=capital,
                daily_pnl=0,
                total_pnl=0,
                daily_loss=0,
                open_positions=0,
                open_positions_count=0,
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
            "Invested": f"₹{s.invested:,.0f}",
            "Available": f"₹{s.available_cash:,.0f}",
            "Daily P&L": f"₹{s.daily_pnl:+,.0f}",
            "Total P&L": f"₹{s.total_pnl:+,.0f}",
            "Losses": f"₹{s.daily_loss:,.0f}",
            "Positions": s.open_positions,
            "Updated": s.last_updated.strftime("%H:%M") if s.last_updated else "—",
        } for s in snapshots]
        st.dataframe(pd.DataFrame(snap_rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# PAGE: Backtest
# ──────────────────────────────────────────────
elif page == "Backtest":
    st.header("Backtest Engine")
    st.caption("Replay historical candles through planner + risk engine. If planner loses in backtest, it will lose live.")

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        bt_symbol = st.text_input("Symbol", value="MFSL", max_chars=20).upper()
    with col_b2:
        bt_capital = st.number_input("Initial Capital (INR)", value=500000, step=50000, min_value=10000)
    with col_b3:
        bt_source = st.radio("Data Source", ["Fetch from Broker", "Upload CSV", "Sample Data"])

    uploaded_file = None
    if bt_source == "Fetch from Broker":
        from datetime import date as _date, timedelta as _td
        date_cols = st.columns(2)
        with date_cols[0]:
            bt_from_date = st.date_input(
                "From Date",
                value=_date.today() - _td(days=10),
                max_value=_date.today(),
            )
        with date_cols[1]:
            bt_to_date = st.date_input(
                "To Date",
                value=_date.today() - _td(days=1),
                max_value=_date.today(),
            )
        st.info(f"Will fetch daily OHLCV candles for **{bt_symbol}** from Angel One broker ({bt_from_date} → {bt_to_date}). Weekends and holidays are auto-skipped.")
    elif bt_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload OHLCV CSV (columns: date, open, high, low, close, volume)",
            type=["csv"],
        )
        st.info("CSV must have columns: date, open, high, low, close, volume")
    else:
        st.info("Will generate 5 synthetic candles for quick testing.")

    if st.button("Run Backtest", type="primary", use_container_width=True):
        from trading.services.backtester import run_backtest

        candles = []

        if bt_source == "Fetch from Broker":
            with st.spinner(f"Fetching historical data for {bt_symbol} from Angel One..."):
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
                        st.warning(f"No data returned for {bt_symbol}. Check if the symbol is correct and market was open in the selected date range.")
                        candles = None
                    else:
                        st.success(f"Fetched {len(candles)} trading days of data for {bt_symbol}")
                except Exception as e:
                    st.error(f"Broker fetch failed: {e}")
                    candles = None
        elif bt_source == "Upload CSV" and uploaded_file is not None:
            import csv, io
            content = uploaded_file.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                candles.append({
                    "date": row["date"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
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
            st.warning("Please upload a CSV file or select a data source.")
            candles = None

        if candles:
            with st.spinner(f"Running backtest on {len(candles)} candles..."):
                try:
                    result = run_backtest(
                        symbol=bt_symbol,
                        candles=candles,
                        initial_capital=bt_capital,
                    )

                    s = result["summary"]

                    # Summary metrics
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

                    # Trade log table
                    st.subheader("Trade Log")
                    trade_rows = []
                    for t in result["trades"]:
                        if t["action"] == "TRADED":
                            sim = t["simulation"]
                            trade_rows.append({
                                "Date": t["candle"],
                                "Side": sim["side"],
                                "Entry": sim["entry"],
                                "SL": sim["sl"],
                                "Target": sim["target"],
                                "Qty": sim["qty"],
                                "Exit": sim["exit_price"],
                                "Outcome": sim["outcome"],
                                "P&L": t["pnl"],
                                "Capital": t["capital_after"],
                            })
                        else:
                            trade_rows.append({
                                "Date": t["candle"],
                                "Side": "—",
                                "Entry": 0, "SL": 0, "Target": 0, "Qty": 0, "Exit": 0,
                                "Outcome": t["action"],
                                "P&L": 0,
                                "Capital": 0,
                            })

                    if trade_rows:
                        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

                    # Equity curve
                    equity_data = [bt_capital]
                    for t in result["trades"]:
                        if t["action"] == "TRADED":
                            equity_data.append(t["capital_after"])
                    if len(equity_data) > 1:
                        st.subheader("Equity Curve")
                        st.line_chart(equity_data)

                except Exception as e:
                    st.error(f"Backtest failed: {e}")


# ──────────────────────────────────────────────
# PAGE: Audit Log
# ──────────────────────────────────────────────
elif page == "Audit Log":
    st.header("Audit Log")
    st.caption("Every LLM prompt, model response, and risk decision — stored for accountability.")

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        event_filter = st.selectbox(
            "Event Type",
            ["ALL"] + [e[0] for e in AuditLog.EventType.choices],
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
        # Summary
        st.markdown(f"**{len(audit_entries)} entries** (showing last {audit_days} days)")

        for entry in audit_entries:
            icon = {
                "PLANNER_REQ": "📤",
                "PLANNER_RES": "📥",
                "PLANNER_ERR": "💥",
                "RISK_APPROVE": "✅",
                "RISK_REJECT": "❌",
                "EXECUTION": "⚡",
                "RECONCILE": "🔄",
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
                if entry.trade_journal_id:
                    st.caption(f"Linked to Journal #{entry.trade_journal_id}")


# ──────────────────────────────────────────────
# PAGE: Settings
# ──────────────────────────────────────────────
elif page == "Settings":
    st.header("System Settings")

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

    st.markdown("---")
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
        test_capital = r_cols4[0].number_input("Portfolio Capital", value=500000, step=50000)
        test_daily_loss = r_cols4[1].number_input("Current Daily Loss (INR)", value=0, step=1000, min_value=0)
        test_open_pos = r_cols4[2].number_input("Open Positions", value=0, step=1, min_value=0, max_value=10)

        if st.form_submit_button("Validate"):
            plan = {
                "symbol": test_symbol,
                "side": test_side,
                "entry_price": test_entry,
                "stop_loss": test_sl,
                "target": test_target,
                "quantity": test_qty,
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
