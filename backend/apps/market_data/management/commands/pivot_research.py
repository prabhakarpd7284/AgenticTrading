"""Pivot Boss research + backtest on NIFTY 50 (spec: docs/strategies/PIVOT_BOSS_SPEC.md).

    .venv/bin/python manage.py pivot_research [--symbols A,B] [--out-dir research_output]
                                              [--slippage-pct 0.02] [--skip-backtest]

Reads only ``market_data.Candle`` (1d NSE bhavcopy for pivot inputs, 1m Angel
bars for intraday paths). No broker calls. Processes one symbol at a time to
keep memory flat. Writes CSVs + ``summary.md`` under ``--out-dir``.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path

from django.core.management.base import BaseCommand
from django.db import connection

from apps.market_data.constants import NIFTY_50_SYMBOLS
from plugins.strategy_pivot_boss import research as pbr
from plugins.strategy_pivot_boss.backtest import BacktestConfig, CostModel, simulate, summarize
from plugins.strategy_pivot_boss.bars import Bar, ist_day, ist_minute
from plugins.strategy_pivot_boss.levels import camarilla, cpr, floor_pivots, money_zone
from plugins.strategy_pivot_boss.relationships import DEFAULT_UNCHANGED_TOL_PCT
from plugins.strategy_pivot_boss.setups import SETUPS, DayContext, SetupParams
from plugins.strategy_pivot_boss.triggers import any_trigger_series, resample

IS_END = date(2026, 8, 31)
LAST_MIN = 15 * 60 + 14  # last tradable 1m bar (15:15–15:27 missing since 2026-08-03)
EPOCH = date(1970, 1, 1)


def _d(day_no: int) -> date:
    return EPOCH + timedelta(days=day_no)


class Command(BaseCommand):
    help = "Re-measure Pivot Boss edges on NIFTY 50 and backtest the main setups."

    def add_arguments(self, parser):
        parser.add_argument("--symbols", default="", help="comma list (default NIFTY 50)")
        parser.add_argument("--out-dir", default="research_output")
        parser.add_argument("--slippage-pct", type=float, default=0.02)
        parser.add_argument("--narrow-pctile", type=float, default=20.0)
        parser.add_argument("--wide-pctile", type=float, default=80.0)
        parser.add_argument("--unchanged-tol-pct", type=float, default=DEFAULT_UNCHANGED_TOL_PCT)
        parser.add_argument("--skip-backtest", action="store_true")

    # ── data ────────────────────────────────────────────────────────────────

    def _symbol_ids(self, names):
        with connection.cursor() as cur:
            cur.execute("SELECT tradingsymbol, id FROM market_data_symbol "
                        "WHERE exchange='NSE' AND tradingsymbol = ANY(%s)", [names])
            return dict(cur.fetchall())

    def _rows(self, sid, interval):
        with connection.cursor() as cur:
            cur.execute(
                "SELECT extract(epoch from t)::bigint, o::float8, h::float8, l::float8, "
                "c::float8, v::float8 FROM market_data_candle "
                "WHERE symbol_id=%s AND interval=%s ORDER BY t", [sid, interval])
            return [Bar(*r) for r in cur.fetchall()]

    # ── main ────────────────────────────────────────────────────────────────

    def handle(self, *args, **opts):
        names = [s.strip() for s in opts["symbols"].split(",") if s.strip()] or NIFTY_50_SYMBOLS
        ids = self._symbol_ids(names)
        missing = sorted(set(names) - set(ids))
        if missing:
            self.stderr.write(f"missing symbols: {missing}")
        out_dir = Path(opts["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        tol = opts["unchanged_tol_pct"]

        # Pass 0 — NSE calibration of CPR width from daily data, IN-SAMPLE ONLY
        is_widths, all_dates = [], set()
        for name in names:
            if name not in ids:
                continue
            daily = self._rows(ids[name], "1d")
            for p, d in zip(daily, daily[1:], strict=False):
                dn = ist_day(d.t)
                all_dates.add(dn)
                if _d(dn) <= IS_END:
                    is_widths.append(cpr(p.h, p.l, p.c).width_pct)
        narrow = pbr.percentile(is_widths, opts["narrow_pctile"])
        wide = pbr.percentile(is_widths, opts["wide_pctile"])
        quint = [pbr.percentile(is_widths, q) for q in (20, 40, 60, 80)]
        params = SetupParams(narrow_cpr_pct=narrow, wide_cpr_pct=wide)
        cfg = BacktestConfig(slippage_pct=opts["slippage_pct"], costs=CostModel())
        self.stdout.write(f"IS CPR width: narrow<{narrow:.4f}% wide>{wide:.4f}% "
                          f"quintile edges {[round(x, 4) for x in quint]} (n={len(is_widths)})")

        rows: list[dict] = []
        trades = []
        for name in names:
            if name not in ids:
                continue
            r, t = self._process_symbol(name, ids[name], params, cfg, tol,
                                        not opts["skip_backtest"])
            rows.extend(r)
            trades.extend(t)
            self.stdout.write(f"{name}: {len(r)} days, {len(t)} trades")

        # ── outputs ────────────────────────────────────────────────────────
        self._write_csv(out_dir / "pivot_research_days.csv", rows)
        self._write_csv(out_dir / "pivot_backtest_trades.csv",
                        [dict(t.__dict__.items()) for t in trades])
        sessions_is = sum(1 for d in all_dates if _d(d) <= IS_END)
        sessions_oos = sum(1 for d in all_dates if _d(d) > IS_END)
        summary = self._summary_md(rows, trades, narrow, wide, quint, is_widths,
                                   sessions_is, sessions_oos, opts)
        (out_dir / "summary.md").write_text(summary)
        self.stdout.write(summary)

    # ── per symbol ──────────────────────────────────────────────────────────

    def _process_symbol(self, name, sid, params, cfg, tol, do_bt):
        daily = self._rows(sid, "1d")
        mins = list(self._rows(sid, "1m"))
        by_day: dict[int, list[Bar]] = {}
        for b in mins:
            by_day.setdefault(ist_day(b.t), []).append(b)
        del mins
        trade_day = {d: [b for b in bs if ist_minute(b.t) <= LAST_MIN] for d, bs in by_day.items()}

        # continuous 15m series for triggers / ATR, per-day slices
        days_sorted = sorted(trade_day)
        bars15_all, day15_start, day15_len = [], {}, {}
        bars5_by_day = {}
        for d in days_sorted:
            b15 = resample(trade_day[d], 15)
            day15_start[d], day15_len[d] = len(bars15_all), len(b15)
            bars15_all.extend(b15)
            bars5_by_day[d] = resample(trade_day[d], 5)
        trig15 = any_trigger_series(bars15_all)
        atr15 = _atr(bars15_all, 14)

        # Money Zone per day (all 1m bars of the session)
        mz = {d: money_zone(bs) for d, bs in by_day.items() if bs}
        del by_day

        # daily ATR(14) known before each session
        tr = [daily[0].h - daily[0].l] + [
            max(b.h, p.c) - min(b.l, p.c) for p, b in zip(daily, daily[1:], strict=False)]
        rows, trades = [], []
        for i in range(1, len(daily)):
            p, dbar = daily[i - 1], daily[i]
            dn = ist_day(dbar.t)
            bars = trade_day.get(dn)
            if not bars or ist_minute(bars[0].t) != 9 * 60 + 15:
                continue
            day = _d(dn)
            atr_d = sum(tr[i - 14:i]) / 14 if i >= 15 else None
            f = floor_pivots(p.h, p.l, p.c)
            cp = cpr(p.h, p.l, p.c)
            cam = camarilla(p.h, p.l, p.c)
            pp_ = daily[i - 2] if i >= 2 else None
            cp_prev = cpr(pp_.h, pp_.l, pp_.c) if pp_ else None
            cam_prev = camarilla(pp_.h, pp_.l, pp_.c) if pp_ else None
            dn_prev = ist_day(p.t)
            mz_prev = mz.get(dn_prev)
            mz_prev2 = mz.get(ist_day(pp_.t)) if pp_ else None

            o, h, l, c = dbar.o, dbar.h, dbar.l, dbar.c  # noqa: E741 — official bhavcopy
            row = {
                "symbol": name, "date": day.isoformat(),
                "sample": "IS" if day <= IS_END else "OOS", "month": day.month,
                "o": o, "h": h, "l": l, "c": c, "prev_h": p.h, "prev_l": p.l, "prev_c": p.c,
                "atr14": atr_d, "range_atr": (h - l) / atr_d if atr_d else None,
                "cpr_width_pct": cp.width_pct,
                "trend_day": pbr.trend_day(o, h, l, c),
                "trend_day2": (pbr.close_at_extreme(h, l, c) if atr_d and (h - l) >= atr_d else 0)
                if atr_d else None,
                "ret_oc_pct": (c - o) / o * 100,
            }
            row.update(pbr.floor_row(o, h, l, c, f, cp))
            for lv in ("h3", "h4", "h5", "l3", "l4", "l5"):
                x = getattr(cam, lv)
                row[f"cam_{lv}_touch"] = h >= x if lv[0] == "h" else l <= x
            row.update(pbr.ib_stats(bars))
            row.update(pbr.camarilla_sequences(bars, bars5_by_day[dn], cam, bars[0].o))
            row.update(pbr.gap_fill(bars, p.c, cp))
            row["gap_atr"] = abs(row["gap_pct"]) * p.c / 100 / atr_d if atr_d else None
            row.update(pbr.day_relationships(cp, cp_prev, cam, cam_prev, mz_prev, mz_prev2, tol))
            row["open_loc"] = pbr.open_location(bars[0].o, p.h, p.l, mz_prev)
            if mz_prev:
                row["prev_vpoc"] = mz_prev.vpoc
            rows.append(row)

            if do_bt:
                s15 = day15_start[dn]
                n15 = day15_len[dn]
                atr_at = _atr_at_1m(bars, bars15_all[s15:s15 + n15], atr15[s15:s15 + n15])
                ctx = DayContext(
                    symbol=name, day=row["date"], bars1m=bars, bars5=bars5_by_day[dn],
                    bars15=bars15_all[s15:s15 + n15], trig15=trig15[s15:s15 + n15],
                    prev_h=p.h, prev_l=p.l, prev_c=p.c, floor=f, cpr=cp, cam=cam,
                    cpr_prev=cp_prev, prev_mz=mz_prev, atr_daily=atr_d, atr15_at=atr_at)
                for fn in SETUPS.values():
                    sig = fn(ctx, params)
                    if sig is not None:
                        res = simulate(name, row["date"], bars, sig, cfg)
                        if res is not None:
                            trades.append(res)

        # virgin VPOC: not touched next session → touched within 5 sessions?
        for j, r in enumerate(rows):
            v = r.get("prev_vpoc")  # VPOC of the prior session, tested from today on
            if v is None:
                continue
            r["vpoc_virgin"] = not (r["l"] <= v <= r["h"])
            if r["vpoc_virgin"]:
                nxt = rows[j + 1:j + 5]
                nxt = [x for x in nxt if x["symbol"] == r["symbol"]]
                r["vpoc_filled_5d"] = any(x["l"] <= v <= x["h"] for x in nxt) if len(nxt) == 4 else None
        return rows, trades

    # ── output helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _write_csv(path, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    def _summary_md(self, rows, trades, narrow, wide, quint, is_widths, s_is, s_oos, opts):
        out = ["# Pivot Boss — NSE research summary (auto-generated)", ""]
        n_sym = len({r["symbol"] for r in rows})
        out.append(f"Stock-days: {len(rows)} across {n_sym} symbols; IS sessions {s_is}, "
                   f"OOS sessions {s_oos}. `*` = n < {pbr.LOW_N} (low confidence).\n")

        out.append("## A. Appendix-A replication (floor pivots, bhavcopy H/L/C)\n")
        out.append(pbr.section_a(rows) + "\n")
        for samp in ("IS", "OOS"):
            out.append(f"### A ({samp} only)\n")
            out.append(pbr.section_a([r for r in rows if r["sample"] == samp]) + "\n")

        out.append("## B. CPR width vs day type\n")
        widths = [r["cpr_width_pct"] for r in rows]
        pcts = (5, 10, 20, 25, 40, 50, 60, 75, 80, 90, 95)
        out.append(pbr.md_table(["sample"] + [f"p{q}" for q in pcts] + ["n"], [
            ["all"] + [f"{pbr.percentile(widths, q):.3f}" for q in pcts] + [len(widths)],
            ["IS"] + [f"{pbr.percentile(is_widths, q):.3f}" for q in pcts] + [len(is_widths)],
        ]) + "\n")
        below = sum(1 for w in is_widths if w < 0.25) / len(is_widths) * 100
        above = sum(1 for w in is_widths if w > 0.75) / len(is_widths) * 100
        out.append(f"Book cut-offs on NSE (IS): {below:.1f}% of stock-days < 0.25%, "
                   f"{above:.1f}% > 0.75%. NSE-calibrated (IS p{opts['narrow_pctile']:.0f}/"
                   f"p{opts['wide_pctile']:.0f}): narrow < {narrow:.3f}%, wide > {wide:.3f}%.\n")
        base = [r for r in rows if r["range_atr"] is not None]
        edges = [-1] + quint + [1e9]
        labs = [f"Q1 (< {quint[0]:.3f}%)", f"Q2 ({quint[0]:.3f}–{quint[1]:.3f})",
                f"Q3 ({quint[1]:.3f}–{quint[2]:.3f})", f"Q4 ({quint[2]:.3f}–{quint[3]:.3f})",
                f"Q5 (≥ {quint[3]:.3f}%)"]
        for samp, sub in (("all", base), ("IS", [r for r in base if r["sample"] == "IS"]),
                          ("OOS", [r for r in base if r["sample"] == "OOS"])):
            out.append(f"### Width quintiles (IS edges), {samp}\n")
            out.append(pbr.width_buckets(sub, edges, labs) + "\n")
        out.append("### Book cut-offs, all days\n")
        out.append(pbr.width_buckets(base, [-1, 0.25, 0.75, 1e9],
                                   ["< 0.25 (book narrow)", "0.25–0.75", "> 0.75 (book wide)"]) + "\n")
        t1, n = pbr.pct(base, lambda r: r["trend_day"] != 0)
        t2, _ = pbr.pct(base, lambda r: r["trend_day2"] != 0)
        out.append(f"Unconditional trend-day rate: def1 {pbr.fmt_pct(t1, n)}, def2 {pbr.fmt_pct(t2, n)} (n={n}).\n")

        out.append("## C. Camarilla\n")
        tl = []
        for lv in ("h3", "l3", "h4", "l4", "h5", "l5"):
            v, n = pbr.pct(rows, lambda r, lv=lv: r[f"cam_{lv}_touch"])
            tl.append([lv.upper(), pbr.fmt_pct(v, n), n])
        out.append(pbr.md_table(["Level reached (daily H/L)", "% of days", "n"], tl) + "\n")
        inside = [r for r in rows if r.get("cam_open_inside")]
        out.append(f"Days opening strictly inside L3–H3: {len(inside)} of {len(rows)}.\n")
        sl = []
        for lv, lab in (("h3", "First touch H3 (from inside)"), ("l3", "First touch L3 (from inside)")):
            sub = [r for r in inside if r.get(f"seq_{lv}")]
            n = len(sub)
            row = [lab, n]
            for k in ("opposite", "break", "none", "both"):
                v, _ = pbr.pct(sub, lambda r, k=k, lv=lv: r[f"seq_{lv}"] == k)
                row.append(pbr.fmt_pct(v, n))
            sl.append(row)
        out.append(pbr.md_table(["Event", "n", "→ opposite band first", "→ H4/L4 first",
                               "neither by 15:14", "same 1m bar"], sl) + "\n")
        bl = []
        for key, lab in (("h4brk_1m", "1m close > H4"), ("h4brk_5m", "5m close > H4"),
                         ("l4brk_1m", "1m close < L4"), ("l4brk_5m", "5m close < L4")):
            sub = [r for r in rows if r.get(key)]
            n = len(sub)
            row = [lab, n]
            for k in ("ext", "back", "none", "both"):
                v, _ = pbr.pct(sub, lambda r, k=k, key=key: r[key] == k)
                row.append(pbr.fmt_pct(v, n))
            bl.append(row)
        out.append(pbr.md_table(["First close beyond (cross)", "n", "→ H5/L5 first",
                               "→ back to H3/L3 first", "neither by 15:14", "same 1m bar"], bl) + "\n")

        out.append("## D. Gaps / Magnet\n")
        out.append(self._gap_tables(rows))

        out.append("## E. Two-day relationships → that day's outcome\n")
        for key, lab in (("cpr_rel", "CPR (BC–TC)"), ("cam_rel", "Camarilla (L3–H3)"),
                         ("va_rel", "Value area (VAL–VAH, 1m volume profile)")):
            out.append(f"### {lab}\n")
            out.append(self._rel_table(rows, key) + "\n")
        out.append("### Open location vs prior range & value (Table 4.1)\n")
        out.append(self._rel_table(rows, "open_loc") + "\n")

        out.append("## F. Virgin VPOC\n")
        vv = [r for r in rows if "vpoc_virgin" in r]
        v, n = pbr.pct(vv, lambda r: r["vpoc_virgin"])
        filled = [r for r in vv if r.get("vpoc_filled_5d") is not None]
        v2, n2 = pbr.pct(filled, lambda r: r["vpoc_filled_5d"])
        out.append(f"Prior-session VPOC not touched today (virgin): {pbr.fmt_pct(v, n)} (n={n}). "
                   f"Of those, filled within the next 4 sessions (≈ a week incl. today): "
                   f"{pbr.fmt_pct(v2, n2)} (n={n2}; book ≈ 80%).\n")

        if trades:
            out.append("## Backtest (net of costs)\n")
            out.append(self._bt_tables(trades, s_is, s_oos, opts) + "\n")
        return "\n".join(out)

    def _gap_tables(self, rows):
        out = []
        gp = [r for r in rows if r.get("gap_atr") is not None]
        pct_edges = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]
        atr_edges = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1e9)]
        for title, sub in (("all months", gp),
                           ("earnings months (Apr 20–30, Jul, Oct)", [r for r in gp if r["month"] in (1, 4, 7, 10)]),
                           ("non-earnings months", [r for r in gp if r["month"] not in (1, 4, 7, 10)])):
            for unit, edges, fn in (("% of prior close", pct_edges, lambda r: abs(r["gap_pct"])),
                                    ("daily ATR14 units", atr_edges, lambda r: r["gap_atr"])):
                tl = []
                for lo, hi in edges:
                    b = [r for r in sub if lo <= fn(r) < hi]
                    bc = [r for r in b if r.get("open_vs_cpr") in ("above", "below")]
                    n, nc = len(b), len(bc)
                    a1, _ = pbr.pct(b, lambda r: r.get("pc_fill_1015"))
                    a2, _ = pbr.pct(b, lambda r: r.get("pc_fill_eod"))
                    c1, _ = pbr.pct(bc, lambda r: r["cpr_fill_1015"])
                    c2, _ = pbr.pct(bc, lambda r: r["cpr_fill_eod"])
                    lab = f"{lo:g}–{hi:g}" if hi < 1e8 else f"≥ {lo:g}"
                    tl.append([lab, n, pbr.fmt_pct(a1, n), pbr.fmt_pct(a2, n), nc,
                               pbr.fmt_pct(c1, nc), pbr.fmt_pct(c2, nc)])
                out.append(f"### Gap size in {unit} — {title}\n")
                out.append(pbr.md_table(["gap", "n", "prior close filled by 10:15", "… by 15:14",
                                       "n (open outside CPR)", "CPR reached by 10:15",
                                       "… by 15:14"], tl) + "\n")
        return "\n".join(out)

    @staticmethod
    def _rel_table(rows, key):
        cats = sorted({r[key] for r in rows if r.get(key)})
        tl = []
        total = sum(1 for r in rows if r.get(key))
        for cat in cats:
            b = [r for r in rows if r.get(key) == cat]
            n = len(b)
            up, _ = pbr.pct(b, lambda r: r["c"] > r["o"])
            t1, _ = pbr.pct(b, lambda r: r["trend_day"] != 0)
            tu, _ = pbr.pct(b, lambda r: r["trend_day"] > 0)
            td, _ = pbr.pct(b, lambda r: r["trend_day"] < 0)
            mr = sum(r["ret_oc_pct"] for r in b) / n
            ra = [r["range_atr"] for r in b if r["range_atr"] is not None]
            tl.append([cat, n, f"{n / total * 100:.1f}%", pbr.fmt_pct(up, n), f"{mr:+.3f}%",
                       pbr.fmt_pct(t1, n), pbr.fmt_pct(tu, n), pbr.fmt_pct(td, n),
                       f"{sum(ra) / len(ra):.2f}" if ra else "—"])
        return pbr.md_table(["relationship", "n", "freq", "close > open", "mean O→C",
                           "trend day", "up trend day", "down trend day", "range/ATR"], tl)

    @staticmethod
    def _bt_tables(trades, s_is, s_oos, opts):
        cost = CostModel()
        lines = [
            f"Assumptions: capital ₹5,00,000, risk 0.5%/trade, notional cap 20%, slippage "
            f"{opts['slippage_pct']}%/side, brokerage min(₹{cost.brokerage_cap:g}, "
            f"{cost.brokerage_pct * 100:g}%), STT {cost.stt_sell_pct * 100:g}% sell, txn "
            f"{cost.txn_pct * 100:g}%, SEBI ₹{cost.sebi_per_crore:g}/cr, stamp "
            f"{cost.stamp_buy_pct * 100:g}% buy, GST {cost.gst_pct * 100:g}%.\n"]
        hdr = ["setup", "sample", "trades", "win%", "avg win ₹", "avg loss ₹", "PF (net)",
               "gross ₹", "costs ₹", "net ₹", "costs % of gross wins", "exp ₹/trade",
               "exp R (net)", "exp R (gross)", "max DD ₹", "trades/day", "exits"]
        tl = []
        js = {}
        for setup in sorted({t.setup for t in trades}):
            for samp, sess in (("IS", s_is), ("OOS", s_oos), ("ALL", s_is + s_oos)):
                sub = [t for t in trades if t.setup == setup and (
                    samp == "ALL" or (samp == "IS") == (date.fromisoformat(t.day) <= IS_END))]
                s = summarize(sub, sess)
                js[f"{setup}|{samp}"] = s
                if not s["trades"]:
                    tl.append([setup, samp, 0] + ["—"] * (len(hdr) - 3))
                    continue
                pf = s["profit_factor"]
                tl.append([
                    setup, samp, s["trades"] if s["trades"] >= pbr.LOW_N else f"{s['trades']} *",
                    f"{s['win_rate'] * 100:.1f}%", f"{s['avg_win']:,.0f}", f"{s['avg_loss']:,.0f}",
                    "∞" if math.isinf(pf) else f"{pf:.2f}", f"{s['gross']:,.0f}",
                    f"{s['costs']:,.0f}", f"{s['net']:,.0f}",
                    "—" if s["costs_pct_gross_wins"] is None else f"{s['costs_pct_gross_wins']:.1f}%",
                    f"{s['expectancy_rs']:,.0f}", f"{s['expectancy_r']:+.3f}",
                    f"{s['gross_expectancy_r']:+.3f}", f"{s['max_drawdown']:,.0f}",
                    f"{s['trades_per_day']:.2f}",
                    " ".join(f"{k}:{v}" for k, v in sorted(s["exit_reasons"].items())),
                ])
        Path(opts["out_dir"], "pivot_backtest_summary.json").write_text(
            json.dumps(js, indent=1, default=str))
        return "\n".join(lines) + "\n" + pbr.md_table(hdr, tl)


def _atr(bars, n):
    """Simple-average true range of the last ``n`` bars, at each bar's close."""
    out = [None] * len(bars)
    trs = []
    for i, b in enumerate(bars):
        pc = bars[i - 1].c if i else b.o
        trs.append(max(b.h, pc) - min(b.l, pc))
        if i >= n - 1:
            out[i] = sum(trs[i - n + 1:i + 1]) / n
    return out


def _atr_at_1m(bars1m, bars15, atr15):
    """Per 1m bar: the 15m ATR published at the close of the 15m bar that this
    1m bar completes (last 1m bar of each bucket), else None."""
    out = [None] * len(bars1m)
    if not bars15:
        return out
    start_min = {ist_minute(b.t): k for k, b in enumerate(bars15)}
    last_idx: dict[int, int] = {}
    for i, b in enumerate(bars1m):
        bucket_start = 9 * 60 + 15 + ((ist_minute(b.t) - (9 * 60 + 15)) // 15) * 15
        k = start_min.get(bucket_start)
        if k is not None:
            last_idx[k] = i
    for k, i in last_idx.items():
        out[i] = atr15[k]
    return out
