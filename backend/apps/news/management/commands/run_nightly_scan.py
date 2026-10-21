"""Run the 21:00 IST nightly scan by hand (TRADING_RULES.md §5).

    manage.py run_nightly_scan                     # next trading day
    manage.py run_nightly_scan --date 2026-10-20   # scan the evening of that session
    manage.py run_nightly_scan --no-telegram --skip-refresh
"""
from __future__ import annotations

import json
from datetime import date

from django.core.management.base import BaseCommand, CommandError

from apps.news.services.nightly_scan import ScanAlreadyRunning, run_nightly_scan


class Command(BaseCommand):
    help = "News + sentiment + levels nightly scan for the next NSE trading day."

    def add_arguments(self, parser):
        parser.add_argument("--date", default="",
                            help="Session (evening) date YYYY-MM-DD; the scan governs the "
                                 "next trading day. Default: the upcoming trading day.")
        parser.add_argument("--no-telegram", action="store_true", help="Don't send the summary.")
        parser.add_argument("--skip-refresh", action="store_true",
                            help="Don't run the bhavcopy / 1m candle backfills first.")
        parser.add_argument("--symbols", default="",
                            help="Comma-separated subset of NIFTY 50 (debugging).")

    def handle(self, *args, **opts):
        session = None
        if opts["date"]:
            try:
                session = date.fromisoformat(opts["date"])
            except ValueError as e:
                raise CommandError(f"--date must be YYYY-MM-DD: {e}") from e
        symbols = [s.strip().upper() for s in opts["symbols"].split(",") if s.strip()] or None
        try:
            scan = run_nightly_scan(session_date=session, telegram=not opts["no_telegram"],
                                    refresh=not opts["skip_refresh"], symbols=symbols)
        except ScanAlreadyRunning as e:
            raise CommandError(str(e)) from e

        s = scan.summary
        self.stdout.write(self.style.SUCCESS(
            f"Nightly scan for {scan.for_date} (session {scan.session_date}): {scan.status}"))
        self.stdout.write(f"timings: {json.dumps(scan.timings)}")
        for k, v in scan.sources.items():
            self.stdout.write(f"  {k}: {json.dumps(v, default=str)[:300]}")
        m = s.get("market", {})
        self.stdout.write(f"market: {m.get('label')} score={m.get('score')} "
                          f"breadth={m.get('breadth')} headlines={s.get('headlines_scored')}/"
                          f"{s.get('headlines_total')}")
        for r in s.get("skip", []):
            self.stdout.write(f"  restricted {r['symbol']} [{r['allowed']}]: {'; '.join(r['reasons'])}")
        if s.get("degraded"):
            self.stdout.write(self.style.WARNING(f"degraded: {s['degraded']}"))
        self.stdout.write(f"telegram sent: {scan.telegram_sent}")
