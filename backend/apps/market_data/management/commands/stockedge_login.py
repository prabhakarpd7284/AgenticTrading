"""Mint a Playwright ``storage_state`` for a logged-in StockEdge session.

Run this once, interactively. A headed Chromium opens; you log into StockEdge
(Google OIDC) yourself; then you press Enter here and the session (cookies +
localStorage) is saved to a JSON file. Thereafter::

    manage.py pull_stockedge_breadth --live --storage-state <path>

runs headless against that saved session. Refresh the session by re-running
this command when the refresh token lapses.

Two modes:

* **fresh** (default) — launch a clean headed Chromium and log in there.
  Note: Google sometimes blocks sign-in inside automation-controlled browsers
  ("this browser or app may not be secure"). If that happens, use CDP mode.

* **CDP attach** (``--cdp http://localhost:9222``) — attach to an EXISTING
  Chrome you already logged into, and snapshot its session. Start Chrome with
  ``--remote-debugging-port=9222`` first. This avoids the Google automation
  block entirely because it's your real browser.
"""
from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

_LOGIN_URL = "https://web.stockedge.com/app/markets"


class Command(BaseCommand):
    help = "Open/attach a browser to log into StockEdge and save a Playwright storage_state."

    def add_arguments(self, parser):
        parser.add_argument(
            "--out", default="stockedge_state.json",
            help="Path to write the storage_state JSON (default: ./stockedge_state.json).",
        )
        parser.add_argument(
            "--cdp", default="",
            help="Attach to an existing Chrome via CDP (e.g. http://localhost:9222) "
                 "instead of launching a fresh browser. Avoids Google's automation block.",
        )
        parser.add_argument(
            "--url", default=_LOGIN_URL,
            help="URL to open in fresh mode (default: the StockEdge app).",
        )

    def handle(self, *args, **o):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise CommandError(
                "Playwright not installed. Run:\n"
                "    pip install playwright && playwright install chromium"
            )

        out = Path(o["out"]).resolve()

        with sync_playwright() as p:
            if o["cdp"]:
                self.stdout.write(self.style.WARNING(
                    f"Attaching to existing Chrome at {o['cdp']} …"
                ))
                try:
                    browser = p.chromium.connect_over_cdp(o["cdp"])
                except Exception as exc:  # noqa: BLE001
                    raise CommandError(
                        f"Could not attach to Chrome at {o['cdp']}: {exc}\n"
                        "Start Chrome with --remote-debugging-port=9222 and make sure "
                        "you're logged into web.stockedge.com in it."
                    )
                context = browser.contexts[0] if browser.contexts else browser.new_context()
                self.stdout.write(
                    "Make sure a tab in that Chrome is logged into web.stockedge.com, then "
                )
                input(">>> press Enter here to snapshot the session… ")
            else:
                self.stdout.write(self.style.WARNING(
                    "Launching a headed Chromium — log into StockEdge there.\n"
                    "(If Google refuses sign-in in this window, re-run with --cdp.)"
                ))
                browser = p.chromium.launch(headless=False)
                context = browser.new_context()
                page = context.new_page()
                page.goto(o["url"])
                input(
                    "\n>>> Log into StockEdge in the opened browser, navigate to the app, "
                    "then press Enter here to save the session… "
                )

            context.storage_state(path=str(out))
            try:
                browser.close()
            except Exception:  # noqa: BLE001
                pass

        self.stdout.write(self.style.SUCCESS(f"Saved storage_state → {out}"))
        self.stdout.write(
            "Next: manage.py pull_stockedge_breadth --live --storage-state " + str(out)
        )
