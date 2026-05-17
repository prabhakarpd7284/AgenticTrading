"""python manage.py run_ai_tester — triggers the AI Tester.

See docs/AI_TESTER_CLAUDE.md for what the tester does.
"""
from __future__ import annotations

import json
import sys

from django.core.management.base import BaseCommand

from apps.agents_core.tester import runner, state, llm


class Command(BaseCommand):
    help = "Run the AlphaDesk AI Tester against the live backend."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--verbose", action="store_true",
                            help="Print per-test status as the suite runs.")
        parser.add_argument("--only", default=None,
                            help="Filter to a single suite (auth/meta/plan_stock/...) "
                                 "or a single test id (e.g. pyramid.nifty).")
        parser.add_argument("--reset", action="store_true",
                            help="Wipe the mind palace before this run.")
        parser.add_argument("--llm", action="store_true",
                            help="After the deterministic run, ask Claude CLI for a "
                                 "summary + per-bug fix proposals (no API key required).")
        parser.add_argument("--llm-model", default=None,
                            help="Override the Claude model when --llm is on.")
        parser.add_argument("--json", action="store_true",
                            help="Dump the mind palace JSON to stdout at the end.")

    def handle(self, *args, **opts) -> None:
        if opts["reset"]:
            state.reset()
            self.stdout.write(self.style.WARNING("Mind palace wiped."))

        self.stdout.write("AI Tester · running NIFTY weekly suite ...\n")
        palace, results = runner.run(only=opts["only"], verbose=opts["verbose"])

        # ── Summary table ──
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        self.stdout.write("")
        self.stdout.write(f"  {len(passed)} passed · {len(failed)} failed · {len(results)} total")
        self.stdout.write(f"  mind palace: {state.MIND_PALACE_PATH}")
        self.stdout.write(f"  open bugs:   {len(palace.open_bugs)}")
        self.stdout.write("")

        if failed:
            self.stdout.write(self.style.ERROR("Failures:"))
            for r in failed:
                self.stdout.write(self.style.ERROR(f"  ✗ {r.case.id:32}  {r.error_msg[:200]}"))
            self.stdout.write("")

        # ── Optional LLM pass ──
        if opts["llm"]:
            self.stdout.write("Asking Claude CLI for summary + fix proposals ...")
            enriched = llm.enrich_palace(palace, model=opts.get("llm_model"))
            if enriched:
                state.save(palace)
                self.stdout.write(self.style.SUCCESS("LLM enrichment applied to mind palace."))
                if palace.notes:
                    self.stdout.write(f"  latest note: {palace.notes[-1][:240]}")
            else:
                self.stdout.write(self.style.WARNING(
                    "LLM enrichment skipped — Claude CLI not found OR returned non-JSON. "
                    "Set CLAUDE_CLI_PATH or install `claude`."))

        # ── Open-bug listing (always) ──
        if palace.open_bugs:
            self.stdout.write(self.style.WARNING(f"\nOpen findings ({len(palace.open_bugs)}):"))
            for b in palace.open_bugs:
                sev_style = {
                    "blocker": self.style.ERROR,
                    "high":    self.style.ERROR,
                    "warning": self.style.WARNING,
                    "info":    self.style.NOTICE,
                }.get(b.severity, self.style.NOTICE)
                self.stdout.write(sev_style(
                    f"  [{b.severity.upper():<7}] {b.title}  (×{b.occurrences})\n"
                    f"            id: {b.id}\n"
                    f"            evidence: {b.evidence[:240]}"
                ))
                if b.suggested_fix:
                    self.stdout.write(f"            fix: {b.suggested_fix[:240]}")
        else:
            self.stdout.write(self.style.SUCCESS("\nNo open findings — every check passed."))

        if opts["json"]:
            self.stdout.write("\n--- mind palace ---")
            self.stdout.write(state.MIND_PALACE_PATH.read_text())

        # Exit code reflects whether the suite is currently green
        sys.exit(0 if not failed else 1)
