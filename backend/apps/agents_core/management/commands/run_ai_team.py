"""python manage.py run_ai_team — orchestrates the whole loop in sequence.

Default order: tester → trader_user → planner → executor

  * tester runs first so we ground every downstream decision in the
    latest factual test results (open_bugs is fresh).
  * trader_user reads the updated palace + adds feature requests.
  * planner converts bugs + requests into tasks.
  * executor picks the highest-priority task and proposes a change.

Each step can be skipped with --skip-<step>. Each step writes its outcome
into the shared mind palace + an agent_runs entry.
"""
from __future__ import annotations

from django.core.management.base import BaseCommand

from apps.agents_core.tester import runner, trader_user, planner, executor, state
from apps.agents_core.tester.trader_profiles import PROFILES


class Command(BaseCommand):
    help = "Run the full AI team loop: tester → trader_user → planner → executor."

    def add_arguments(self, parser):
        parser.add_argument("--skip-tester", action="store_true")
        parser.add_argument("--skip-trader", action="store_true")
        parser.add_argument("--skip-planner", action="store_true")
        parser.add_argument("--skip-executor", action="store_true")
        parser.add_argument("--verbose", action="store_true",
                            help="Per-test detail during the tester step.")
        parser.add_argument(
            "--profile", default="default",
            choices=list(PROFILES.keys()),
            help="Trader-user persona to use for the trader step.",
        )
        parser.add_argument(
            "--all-profiles", action="store_true",
            help="Run trader_user against every profile in sequence (ignores --profile).",
        )

    def handle(self, *args, **opts):
        self.stdout.write(self.style.MIGRATE_HEADING("AI Team — start"))

        # 1) Tester
        if not opts["skip_tester"]:
            self.stdout.write("\n[1/4] tester ...")
            palace, results = runner.run(only=None, verbose=opts["verbose"])
            passed = sum(1 for r in results if r.passed)
            failed = len(results) - passed
            tone = self.style.SUCCESS if failed == 0 else self.style.WARNING
            self.stdout.write(tone(f"      {passed} passed · {failed} failed · {len(palace.open_bugs)} open bugs total"))
        else:
            self.stdout.write(self.style.WARNING("[1/4] tester — skipped"))

        # 2) Trader user — single profile or fan out across all
        if not opts["skip_trader"]:
            if opts["all_profiles"]:
                self.stdout.write("\n[2/4] trader_user × all profiles ...")
                results = trader_user.run_all_profiles()
                total = sum(results.values())
                for pid, n in results.items():
                    self.stdout.write(f"      {pid}: {n}")
                palace = state.load()
                self.stdout.write(self.style.SUCCESS(
                    f"      {total} feature request(s) added/updated across {len(results)} personas "
                    f"· {len(palace.feature_requests)} total"
                ))
            else:
                profile = opts["profile"]
                self.stdout.write(f"\n[2/4] trader_user [{profile}] ...")
                palace, added = trader_user.run(profile=profile)
                self.stdout.write(self.style.SUCCESS(
                    f"      {added} feature request(s) added/updated · "
                    f"{len(palace.feature_requests)} total"
                ))
        else:
            self.stdout.write(self.style.WARNING("[2/4] trader_user — skipped"))

        # 3) Planner
        if not opts["skip_planner"]:
            self.stdout.write("\n[3/4] planner ...")
            palace, added = planner.run()
            open_tasks = sum(1 for t in palace.tasks if t.status == "open")
            self.stdout.write(self.style.SUCCESS(f"      {added} task(s) added/updated · {open_tasks} open task(s)"))
        else:
            self.stdout.write(self.style.WARNING("[3/4] planner — skipped"))

        # 4) Executor
        if not opts["skip_executor"]:
            self.stdout.write("\n[4/4] executor ...")
            palace, proposal_id = executor.run()
            if proposal_id:
                self.stdout.write(self.style.SUCCESS(f"      proposed {proposal_id}"))
            else:
                self.stdout.write(self.style.WARNING("      no proposal produced (no open tasks or LLM unreachable)"))
        else:
            self.stdout.write(self.style.WARNING("[4/4] executor — skipped"))

        # Summary
        palace = state.load()
        self.stdout.write(self.style.MIGRATE_HEADING(f"\nAI Team — done"))
        self.stdout.write(f"  palace:           {state.MIND_PALACE_PATH}")
        self.stdout.write(f"  open bugs:        {len(palace.open_bugs)}")
        self.stdout.write(f"  feature requests: {len(palace.feature_requests)}")
        self.stdout.write(f"  open tasks:       {sum(1 for t in palace.tasks if t.status == 'open')}")
        self.stdout.write(f"  proposals (new):  {sum(1 for p in palace.proposals if p.status == 'proposed')}")
        self.stdout.write(f"  agent runs total: {len(palace.agent_runs)}")
        if palace.notes:
            self.stdout.write(f"  latest note:      {palace.notes[-1][:240]}")
