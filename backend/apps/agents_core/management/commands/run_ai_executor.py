"""python manage.py run_ai_executor — produce a code-change PROPOSAL for
the next highest-priority open task. Does NOT auto-apply.
"""
from __future__ import annotations

from django.core.management.base import BaseCommand

from apps.agents_core.tester import executor, state


class Command(BaseCommand):
    help = "Run the AI Executor agent (proposes a change for the next task; never applies)."

    def handle(self, *args, **opts):
        palace, proposal_id = executor.run()
        if not proposal_id:
            self.stdout.write(self.style.WARNING("executor: no proposal produced (no open tasks or LLM unreachable)."))
            return
        p = next((x for x in palace.proposals if x.id == proposal_id), None)
        if not p:
            return
        self.stdout.write(self.style.SUCCESS(f"executor: proposed {proposal_id}"))
        self.stdout.write(f"\n  task:    {p.task_id}")
        self.stdout.write(f"  summary: {p.summary}")
        if p.files_changed:
            self.stdout.write("\n  files:")
            for fc in p.files_changed:
                self.stdout.write(f"    [{fc.get('change_kind', '?'):<6}] {fc.get('path', '?')}")
                if fc.get("approach"):
                    self.stdout.write(f"             {fc['approach'][:200]}")
        if p.risks:
            self.stdout.write("\n  risks:")
            for r in p.risks: self.stdout.write(f"    - {r}")
        if p.tests_needed:
            self.stdout.write("\n  tests_needed:")
            for t in p.tests_needed: self.stdout.write(f"    - {t}")
        self.stdout.write(
            f"\n  Full proposal stored in {state.MIND_PALACE_PATH}."
        )
