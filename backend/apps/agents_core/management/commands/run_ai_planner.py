"""python manage.py run_ai_planner — convert bugs + feature requests
into ordered tasks. Writes into the shared mind palace.
"""
from __future__ import annotations

from django.core.management.base import BaseCommand

from apps.agents_core.tester import planner


class Command(BaseCommand):
    help = "Run the AI Planner agent (produces tasks from bugs + feature requests)."

    def handle(self, *args, **opts):
        palace, added = planner.run()
        self.stdout.write(self.style.SUCCESS(
            f"planner: {added} task{'s' if added != 1 else ''} added/updated."
        ))
        if palace.tasks:
            self.stdout.write("\nOpen tasks (top 8):")
            open_tasks = [t for t in palace.tasks if t.status == "open"][:8]
            for t in open_tasks:
                self.stdout.write(f"  [{t.priority.upper():<6}] {t.title}")
                self.stdout.write(f"           scope: {t.scope[:160]}")
                if t.files:
                    self.stdout.write(f"           files: {', '.join(t.files[:5])}")
