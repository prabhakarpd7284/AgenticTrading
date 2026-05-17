"""python manage.py run_ai_trader_user — generate feature requests
from a trader's POV. Writes into the shared mind palace.
"""
from __future__ import annotations

from django.core.management.base import BaseCommand

from apps.agents_core.tester import trader_user


class Command(BaseCommand):
    help = "Run the AI Trader-User agent (generates feature requests)."

    def handle(self, *args, **opts):
        palace, added = trader_user.run()
        self.stdout.write(self.style.SUCCESS(
            f"trader_user: {added} feature request{'s' if added != 1 else ''} added/updated."
        ))
        if palace.feature_requests:
            self.stdout.write("\nLatest feature requests:")
            for f in palace.feature_requests[-5:]:
                self.stdout.write(f"  [{f.category}] {f.title}\n    rationale: {f.rationale[:200]}")
