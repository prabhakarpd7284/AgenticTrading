"""Idempotent backfill: ensure every active user has a personal tenant.

Run once after deploying the auto-bootstrap change so users created before
the change can also obtain JWTs that include a ``tenant_id`` claim.

    python manage.py backfill_personal_tenants
    python manage.py backfill_personal_tenants --dry-run
"""
from __future__ import annotations

from django.core.management.base import BaseCommand

from apps.accounts.models import User
from apps.accounts.services.tenant_bootstrap import ensure_tenant
from apps.tenants.models import Membership


class Command(BaseCommand):
    help = "Ensure every active user has at least one tenant membership."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run", action="store_true",
            help="Report users that would be backfilled without writing.",
        )

    def handle(self, *args, dry_run: bool = False, **opts):
        users = User.objects.filter(is_active=True)
        without_tenant = users.exclude(memberships__is_active=True).distinct()
        total = without_tenant.count()
        self.stdout.write(f"{total} active user(s) need a personal tenant.")
        if dry_run:
            for u in without_tenant.values_list("email", flat=True):
                self.stdout.write(f"  would create for {u}")
            return

        for user in without_tenant:
            tenant = ensure_tenant(user)
            self.stdout.write(self.style.SUCCESS(
                f"  {user.email}  →  {tenant.slug} ({tenant.id})"
            ))
        self.stdout.write(self.style.SUCCESS(
            f"Done. {Membership.objects.count()} memberships in DB."
        ))
