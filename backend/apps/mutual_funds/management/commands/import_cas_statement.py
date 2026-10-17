"""Import a CAMS/KFintech Consolidated Account Statement PDF into MFHolding + MFTransaction.

Usage:
    .venv/bin/python manage.py import_cas_statement --file ~/cas.pdf --tenant <slug|uuid> --portfolio <name|uuid>
    .venv/bin/python manage.py import_cas_statement --file ~/cas.pdf --tenant <slug> --dry-run

The PDF password is PAN+DOB. Prefer the CAS_PASSWORD env var, or omit
--password to be prompted — a --password flag lands in shell history.
The password is never logged or printed.

--tenant may be omitted only when exactly one tenant exists; --portfolio only
when the tenant has exactly one portfolio. Nothing is created implicitly.
Load the AMFI catalog first (pull_amfi_nav) or no scheme will match.
Idempotent: re-running the same statement adds nothing.
"""
from __future__ import annotations

import getpass
import os
import uuid

from django.core.management.base import BaseCommand, CommandError

from apps.mutual_funds.models import MFScheme
from apps.mutual_funds.services.cas_import import import_cas
from apps.mutual_funds.services.cas_parser import CASParseError, parse_cas_pdf
from apps.tenants.models import Tenant
from apps.trading.models import Portfolio


def _as_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


def resolve_tenant(ref: str | None) -> Tenant:
    if ref:
        qs = Tenant.objects.filter(id=_as_uuid(ref)) if _as_uuid(ref) else Tenant.objects.filter(slug=ref)
        tenant = qs.first()
        if tenant is None:
            raise CommandError(f"No tenant with id/slug '{ref}'.")
        return tenant
    tenants = list(Tenant.objects.all()[:10])
    if len(tenants) == 1:
        return tenants[0]
    listing = "\n".join(f"  {t.slug}  ({t.id})  {t.name}" for t in tenants) or "  (none)"
    raise CommandError(f"--tenant is required when there is not exactly one tenant. Choose:\n{listing}")


def resolve_portfolio(tenant: Tenant, ref: str | None) -> Portfolio:
    qs = Portfolio.objects.filter(tenant=tenant)
    if ref:
        pf = (qs.filter(id=_as_uuid(ref)) if _as_uuid(ref) else qs.filter(name=ref)).first()
        if pf is None:
            raise CommandError(f"No portfolio '{ref}' in tenant '{tenant.slug}'.")
        return pf
    portfolios = list(qs[:10])
    if len(portfolios) == 1:
        return portfolios[0]
    listing = "\n".join(f"  {p.name}  ({p.id})  mode={p.mode}" for p in portfolios) or "  (none — create one first)"
    raise CommandError(f"--portfolio is required when tenant '{tenant.slug}' does not have exactly one. Choose:\n{listing}")


class Command(BaseCommand):
    help = "Import a CAMS/KFintech CAS PDF into MFHolding + MFTransaction (source='cas')."

    def add_arguments(self, parser):
        parser.add_argument("--file", required=True, help="Path to the CAS PDF")
        parser.add_argument("--password", default=None, help="PDF password (PAN+DOB). Prefer $CAS_PASSWORD or the prompt.")
        parser.add_argument("--tenant", default=None, help="Tenant slug or UUID")
        parser.add_argument("--portfolio", default=None, help="Portfolio name or UUID within the tenant")
        parser.add_argument("--dry-run", action="store_true", help="Parse and match; write nothing")

    def handle(self, *args, **options):
        path = os.path.expanduser(options["file"])
        if not os.path.isfile(path):
            raise CommandError(f"File not found: {path}")
        if not MFScheme.objects.exists():
            raise CommandError("MFScheme catalog is empty — run `manage.py pull_amfi_nav` first.")

        tenant = resolve_tenant(options["tenant"])
        portfolio = resolve_portfolio(tenant, options["portfolio"])
        password = options["password"] or os.environ.get("CAS_PASSWORD") or getpass.getpass("CAS PDF password (PAN+DOB): ")

        try:
            parsed = parse_cas_pdf(path, password)
        except CASParseError as exc:
            raise CommandError(str(exc)) from None

        self.stdout.write(
            f"{parsed.file_type} {parsed.cas_type} statement {parsed.period_from} -> {parsed.period_to}: "
            f"{len(parsed.schemes)} scheme(s), {sum(len(s.transactions) for s in parsed.schemes)} transaction(s)."
        )
        if parsed.cas_type == "SUMMARY":
            self.stdout.write(self.style.WARNING(
                "SUMMARY statement: holdings only, no transaction history. Request a DETAILED CAS for SIP/cost analysis."
            ))
        for w in parsed.warnings:
            self.stdout.write(self.style.WARNING(f"parse warning: {w}"))
        if parsed.skipped_txn_counts:
            self.stdout.write(f"Skipped non-unit lines: {parsed.skipped_txn_counts}")

        res = import_cas(parsed, tenant=tenant, portfolio=portfolio, dry_run=options["dry_run"])

        tag = "[dry-run] " if res.dry_run else ""
        self.stdout.write(
            f"{tag}matched {res.matched}/{res.schemes_total} schemes | holdings +{res.holdings_created} "
            f"~{res.holdings_updated} | transactions +{res.txns_created} ({res.txns_existing} already present)"
        )
        for u in res.unmatched:
            self.stdout.write(self.style.WARNING(f"unmatched [{u['reason']}]: {u['scheme']} (ISIN {u['isin'] or '-'})"))
        for e in res.errors:
            self.stderr.write(self.style.ERROR(f"error: {e}"))
        if res.matched == 0 and res.schemes_total:
            raise CommandError("No scheme in the statement matched the AMFI catalog.")
        self.stdout.write(self.style.SUCCESS("Done." if not res.errors else "Done with errors."))
