"""pull_amfi_nav + backfill_amfi_history commands, all network mocked."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.mutual_funds.models import MFNav, MFScheme

PULL = "apps.mutual_funds.management.commands.pull_amfi_nav"
BACKFILL = "apps.mutual_funds.management.commands.backfill_amfi_history"

DAILY = """\
Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date

Open Ended Schemes(Equity Scheme - Multi Cap Fund)

Aditya Birla Sun Life Mutual Fund
148921;INF209KB1Y49;-;ABSL Multi-Cap;Direct Plan;Growth Option;22.04;14-Oct-2026
148922;-;-;ABSL Multi-Cap;Regular Plan;Growth Option;20.00;14-Oct-2026
"""


def _run(cmd, **kw):
    out, err = StringIO(), StringIO()
    call_command(cmd, stdout=out, stderr=err, **kw)
    return out.getvalue(), err.getvalue()


@pytest.mark.django_db
def test_pull_upserts_and_is_idempotent(monkeypatch):
    monkeypatch.setattr(f"{PULL}.fetch_nav_text", lambda: DAILY)
    out, _ = _run("pull_amfi_nav")
    assert "2 schemes upserted, 2 new NAV rows" in out
    assert MFScheme.objects.count() == 2 and MFNav.objects.count() == 2
    direct = MFScheme.objects.get(scheme_code="148921")
    assert direct.is_direct is True and direct.amc_name == "Aditya Birla Sun Life Mutual Fund"
    assert MFScheme.objects.get(scheme_code="148922").is_direct is False
    assert MFNav.objects.get(scheme_id="148921").nav == Decimal("22.0400")

    out, _ = _run("pull_amfi_nav")
    assert "0 new NAV rows (2 already present" in out
    assert MFNav.objects.count() == 2


@pytest.mark.django_db
def test_pull_dry_run_writes_nothing(monkeypatch):
    monkeypatch.setattr(f"{PULL}.fetch_nav_text", lambda: DAILY)
    out, _ = _run("pull_amfi_nav", dry_run=True)
    assert "[dry-run] 2 distinct schemes" in out
    assert MFScheme.objects.count() == 0


@pytest.mark.django_db
def test_pull_zero_rows_guard_writes_nothing(monkeypatch):
    monkeypatch.setattr(f"{PULL}.fetch_nav_text", lambda: "garbage with no rows\n")
    _, err = _run("pull_amfi_nav")
    assert "Zero rows parsed" in err
    assert MFScheme.objects.count() == 0


@pytest.mark.django_db
def test_pull_fetch_failure_does_not_crash(monkeypatch):
    def boom():
        raise RuntimeError("Could not fetch AMFI NAV file")

    monkeypatch.setattr(f"{PULL}.fetch_nav_text", boom)
    _, err = _run("pull_amfi_nav")
    assert "Could not fetch" in err
    assert MFScheme.objects.count() == 0


@pytest.mark.django_db
def test_pull_updates_existing_scheme_metadata(monkeypatch):
    MFScheme.objects.create(scheme_code="148921", amc_name="Old", scheme_name="Old name")
    monkeypatch.setattr(f"{PULL}.fetch_nav_text", lambda: DAILY)
    _run("pull_amfi_nav")
    assert MFScheme.objects.get(scheme_code="148921").scheme_name == "ABSL Multi-Cap"


HIST = """\
Aditya Birla Sun Life Mutual Fund
148921;ABSL;Direct Plan;GROWTH;INF209KB1Y49;;22.04;01-Oct-2026
148921;ABSL;Direct Plan;GROWTH;INF209KB1Y49;;22.15;05-Oct-2026
777777;Not requested;Direct Plan;GROWTH;X;;9.9;01-Oct-2026
"""


@pytest.fixture
def seeded_scheme(db):
    return MFScheme.objects.create(
        scheme_code="148921", amc_name="Aditya Birla Sun Life Mutual Fund", scheme_name="ABSL"
    )


@pytest.mark.django_db
def test_backfill_inserts_only_requested_scheme_and_is_idempotent(monkeypatch, seeded_scheme):
    monkeypatch.setattr(f"{BACKFILL}.resolve_amc_id", lambda name: 3)
    monkeypatch.setattr(f"{BACKFILL}.fetch_history_text", lambda amc, s, e: HIST)
    out, _ = _run("backfill_amfi_history", scheme=["148921"], years=1)
    assert MFNav.objects.filter(scheme_id="148921").count() == 2
    assert not MFNav.objects.filter(scheme_id="777777").exists()
    assert "Done." in out

    out, _ = _run("backfill_amfi_history", scheme=["148921"], years=1)
    assert "Done. 0 new NAV rows." in out
    assert MFNav.objects.count() == 2


@pytest.mark.django_db
def test_backfill_unknown_scheme_raises():
    with pytest.raises(CommandError, match="Unknown scheme"):
        call_command("backfill_amfi_history", scheme=["000000"])


@pytest.mark.django_db
def test_backfill_unresolvable_amc_is_skipped_not_fatal(monkeypatch, seeded_scheme):
    monkeypatch.setattr(f"{BACKFILL}.resolve_amc_id", lambda name: None)
    _, err = _run("backfill_amfi_history", scheme=["148921"], years=1)
    assert "Could not resolve AMFI id" in err
    assert MFNav.objects.count() == 0


@pytest.mark.django_db
def test_backfill_fetch_failure_continues(monkeypatch, seeded_scheme):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(f"{BACKFILL}.resolve_amc_id", lambda name: 3)
    monkeypatch.setattr(f"{BACKFILL}.fetch_history_text", boom)
    out, err = _run("backfill_amfi_history", scheme=["148921"], years=1)
    assert "fetch failed" in err and "Done. 0 new NAV rows." in out


@pytest.mark.django_db
def test_backfill_dry_run_and_amc_id_override(monkeypatch, seeded_scheme):
    def no_probe(name):
        raise AssertionError("resolve_amc_id must not run when --amc-id is given")

    monkeypatch.setattr(f"{BACKFILL}.resolve_amc_id", no_probe)
    monkeypatch.setattr(f"{BACKFILL}.fetch_history_text", lambda amc, s, e: HIST)
    out, _ = _run("backfill_amfi_history", scheme=["148921"], years=1, amc_id=3, dry_run=True)
    assert "[dry-run]" in out and MFNav.objects.count() == 0


@pytest.mark.django_db
def test_backfill_amc_id_with_multiple_amcs_rejected(monkeypatch, seeded_scheme):
    MFScheme.objects.create(scheme_code="2", amc_name="Other Mutual Fund", scheme_name="o")
    with pytest.raises(CommandError, match="--amc-id"):
        call_command("backfill_amfi_history", "--scheme", "148921", "--scheme", "2", "--amc-id", "3")
