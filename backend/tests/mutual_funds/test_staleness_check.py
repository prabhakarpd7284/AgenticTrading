"""staleness: trading_days_back + stale_schemes + check_stale_navs command."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from io import StringIO

import pytest
from django.core.management import call_command

from apps.mutual_funds.models import MFNav, MFScheme
from apps.mutual_funds.services.staleness import stale_schemes, trading_days_back

TODAY = date(2026, 10, 15)  # Thursday; 12-14 Oct are plain trading days


def test_trading_days_back_skips_weekends():
    assert trading_days_back(TODAY, 3) == date(2026, 10, 12)
    # Monday back 1 trading day -> previous Friday
    assert trading_days_back(date(2026, 10, 12), 1) == date(2026, 10, 9)
    assert trading_days_back(TODAY, 0) == TODAY


def test_trading_days_back_skips_holiday():
    # 2 Oct 2026 (Gandhi Jayanti, Fri) is an NSE holiday -> Mon 5 Oct back 1 = Thu 1 Oct.
    assert trading_days_back(date(2026, 10, 5), 1) == date(2026, 10, 1)


def _scheme(code, amc="Test Mutual Fund"):
    return MFScheme.objects.create(scheme_code=code, amc_name=amc, scheme_name=f"Scheme {code}")


def _nav(code, d):
    MFNav.objects.create(scheme_id=code, nav_date=d, nav=Decimal("10"))


@pytest.mark.django_db
def test_stale_schemes_boundary_and_ordering():
    _scheme("fresh"); _nav("fresh", date(2026, 10, 14))
    _scheme("edge"); _nav("edge", date(2026, 10, 12))    # exactly 3 trading days: NOT stale
    _scheme("stale"); _nav("stale", date(2026, 10, 9))   # 4 trading days: stale
    _scheme("older"); _nav("older", date(2020, 1, 1))
    _scheme("never")
    _scheme("multi"); _nav("multi", date(2020, 1, 1)); _nav("multi", date(2026, 10, 14))  # latest wins

    result = stale_schemes(today=TODAY)
    codes = [s.scheme_code for s, _ in result]
    assert codes == ["older", "stale", "never"]  # oldest first, never-priced last
    assert dict((s.scheme_code, d) for s, d in result)["never"] is None


@pytest.mark.django_db
def test_stale_schemes_custom_threshold():
    _scheme("a"); _nav("a", date(2026, 10, 9))
    assert [s.scheme_code for s, _ in stale_schemes(today=TODAY, max_trading_days=5)] == []
    assert [s.scheme_code for s, _ in stale_schemes(today=TODAY, max_trading_days=3)] == ["a"]


@pytest.mark.django_db
def test_check_stale_navs_command_output(monkeypatch):
    s = _scheme("a", amc="Zed Mutual Fund"); _nav("a", date(2020, 1, 1))
    monkeypatch.setattr(
        "apps.mutual_funds.management.commands.check_stale_navs.stale_schemes",
        lambda max_trading_days: [(s, date(2020, 1, 1))],
    )
    out = StringIO()
    call_command("check_stale_navs", stdout=out)
    text = out.getvalue()
    assert "1 scheme(s) with no NAV in >3 trading days" in text
    assert "Zed Mutual Fund" in text and "2020-01-01" in text

    out = StringIO()
    call_command("check_stale_navs", limit=0, days=7, stdout=out)
    assert ">7 trading days" in out.getvalue()
    assert "last NAV" not in out.getvalue()
