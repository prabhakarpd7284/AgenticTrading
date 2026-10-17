"""parse_nav_text: header interleaving, sentinels, malformed rows, UTF-8 fix."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from apps.mutual_funds.services import amfi_nav_collector as c

SAMPLE = """\
Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date

Open Ended Schemes(Children's Fund - Childrens' Fund)

Axis Mutual Fund
135762;INF846K01WO1;-;Axis Children's Fund;Direct Plan;Growth Option;28.9937;12-Oct-2026
135763;INF846K01WP8;INF846K01WQ6;Axis Children's Fund;Regular Plan;IDCW Option;N.A.;12-Oct-2026

Open Ended Schemes(Equity Scheme - Multi Cap Fund)

Baroda BNP Paribas Mutual Fund
148921;-;-;Baroda Multi Cap;Direct Plan;Growth Option;22.04;01-Oct-2026
bad;row
200001;X;Y;Bad Date;Direct Plan;Growth Option;10.0;not-a-date
"""


def test_headers_attach_to_following_rows():
    rows = c.parse_nav_text(SAMPLE)
    assert [r.scheme_code for r in rows] == ["135762", "148921"]
    first, second = rows
    assert first.amc_name == "Axis Mutual Fund"
    assert first.category == "Open Ended Schemes(Children's Fund - Childrens' Fund)"
    assert second.amc_name == "Baroda BNP Paribas Mutual Fund"
    assert second.category == "Open Ended Schemes(Equity Scheme - Multi Cap Fund)"


def test_fields_and_sentinels():
    first, second = c.parse_nav_text(SAMPLE)
    assert first.isin_growth == "INF846K01WO1"
    assert first.isin_div_reinvestment == ""  # "-" -> empty
    assert first.nav == "28.9937"
    assert first.nav_date == date(2026, 10, 12)
    assert (first.plan, first.option) == ("Direct Plan", "Growth Option")
    assert second.isin_growth == "" and second.isin_div_reinvestment == ""


def test_na_nav_malformed_and_bad_date_rows_are_skipped():
    codes = {r.scheme_code for r in c.parse_nav_text(SAMPLE)}
    assert "135763" not in codes  # N.A. NAV
    assert "200001" not in codes  # unparseable date
    assert "bad" not in codes     # too few fields


def test_empty_and_header_only_input():
    assert c.parse_nav_text("") == []
    assert c.parse_nav_text("Scheme Code;a;b;c;d;e;f;g\n") == []


def _resp(content: bytes):
    r = MagicMock()
    r.content = content
    r.raise_for_status.return_value = None
    return r


def test_fetch_decodes_utf8_not_latin1():
    body = "Open Ended Schemes(Children’s Fund)".encode("utf-8")
    with patch.object(c.requests, "get", return_value=_resp(body)):
        assert c.fetch_nav_text() == "Open Ended Schemes(Children’s Fund)"


def test_fetch_falls_back_to_second_url_then_raises():
    import requests

    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        if url == c.AMFI_NAV_URL:
            raise requests.ConnectionError("down")
        return _resp(b"ok")

    with patch.object(c.requests, "get", side_effect=fake_get):
        assert c.fetch_nav_text() == "ok"
    assert calls == [c.AMFI_NAV_URL, c.AMFI_NAV_URL_FALLBACK]

    with patch.object(c.requests, "get", side_effect=requests.ConnectionError("down")):
        with pytest.raises(RuntimeError, match="Could not fetch"):
            c.fetch_nav_text()
