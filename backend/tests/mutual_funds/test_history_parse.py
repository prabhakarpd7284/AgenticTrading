"""amfi_history: parsing, year chunking, AMC-id resolution (requests mocked)."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import patch

import pytest
import requests

from apps.mutual_funds.services import amfi_history as h

HIST = """\
Scheme Code;NAV Name;Plan;Option;ISIN Div Payout/ISIN Growth;ISIN Div Reinvestment;Net Asset Value;Date

Open Ended Schemes ( Equity Scheme - Multi Cap Fund )

Aditya Birla Sun Life Mutual Fund
148921;ABSL Multi-Cap Direct Growth;Direct Plan;GROWTH;INF209KB1Y49;;22.04;01-Oct-2026
148921;ABSL Multi-Cap Direct Growth;Direct Plan;GROWTH;INF209KB1Y49;;N.A.;02-Oct-2026
148921;ABSL Multi-Cap Direct Growth;Direct Plan;GROWTH;INF209KB1Y49;;22.15;bad-date
999999;Other Scheme;Direct Plan;GROWTH;X;;10.5;01-Oct-2026
"""


def test_parse_keeps_good_rows_and_skips_bad():
    rows = h.parse_history_text(HIST)
    assert [(r.scheme_code, r.nav_date, r.nav) for r in rows] == [
        ("148921", date(2026, 10, 1), Decimal("22.04")),
        ("999999", date(2026, 10, 1), Decimal("10.5")),
    ]


def test_parse_filters_by_scheme_codes():
    rows = h.parse_history_text(HIST, {"148921"})
    assert [r.scheme_code for r in rows] == ["148921"]
    assert h.parse_history_text(HIST, set()) == []


def test_year_chunks_cover_range_without_gaps_or_overlap():
    start, end = date(2024, 10, 15), date(2026, 10, 15)
    chunks = h.year_chunks(start, end)
    assert chunks[0][0] == start and chunks[-1][1] == end
    for (_, prev_end), (nxt_start, _) in zip(chunks, chunks[1:]):
        assert (nxt_start - prev_end).days == 1
    assert all((e - s).days <= 364 for s, e in chunks)


def test_year_chunks_short_and_single_day():
    d = date(2026, 1, 1)
    assert h.year_chunks(d, d) == [(d, d)]
    assert h.year_chunks(d, date(2025, 1, 1)) == []


@pytest.fixture(autouse=True)
def _clear_amc_cache(tmp_path, monkeypatch):
    # Probe tests must not be short-circuited by the persisted id map.
    monkeypatch.setattr(h, "AMC_ID_FILE", tmp_path / "missing.json")
    h._AMC_ID_CACHE.clear()
    yield
    h._AMC_ID_CACHE.clear()


def test_resolve_amc_id_uses_persisted_map_without_probing(tmp_path, monkeypatch):
    f = tmp_path / "ids.json"
    f.write_text('{"amc_ids": {"HDFC Mutual Fund": 9}}')
    monkeypatch.setattr(h, "AMC_ID_FILE", f)
    with patch.object(h, "fetch_history_text", side_effect=AssertionError("no probing")):
        assert h.resolve_amc_id("hdfc mutual fund") == 9
        assert h.resolve_amc_id("Unknown Mutual Fund", probe=False) is None


def test_shipped_amc_id_map_is_valid():
    import json
    ids = json.loads(h.Path(h.__file__).resolve().parent.parent.joinpath("data", "amfi_amc_ids.json").read_text())
    assert ids["amc_ids"]["HDFC Mutual Fund"] == 9


def test_resolve_amc_id_probes_until_match_and_caches():
    texts = {
        1: 'alert("Select Mutual Fund");',
        2: "Axis Mutual Fund\n1;x;y;z;1;;1;01-Oct-2026",
        3: "Aditya Birla Sun Life Mutual Fund\n1;x;y;z;1;;1;01-Oct-2026",
        4: "Never Reached Mutual Fund",
    }
    calls = []

    def fake_fetch(amc_id, start, end, timeout=120):
        calls.append(amc_id)
        return texts[amc_id]

    with patch.object(h, "fetch_history_text", side_effect=fake_fetch):
        assert h.resolve_amc_id("Aditya Birla Sun Life Mutual Fund") == 3
        assert calls == [1, 2, 3]
        # Cached: no further fetches, and ids seen while probing are cached too.
        assert h.resolve_amc_id("Aditya Birla Sun Life Mutual Fund") == 3
        assert h.resolve_amc_id("axis mutual fund") == 2
        assert calls == [1, 2, 3]


def test_resolve_amc_id_survives_request_errors_and_returns_none():
    def fake_fetch(amc_id, start, end, timeout=120):
        raise requests.ConnectionError("down")

    with patch.object(h, "fetch_history_text", side_effect=fake_fetch):
        assert h.resolve_amc_id("Nobody Mutual Fund", max_id=3) is None


def test_fetch_history_text_sends_params_and_decodes_utf8():
    class R:
        content = "Children’s".encode("utf-8")

        def raise_for_status(self):
            pass

    with patch.object(h.requests, "get", return_value=R()) as get:
        out = h.fetch_history_text(3, date(2026, 10, 1), date(2026, 10, 14))
    assert out == "Children’s"
    assert get.call_args.kwargs["params"] == {"mf": 3, "frmdt": "01-Oct-2026", "todt": "14-Oct-2026"}
