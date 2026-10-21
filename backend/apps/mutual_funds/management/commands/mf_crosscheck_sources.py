"""Cross-check mfapi.in NAV history against AMFI's own NAV-history report.

Usage:
    .venv/bin/python manage.py mf_crosscheck_sources                      # 20 schemes, 2 windows
    .venv/bin/python manage.py mf_crosscheck_sources --sample 20 --amcs 5 --seed 7

Picks ``--sample`` universe schemes spread over ``--amcs`` AMCs whose AMFI id
is known (data/amfi_amc_ids.json), downloads one old and one recent month
from BOTH sources, and compares every common date. Exits non-zero (fails
loudly) if any NAV differs by more than ``--tol`` (relative) or if more than
``--max-missing-pct`` of one source's dates are missing in the other.
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from datetime import date, timedelta

import requests
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Min

from apps.mutual_funds.models import MFNav

from apps.mutual_funds.services.amfi_history import fetch_history_text, known_amc_ids, parse_history_text, _norm
from apps.mutual_funds.services.mfapi_client import MfapiClient
from apps.mutual_funds.services.universe import universe_schemes


class Command(BaseCommand):
    help = "Compare mfapi.in vs AMFI NAV history on a sample of schemes; fail loudly on mismatch."

    def add_arguments(self, parser):
        parser.add_argument("--sample", type=int, default=20)
        parser.add_argument("--amcs", type=int, default=5)
        parser.add_argument("--seed", type=int, default=7)
        parser.add_argument("--old-window", default="2019-03-01:2019-03-31", help="YYYY-MM-DD:YYYY-MM-DD")
        parser.add_argument("--recent-days", type=int, default=30)
        parser.add_argument("--tol", type=float, default=1e-4, help="Max relative NAV difference")
        parser.add_argument("--max-missing-pct", type=float, default=5.0)

    def handle(self, *args, **o):
        ids = known_amc_ids()
        a, b = o["old_window"].split(":")
        old_start = date.fromisoformat(a)
        # Prefer schemes that already existed in the old window (so it is
        # actually compared); falls back to any scheme if none are loaded yet.
        first_nav = dict(MFNav.objects.values("scheme_id").annotate(f=Min("nav_date")).values_list("scheme_id", "f"))
        by_amc = defaultdict(list)
        for s in universe_schemes():
            if _norm(s.amc_name) in ids and (first_nav.get(s.scheme_code) or date.max) < old_start:
                by_amc[s.amc_name].append(s)
        if not by_amc:
            for s in universe_schemes():
                if _norm(s.amc_name) in ids:
                    by_amc[s.amc_name].append(s)
        rng = random.Random(o["seed"])
        amcs = rng.sample(sorted(a for a, v in by_amc.items() if len(v) >= 4), o["amcs"])
        per = max(1, o["sample"] // len(amcs))
        picked = {a: rng.sample(sorted(by_amc[a], key=lambda s: s.scheme_code), per) for a in amcs}

        today = date.today()
        windows = [(date.fromisoformat(a), date.fromisoformat(b)),
                   (today - timedelta(days=o["recent_days"]), today)]
        client = MfapiClient()
        amfi_bytes = 0
        rows_out = []
        failures = []
        for amc, schemes in picked.items():
            codes = {s.scheme_code for s in schemes}
            for start, end in windows:
                try:
                    text = fetch_history_text(ids[_norm(amc)], start, end)
                except requests.RequestException as exc:
                    failures.append(f"AMFI fetch failed for {amc} {start}: {exc}")
                    continue
                amfi_bytes += len(text.encode())
                amfi = defaultdict(dict)
                for r in parse_history_text(text, codes):
                    amfi[r.scheme_code][r.nav_date] = r.nav
                time.sleep(1.0)
                for s in schemes:
                    h = client.scheme_history(s.scheme_code, start=start, end=end)
                    mf = dict(h.navs)
                    am = amfi.get(s.scheme_code, {})
                    common = sorted(set(mf) & set(am))
                    max_rel = max((abs(float(mf[d]) / float(am[d]) - 1) for d in common if am[d]), default=0.0)
                    # Liquid/overnight funds get forward-dated weekend NAVs in
                    # AMFI's file; mfapi drops some. Weekend-only gaps are
                    # reported but not counted as a mismatch.
                    diff = (set(mf) ^ set(am))
                    weekend_only = sum(1 for d in diff if d.weekday() >= 5)
                    only_mf = len({d for d in set(mf) - set(am) if d.weekday() < 5})
                    only_am = len({d for d in set(am) - set(mf) if d.weekday() < 5})
                    denom = max(len(set(mf) | set(am)), 1)
                    missing_pct = (only_mf + only_am) / denom * 100
                    ok = max_rel <= o["tol"] and missing_pct <= o["max_missing_pct"] and (common or not (mf or am))
                    rows_out.append((s.scheme_code, s.scheme_name[:48], f"{start}..{end}", len(common),
                                     only_mf, only_am, max_rel, ("OK" if ok else "MISMATCH")
                                     + (f" (+{weekend_only} weekend-only)" if weekend_only else "")))
                    if not ok:
                        failures.append(f"{s.scheme_code} {start}..{end}: common={len(common)} "
                                        f"only_mfapi={only_mf} only_amfi={only_am} max_rel={max_rel:.2e}")

        self.stdout.write(f"{'code':>7} {'scheme':48} {'window':24} {'common':>6} {'only_mf':>7} "
                          f"{'only_amfi':>9} {'max_rel_diff':>12} result")
        for r in rows_out:
            self.stdout.write(f"{r[0]:>7} {r[1]:48} {r[2]:24} {r[3]:>6} {r[4]:>7} {r[5]:>9} {r[6]:>12.2e} {r[7]}")
        n_schemes = sum(len(v) for v in picked.values())
        self.stdout.write(f"\n{n_schemes} schemes x {len(windows)} windows, {len(picked)} AMCs; "
                          f"AMFI downloaded {amfi_bytes / 1e6:.1f} MB, mfapi {client.bytes_downloaded / 1e6:.2f} MB.")
        if failures:
            for f in failures:
                self.stderr.write(self.style.ERROR(f))
            raise CommandError(f"Cross-check FAILED: {len(failures)} problem(s).")
        self.stdout.write(self.style.SUCCESS("Cross-check passed: every common date matches within tolerance."))
