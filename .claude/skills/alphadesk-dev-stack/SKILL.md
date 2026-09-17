---
name: alphadesk-dev-stack
description: Use when booting, restarting, or debugging the AlphaDesk local dev stack, when broker links read errored/expired, when Angel One hits rate limits or the breaker trips, or before trusting any strategy output as real market data
---

# AlphaDesk Dev Stack

## Overview

Booting AlphaDesk is not just "start the processes." Two failure modes produce
**confident, plausible-looking output built on nothing real**, and neither
announces itself as an error. Check both before trusting anything the system says.

## Boot Sequence

```bash
docker compose -f docker-compose.dev.yml up -d   # Postgres :5444, Redis :6380
bash scripts/dev_up.sh                           # web :8000, celery, beat, flower :5555, vite :5173
```

Ports (the docs drift — these are the compose-file truth):

| Service | Port | Note |
|---|---|---|
| Postgres | **5444** | 5432–5443 belong to Hopzy projects; never bind there |
| Redis | **6380** | `:6379` is a *different* instance — never point AlphaDesk at it |
| Web (Daphne) | 8000 | ASGI; `runserver` is WSGI and breaks `/ws/*` |
| Vite | 5173 | |
| Flower | 5555 | |

## Check 1: Which Tenant Are You In

`admin@local.dev` / `devadmin` — the login advertised in CLAUDE.md and README —
is **not** the working account. It is a separate tenant with its own stale
broker links. Inspecting it and reporting the result produces confidently wrong
answers about broker state.

Working account: `prabhakarpd7284@gmail.com` → tenant
`29e91b00-af66-4e36-8a80-9f2a9a304cf0`.

Mint a token rather than guessing a password:

```python
from apps.accounts.models import User
from apps.accounts.api.jwt import _attach_tenant_claims
from rest_framework_simplejwt.tokens import RefreshToken
u = User.objects.get(email='prabhakarpd7284@gmail.com')
r = RefreshToken.for_user(u); _attach_tenant_claims(r, u)
print(str(r.access_token))
```

## Check 2: Is Market Data Real

`GET /api/v1/brokers/` — at least one link on the working tenant must be
`status: active`. If none is, `apps/market_data/services/data_port.py` falls
back to `PaperBrokerAdapter` and serves a **synthesised BSM options chain**.
Its own comment: *"Strategy code can't tell the difference; only the `.source`
field changes."* Strategies emit normal-looking signals on fabricated prices,
with nothing louder than a `WARNING` in the log.

Distinguish the two broker failure shapes — they need opposite responses:

| Field | Meaning | Fix |
|---|---|---|
| `requires_daily_login: true`, `token_valid_today: false` | Zerodha/Fyers token expired overnight (by design) | Browser OAuth at `/broker` — no way around it |
| `status: errored` + `rate-limit cooldown` in `last_error` | Angel One is *fine*; we are throttling ourselves | Stop the load source, wait for the breaker |

Angel uses TOTP and does **not** expire daily. `token_valid_today: True` with
`status: errored` means auth is healthy and something is hammering the API.

Prove the broker end-to-end rather than reading status fields:

```python
from trading.services.data_service import BrokerClient
print(BrokerClient().ltp('NSE', 'NIFTY', '99926000'))
```

## Symptom: Angel Rate-Limited Right After Boot

The daily pipeline is crontab-scheduled (`config/settings/base.py`): 08:20
swing scan over NIFTY 100, 16:00 EOD enrichment, 16:30 trade derivation.
`backend/celerybeat-schedule.db` persists `last_run_at`, so after the stack has
been down a while **every entry reads overdue and beat fires them all at once**.
The burst blows Angel's rate limit, the breaker trips repeatedly, and each
retry re-trips it — links stay `errored` and never recover on their own.

`scripts/dev_up.sh` now clears that file automatically when it predates today
(same-day restarts still catch up legitimately). If you hit the storm anyway:

```bash
bash scripts/dev_down.sh
cd backend && DJANGO_SETTINGS_MODULE=config.settings.dev .venv/bin/celery -A config purge -f
rm -f backend/celerybeat-schedule.db
bash scripts/dev_up.sh
```

Links return to `active` within ~2 minutes, on the 30s refresh cycle. Verify
before declaring it fixed — do not assume the restart worked.

## Common Mistakes

| Mistake | Reality |
|---|---|
| Reporting on `admin@local.dev` | Wrong tenant, wrong brokers, wrong answer |
| Treating `status: errored` as a dead broker | Usually our own rate-limiting; check `token_valid_today` |
| Trusting strategy output without checking for an `active` link | Synthesised BSM prices look completely normal |
| Verifying a boot by port binds alone | A bound port proves nothing about auth or data |
| Trusting CLAUDE.md on ports/brokers | Doc drift: says Postgres 5436, Redis 6379, Zerodha "stub" — all wrong |

## Evolving This Skill

Append each new operational lesson as a dated line, then fold it into the
sections above once the pattern is clear. Keep the failure *mechanism*, not the
story of the session.

- **2026-09-09** — Boot after 4 weeks idle: stale beat state fired 3 past-due
  crontabs at once → 32 Angel breaker trips in 8 min → all links `errored` →
  silent synthesised-price fallback. Fixed in `dev_up.sh`; Postgres also moved
  5436 → 5444 to clear Hopzy's range.
