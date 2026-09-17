"""Who may run what in the ops console — one policy, two surfaces.

The console is really two different things wearing one UI:

* the **arbitrary** ``manage.py`` runner on /ops. That is remote code
  execution as the server OS user, so it stays platform-admin only
  (``is_superuser``) — never tenant "owner", because every self-service
  signup owns their personal tenant.
* the **product** commands that trader-facing pages embed via <OpButton>
  (Setup's "plan a trade for X", Monthly's "refresh signal outcomes",
  Swing Scanner's "re-scan now", …). Those are ordinary product
  features, not an admin backdoor: an active tenant owner may run them,
  and only them.

Both the DRF views and the WebSocket consumer gate on the helpers here so
the two surfaces can never disagree. They used to: REST was owner-gated
while the socket was superuser-gated, so an owner saw the console, hit
Run, and got a handshake rejection the browser reports as a generic
close-1006 ("websocket error") with no explanation.

Note the product commands still run as the *server's* OS user against the
server's broker config — they are not tenant-scoped subprocesses. Any
deployment where tenants are mutually untrusting should set
``OPS_CONSOLE_ENABLED=false`` (the default whenever DEBUG is off), which
turns every tier off.
"""
from __future__ import annotations

from django.conf import settings

# Access tiers, most-privileged first.
TIER_ADMIN = "admin"      # platform admin — any visible command
TIER_PRODUCT = "product"  # tenant owner — only PRODUCT_COMMANDS

# Commands the product itself embeds in trader-facing pages. Everything
# here is a trading/analysis command whose blast radius is market data,
# signals and paper/live orders — i.e. what the owner already controls
# through the UI. Deliberately excluded: `backfill_personal_tenants`
# (cross-tenant data mutation) and `stockedge_login` (captures a
# third-party credential), which stay admin-only.
PRODUCT_COMMANDS: frozenset[str] = frozenset({
    # equity / directional
    "run_trading_agent",
    "run_trading_day",
    "run_intraday",
    "run_screener",
    # options
    "manage_straddle",
    "run_straddle_monitor",
    "run_pyramid",
    "run_scalp",
    # swing (Oliver Kell)
    "run_ok_scanner",
    "run_ok_scanner_v2",
    "run_ok_backtest",
    "run_ok_intraday_backtest",
    "enrich_swing_signals",
    "derive_swing_trades",
    # backtests + feedback loop
    "run_backtest",
    "run_backtest_intraday",
    "enrich_signals",
    "enrich_signals_v2",
    "derive_trades",
    # premarket + external data overlays
    "run_morning_basket",
    "pull_stockedge_breadth",
    "pull_stockedge_csv",
    "stockedge_momentum",
    "stockedge_momentum_signals",
})


def console_enabled() -> bool:
    """The master switch. Off in production unless explicitly enabled."""
    return bool(getattr(settings, "OPS_CONSOLE_ENABLED", settings.DEBUG))


def is_ops_admin(user) -> bool:
    """Platform admin — may run any *visible* command (see ops_runner._HIDDEN)."""
    return (
        user is not None
        and not getattr(user, "is_anonymous", True)
        and getattr(user, "is_superuser", False)
    )


def is_tenant_owner(user, tenant) -> bool:
    """Active owner membership on the request/socket's tenant. Hits the DB."""
    if user is None or getattr(user, "is_anonymous", True) or tenant is None:
        return False
    memberships = getattr(user, "memberships", None)
    if memberships is None:
        return False
    return memberships.filter(tenant=tenant, role="owner", is_active=True).exists()


def ops_tier(user, tenant) -> str | None:
    """Return TIER_ADMIN / TIER_PRODUCT, or None when the console is closed."""
    if not console_enabled():
        return None
    if is_ops_admin(user):
        return TIER_ADMIN
    if is_tenant_owner(user, tenant):
        return TIER_PRODUCT
    return None


def can_run(tier: str | None, command: str) -> bool:
    """Is `command` runnable at this tier? Assumes the name is already validated."""
    if tier == TIER_ADMIN:
        return True
    if tier == TIER_PRODUCT:
        return command in PRODUCT_COMMANDS
    return False
