"""Base Django settings. Shared between dev/prod/test."""
from __future__ import annotations

from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parents[2]

env = environ.Env(
    DEBUG=(bool, False),
    ALLOWED_HOSTS=(list, ["*"]),
    CORS_ALLOWED_ORIGINS=(list, ["http://localhost:5173"]),
)
# Load env files in order of increasing precedence:
#   1. repo-root .env   — shared infra creds (SMARTAPI_*, TELEGRAM_*,
#                          SYMBOL_MASTER_JSON) used by the broker client
#                          + market-data services that need to reach the
#                          Django process.
#   2. backend/.env     — Django-specific config (DATABASE_URL,
#                          DJANGO_SECRET_KEY, REDIS_URL, …) — overrides
#                          anything set by the root file.
# python-environ's read_env() does NOT override existing process env vars
# by default — so if you `export SMARTAPI_KEY=...` in your shell, that wins.
_REPO_ROOT_ENV = BASE_DIR.parent / ".env"
if _REPO_ROOT_ENV.exists():
    environ.Env.read_env(_REPO_ROOT_ENV)
environ.Env.read_env(BASE_DIR / ".env")

# Core -----------------------------------------------------------------
SECRET_KEY = env("DJANGO_SECRET_KEY", default="insecure-dev-key-change-me")
DEBUG = env("DEBUG")
ALLOWED_HOSTS = env("ALLOWED_HOSTS")

TRADING_MODE = env("TRADING_MODE", default="paper")  # paper | live | halt

# The /ws/ops/ developer console runs arbitrary management commands as the
# server user — superuser-only AND off unless explicitly enabled. Defaults to
# DEBUG so it's available in local dev but disabled in prod unless opted in.
OPS_CONSOLE_ENABLED = env.bool("OPS_CONSOLE_ENABLED", default=DEBUG)

# Applications ---------------------------------------------------------
# `daphne` must come BEFORE django.contrib.staticfiles so Django's `runserver`
# delegates WebSocket upgrades to Daphne (ASGI) while keeping HTTP on WSGI.
# Without this, `ws://localhost:8000/ws/pnl/` and /ws/alerts/ fail to connect.
DJANGO_APPS = [
    "daphne",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]
THIRD_PARTY_APPS = [
    "rest_framework",
    "rest_framework_simplejwt",
    "drf_spectacular",
    "channels",
    "corsheaders",
]
LOCAL_APPS = [
    "apps.common",
    "apps.accounts",
    "apps.tenants",
    "apps.billing",
    "apps.market_data",  # absorbed the broker app in Phase 4b (BrokerLink + adapters)
    "apps.trading",      # Phase 4c: portfolio + orders + trades merged into one
    "apps.strategies",
    "apps.agents_core",
    "apps.rag",
    "apps.events",       # unified event log — absorbed journals + audit in Phase 4a
    "apps.system",       # SystemControl + TraderNote
    "apps.notifications",
]
INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

# Middleware -----------------------------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "apps.common.middleware.RequestIdMiddleware",
    "apps.common.middleware.TenantMiddleware",
    "apps.common.middleware.StructlogContextMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

ASGI_APPLICATION = "config.asgi.application"
WSGI_APPLICATION = "config.wsgi.application"

# Database -------------------------------------------------------------
DATABASES = {
    "default": env.db_url("DATABASE_URL", default="sqlite:///db.sqlite3"),
}
# Reuse connections across requests/tasks instead of connect+close every time
# (the DB-churn the audit flagged). 0 in tests keeps each test isolated.
DATABASES["default"]["CONN_MAX_AGE"] = env.int("DB_CONN_MAX_AGE", default=60)
if str(DATABASES["default"].get("ENGINE", "")).endswith("postgresql"):
    # Require TLS to Postgres in prod (RDS enforces rds.force_ssl=1). Local dev
    # PG has no certs, so default to 'prefer'; set DB_SSLMODE=require in prod.
    DATABASES["default"].setdefault("OPTIONS", {})["sslmode"] = env(
        "DB_SSLMODE", default="prefer"
    )
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Auth -----------------------------------------------------------------
AUTH_USER_MODEL = "accounts.User"
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.Argon2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
]

# DRF ------------------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": (
        "rest_framework.permissions.IsAuthenticated",
        "apps.common.permissions.TenantScoped",
    ),
    "DEFAULT_PAGINATION_CLASS": "apps.common.pagination.CursorPagination",
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "EXCEPTION_HANDLER": "apps.common.exceptions.problem_json_handler",
    "DEFAULT_THROTTLE_CLASSES": (
        "rest_framework.throttling.UserRateThrottle",
        "rest_framework.throttling.AnonRateThrottle",
    ),
    "DEFAULT_THROTTLE_RATES": {
        "user": "1000/minute",
        "anon": "60/minute",
    },
}

SPECTACULAR_SETTINGS = {
    "TITLE": "AlphaDesk API",
    "DESCRIPTION": "AlphaDesk — AI-assisted trading desk",
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME_MINUTES": 15,
    "REFRESH_TOKEN_LIFETIME_DAYS": 14,
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
}

# Channels -------------------------------------------------------------
REDIS_URL = env("REDIS_URL", default="redis://localhost:6379/0")
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {"hosts": [REDIS_URL]},
    },
}

# Cache ----------------------------------------------------------------
# A SHARED cache is a correctness requirement, not an optimisation. The default
# LocMemCache is per-process, so across web + worker + beat:
#   * DRF throttle counters multiply by worker count (~24× the configured rate);
#   * RiskEngine's regime gate reads `market:pulse:v1` written by another
#     process — an invisible miss makes the gate silently no-op;
#   * `ltp:` / pulse keys are never shared and vanish on restart.
# Django 5's built-in RedisCache uses redis-py (already a dependency). Tests
# override this with LocMemCache so they need no live Redis.
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": env("CACHE_URL", default=REDIS_URL),
    },
}

# Celery ---------------------------------------------------------------
from celery.schedules import crontab  # noqa: E402

CELERY_BROKER_URL = env("CELERY_BROKER_URL", default=REDIS_URL)
CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", default=REDIS_URL)
# High-frequency periodic tasks store a celery-task-meta-* result blob nobody
# reads; cap their lifetime (was an implicit 24h) so Redis doesn't accumulate
# them. Per-task ignore_result=True on the fire-and-forget periodics suppresses
# the write entirely; this TTL is the safety net for the rest.
CELERY_RESULT_EXPIRES = 3600
CELERY_TASK_ACKS_LATE = True
CELERY_TASK_REJECT_ON_WORKER_LOST = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
# Daily-pipeline beat entries use crontab() with IST clock times — set the
# Celery timezone so 09:15 means 09:15 IST, not UTC. Interval-based entries
# (the float schedules below) are timezone-agnostic, so this is safe.
CELERY_TIMEZONE = "Asia/Kolkata"
# Every entry carries options.expires >= its cadence so a wake-up produced
# while the worker is DOWN self-discards instead of piling into a durable
# queue. (On 2026-06-24 a stopped worker let process_outbox accumulate 13,669
# stale messages over ~11h.) A 1s poll is worthless 10s later, so its TTL is
# tight; longer cadences get >= one interval.
CELERY_BEAT_SCHEDULE = {
    "process-order-outbox": {
        "task": "apps.trading.tasks.outbox.process_outbox",
        "schedule": 1.0,
        "options": {"expires": 10},
    },
    "refresh-portfolio-snapshots": {
        "task": "apps.trading.tasks.snapshots.refresh_all",
        "schedule": 60.0,
        "options": {"expires": 120},
    },
    "expire-old-agent-runs": {
        "task": "apps.agents_core.tasks.housekeeping.expire_runs",
        "schedule": 300.0,
        "options": {"expires": 600},
    },
    # Broker snapshot refresh — fan-out task picks every ACTIVE BrokerLink
    # and dispatches per-link refreshes. The task itself derives the next
    # cadence (30s market hours / 5min off-hours), but beat fires at the
    # tightest interval; off-hours runs cheaply detect no work to do.
    "refresh-broker-positions": {
        "task": "apps.market_data.tasks.broker_refresh.refresh_broker_positions",
        "schedule": 30.0,
        "options": {"expires": 60},
    },
    "prune-broker-snapshots": {
        "task": "apps.market_data.tasks.broker_refresh.prune_old_snapshots",
        "schedule": 3600.0 * 6,  # every 6h
        "options": {"expires": 3600},
    },
    # Auto-kind watchlists (SIGNAL_RANK, SOURCE_HOT, RECENT_ACTIVE, …)
    # re-resolve every 5 min. Task is cheap — pure DB aggregates with no
    # broker hits — so the cadence is mostly about operator-visible
    # freshness. /refresh/ endpoint provides on-demand sync re-resolve.
    "refresh-auto-watchlists": {
        "task": "apps.notifications.tasks.watchlists.refresh_auto_watchlists",
        "schedule": 300.0,
        "options": {"expires": 600},
    },
    # ── Daily trading pipeline ────────────────────────────────────────
    # Three crontab tasks (IST) make AlphaDesk produce data on its own
    # every trading day. Each task self-skips on weekends / NSE holidays
    # via apps.market_data.services.market_calendar.is_trading_day, so
    # firing Mon–Fri is enough.
    #
    # Premarket Oliver Kell swing scan — 08:20 IST.
    "daily-swing-scan": {
        "task": "apps.strategies.tasks.daily_pipeline.run_swing_scan",
        "schedule": crontab(hour=8, minute=20, day_of_week="mon-fri"),
    },
    # Premarket morning basket — mood + signal scan — 08:45 IST.
    "daily-premarket-basket": {
        "task": "apps.strategies.tasks.daily_pipeline.run_premarket_basket",
        "schedule": crontab(hour=8, minute=45, day_of_week="mon-fri"),
    },
    # Intraday live screener — starts at market open, runs to 15:30 IST.
    "daily-screener-session": {
        "task": "apps.strategies.tasks.daily_pipeline.run_screener_session",
        "schedule": crontab(hour=9, minute=15, day_of_week="mon-fri"),
    },
    # EOD signal enrichment — 16:00 IST, after candles settle.
    "daily-eod-enrichment": {
        "task": "apps.strategies.tasks.daily_pipeline.run_eod_enrichment",
        "schedule": crontab(hour=16, minute=0, day_of_week="mon-fri"),
    },
    # EOD trade derivation — 16:30 IST, replays the just-closed session into
    # paper trades. No-op unless the operator opts in (auto_execute_enabled),
    # so the pipeline stays scan-only by default.
    "daily-derive-trades": {
        "task": "apps.strategies.tasks.daily_pipeline.derive_intraday_trades",
        "schedule": crontab(hour=16, minute=30, day_of_week="mon-fri"),
    },
}

# CORS/CSRF ------------------------------------------------------------
CORS_ALLOWED_ORIGINS = env("CORS_ALLOWED_ORIGINS")
CSRF_TRUSTED_ORIGINS = env.list("CSRF_TRUSTED_ORIGINS", default=[])

# Internationalization -------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# Logging --------------------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "apps.common.logging.JSONFormatter",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        }
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "django.request": {"level": "WARNING", "propagate": True},
        "apps": {"level": "INFO", "propagate": True},
        # Angel SmartApi SDK stdlib loggers — keep at ERROR so verbose request
        # dumps stay quiet. (The plaintext-credential leak comes from the SDK's
        # SEPARATE logzero logger, neutralised by CredentialRedactionFilter in
        # apps.common.logging, installed via the apps.common AppConfig.ready.)
        "SmartApi": {"level": "ERROR", "propagate": True},
        "SmartApi.smartConnect": {"level": "ERROR", "propagate": True},
    },
}

# AlphaDesk runtime ----------------------------------------------------
ALPHADESK = {
    "DEFAULT_CAPITAL": env.int("DEFAULT_CAPITAL", default=500_000),
    "MAX_RISK_PER_TRADE_PCT": env.float("MAX_RISK_PER_TRADE_PCT", default=1.0),
    "MAX_DAILY_LOSS_PCT": env.float("MAX_DAILY_LOSS_PCT", default=3.0),
    "MAX_POSITION_SIZE_PCT": env.float("MAX_POSITION_SIZE_PCT", default=10.0),
    # Risk-gate thresholds the RiskEngine reads from here. Without these keys the
    # engine's .get(..., default) fell back to hardcoded values and silently
    # ignored the documented env overrides.
    "MIN_RISK_REWARD_RATIO": env.float("MIN_RISK_REWARD_RATIO", default=1.5),
    "MIN_CONFIDENCE": env.float("MIN_CONFIDENCE", default=0.55),
    "MAX_OPEN_POSITIONS": env.int("MAX_OPEN_POSITIONS", default=3),
    # Absolute order backstops — NOT normal sizing limits (those are the % of
    # capital above). These bound a malformed or hostile order regardless of
    # the client-supplied price, so a tiny-price + huge-qty MARKET order can't
    # slip under the % caps and then fill large.
    "MAX_ORDER_QTY": env.int("MAX_ORDER_QTY", default=100_000),
    "MAX_ORDER_NOTIONAL": env.float("MAX_ORDER_NOTIONAL", default=10_000_000.0),
    "LLM_MODEL": env("LLM_MODEL", default="claude-sonnet-4-6"),
    "ANTHROPIC_API_KEY": env("ANTHROPIC_API_KEY", default=""),
    "STRATEGY_REGISTRY_AUTOLOAD": True,
    "BROKER_REGISTRY_AUTOLOAD": True,
    "RAG_REGISTRY_AUTOLOAD": True,
    # Explicit strategy registrations applied AFTER entry-point auto-load.
    # Use this when a new plugin has been added to backend/plugins/ but
    # `uv pip install -e .` hasn't been re-run in the dev venv yet — the
    # entry-point isn't picked up by importlib.metadata until reinstall.
    # Format: "import.path:ClassName"  (one or many)
    "STRATEGY_REGISTRY_EXTRA": [
        "plugins.strategy_vertical_spread.strategy:VerticalSpreadStrategy",
    ],
}
