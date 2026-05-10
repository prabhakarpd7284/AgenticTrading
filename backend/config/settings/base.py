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
environ.Env.read_env(BASE_DIR / ".env")

# Core -----------------------------------------------------------------
SECRET_KEY = env("DJANGO_SECRET_KEY", default="insecure-dev-key-change-me")
DEBUG = env("DEBUG")
ALLOWED_HOSTS = env("ALLOWED_HOSTS")

TRADING_MODE = env("TRADING_MODE", default="paper")  # paper | live | halt

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
    "apps.broker",
    "apps.market_data",
    "apps.portfolio",
    "apps.orders",
    "apps.strategies",
    "apps.agents_core",
    "apps.rag",
    "apps.journals",
    "apps.audit",
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

# Celery ---------------------------------------------------------------
CELERY_BROKER_URL = env("CELERY_BROKER_URL", default=REDIS_URL)
CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", default=REDIS_URL)
CELERY_TASK_ACKS_LATE = True
CELERY_TASK_REJECT_ON_WORKER_LOST = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_BEAT_SCHEDULE = {
    "process-order-outbox": {
        "task": "apps.orders.tasks.outbox.process_outbox",
        "schedule": 1.0,
    },
    "refresh-portfolio-snapshots": {
        "task": "apps.portfolio.tasks.snapshots.refresh_all",
        "schedule": 60.0,
    },
    "expire-old-agent-runs": {
        "task": "apps.agents_core.tasks.housekeeping.expire_runs",
        "schedule": 300.0,
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
    },
}

# AlphaDesk runtime ----------------------------------------------------
ALPHADESK = {
    "DEFAULT_CAPITAL": env.int("DEFAULT_CAPITAL", default=500_000),
    "MAX_RISK_PER_TRADE_PCT": env.float("MAX_RISK_PER_TRADE_PCT", default=1.0),
    "MAX_DAILY_LOSS_PCT": env.float("MAX_DAILY_LOSS_PCT", default=3.0),
    "MAX_POSITION_SIZE_PCT": env.float("MAX_POSITION_SIZE_PCT", default=10.0),
    "LLM_MODEL": env("LLM_MODEL", default="claude-sonnet-4-6"),
    "ANTHROPIC_API_KEY": env("ANTHROPIC_API_KEY", default=""),
    "STRATEGY_REGISTRY_AUTOLOAD": True,
    "BROKER_REGISTRY_AUTOLOAD": True,
    "RAG_REGISTRY_AUTOLOAD": True,
}
