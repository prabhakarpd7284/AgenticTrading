from .base import *  # noqa: F401,F403
from .base import SIMPLE_JWT, env  # explicit re-import so names resolve (not star-hidden)

# Fail closed: prod MUST supply a real secret + host allowlist. With no default,
# environ raises ImproperlyConfigured at boot if either is unset — so the
# insecure dev SECRET_KEY (which signs JWTs AND derives the broker-credential
# Fernet key) and a wildcard ALLOWED_HOSTS can never reach production.
SECRET_KEY = env("DJANGO_SECRET_KEY")
ALLOWED_HOSTS = env.list("ALLOWED_HOSTS")
# A dedicated signing key for JWTs (falls back to SECRET_KEY if unset) so the
# token-signing and credential-encryption keys can be rotated independently.
SIMPLE_JWT = {**SIMPLE_JWT, "SIGNING_KEY": env("JWT_SIGNING_KEY", default=SECRET_KEY)}

DEBUG = False
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
X_FRAME_OPTIONS = "DENY"

# Sentry
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

sentry_sdk.init(
    dsn=env("SENTRY_DSN", default=""),
    integrations=[DjangoIntegration()],
    traces_sample_rate=0.1,
    send_default_pii=False,
    environment=env("SENTRY_ENV", default="prod"),
)
