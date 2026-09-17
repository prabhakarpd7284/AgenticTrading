"""ASGI entry — REST + WebSocket."""
import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")
django.setup()

from channels.routing import ProtocolTypeRouter, URLRouter  # noqa: E402
from channels.security.websocket import AllowedHostsOriginValidator  # noqa: E402
from django.core.asgi import get_asgi_application  # noqa: E402

from apps.common.ws_auth import JWTAuthMiddlewareStack  # noqa: E402
from config.channels_router import websocket_urlpatterns  # noqa: E402

# WebSocket pipeline:
#   Origin check → JWT-from-subprotocol/query → URL routing → consumer
# Browsers can't attach an `Authorization` header to a WS upgrade, so we
# expect the SPA to negotiate two subprotocols: ["jwt", "<access_token>"].
application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": AllowedHostsOriginValidator(
            JWTAuthMiddlewareStack(URLRouter(websocket_urlpatterns))
        ),
    }
)
