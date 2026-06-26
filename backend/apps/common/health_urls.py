from django.http import JsonResponse
from django.urls import path


def healthz(_request):
    return JsonResponse({"status": "ok"})


def livez(_request):
    """Liveness — process is up and serving. No dependency checks (that's
    readiness). The ALB target-group health check hits /healthz/live."""
    return JsonResponse({"status": "alive"})


def readyz(_request):
    from django.db import connection

    try:
        with connection.cursor() as c:
            c.execute("SELECT 1")
        return JsonResponse({"status": "ready"})
    except Exception as e:  # noqa: BLE001
        return JsonResponse({"status": "degraded", "detail": str(e)}, status=503)


urlpatterns = [
    path("", healthz, name="healthz"),
    # Match both /healthz/live and /healthz/live/ — the ALB check uses the
    # slash-less form and doesn't follow APPEND_SLASH redirects.
    path("live", livez, name="livez"),
    path("live/", livez),
    path("ready/", readyz, name="readyz"),
]
