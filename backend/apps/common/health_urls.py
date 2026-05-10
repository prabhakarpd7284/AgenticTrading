from django.http import JsonResponse
from django.urls import path


def healthz(_request):
    return JsonResponse({"status": "ok"})


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
    path("ready/", readyz, name="readyz"),
]
