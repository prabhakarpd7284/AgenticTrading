from django.apps import AppConfig


class LegacyConfig(AppConfig):
    """URL bridge serving /api/v1/legacy/* routes.

    Post redesign-v2 (Phase 6), every view reads from v2 Postgres tables.
    This app has no models; it exists only so the frontend's existing
    legacy URL namespace keeps resolving while we migrate the React side
    to /api/v1/{trades,events,signals,…}/ proper.
    """

    name = "apps.legacy"
    label = "legacy"
