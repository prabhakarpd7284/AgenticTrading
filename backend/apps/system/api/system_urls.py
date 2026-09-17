"""V2 URLs for system control: kill switch, AI pause/resume, status snapshot.

Mounted at /api/v1/system/ via config/urls.py (alongside the ops console
URLs at /api/v1/ops/). View functions live in apps.trading.api.legacy_compat_views
for the migration window; they'll move into a proper apps.system.api.views
once the legacy bridge is fully retired.
"""
from django.urls import path

from apps.system.api.pipeline_views import (
    PipelineConfigView, PipelineStatusView, PipelineTriggerView,
)
from apps.trading.api import legacy_compat_views as _compat

urlpatterns = [
    # GET — current kill switch state, market-open flag, etc.
    path("",         _compat.system,     name="system-status"),
    path("pause/",   _compat.pause_ai,   name="system-pause"),
    path("resume/",  _compat.resume_ai,  name="system-resume"),
    # Daily-pipeline debugger — status + manual trigger.
    path("pipeline/",                 PipelineStatusView.as_view(),  name="pipeline-status"),
    path("pipeline/config/",          PipelineConfigView.as_view(),  name="pipeline-config"),
    path("pipeline/<str:task>/run/",  PipelineTriggerView.as_view(), name="pipeline-trigger"),
]
