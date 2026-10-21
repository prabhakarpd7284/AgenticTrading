"""/api/v1/mutual-funds/ — see apps.mutual_funds.api.views for the contract."""
from django.urls import path

from apps.mutual_funds.api import views

urlpatterns = [
    path("holdings/", views.HoldingsView.as_view(), name="mf-holdings"),
    path("schemes/", views.SchemeSearchView.as_view(), name="mf-scheme-search"),
    path("schemes/<str:scheme_code>/", views.SchemeDetailView.as_view(), name="mf-scheme-detail"),
    path("schemes/<str:scheme_code>/metrics/", views.SchemeMetricsView.as_view(), name="mf-scheme-metrics"),
    path("schemes/<str:scheme_code>/nav-history/", views.SchemeNavHistoryView.as_view(), name="mf-scheme-nav-history"),
    path("screener/", views.ScreenerView.as_view(), name="mf-screener"),
    path("advisor/findings/", views.AdvisorFindingsView.as_view(), name="mf-advisor-findings"),
    path("advisor/run/", views.AdvisorRunView.as_view(), name="mf-advisor-run"),
    path("status/", views.StatusView.as_view(), name="mf-status"),
]
