"""/api/v1/nightly-scan/ — see apps.news.api.views."""
from django.urls import path

from apps.news.api import views

urlpatterns = [
    path("", views.ScanListView.as_view(), name="nightly-scan-list"),
    path("latest/", views.LatestScanView.as_view(), name="nightly-scan-latest"),
    path("<str:for_date>/", views.ScanByDateView.as_view(), name="nightly-scan-date"),
    path("<str:for_date>/<str:symbol>/", views.ScanRowView.as_view(), name="nightly-scan-row"),
]
