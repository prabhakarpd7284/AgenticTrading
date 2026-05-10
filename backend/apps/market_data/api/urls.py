from django.urls import path
from apps.market_data.api.views import (
    BasketView,
    CandleView,
    MarketPulseView,
    OKBacktestView,
    PyramidView,
    SectorRotationView,
    SetupPreviewView,
    ShortlistView,
    SwingScannerView,
    SymbolSearchView,
)

urlpatterns = [
    path("symbols/",        SymbolSearchView.as_view(),   name="symbol-search"),
    path("candles/",        CandleView.as_view(),         name="candles"),
    path("pulse/",          MarketPulseView.as_view(),    name="market-pulse"),
    path("rotation/",       SectorRotationView.as_view(), name="sector-rotation"),
    path("shortlist/",      ShortlistView.as_view(),      name="shortlist"),
    path("setup/",          SetupPreviewView.as_view(),   name="setup-preview"),
    path("swing-scanner/",  SwingScannerView.as_view(),   name="swing-scanner"),
    path("ok-backtest/",    OKBacktestView.as_view(),     name="ok-backtest"),
    path("basket/",         BasketView.as_view(),         name="basket-status"),
    path("pyramid/",        PyramidView.as_view(),        name="pyramid"),
]
