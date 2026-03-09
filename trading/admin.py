from django.contrib import admin
from .models import TradeJournal, StrategyDoc, PortfolioSnapshot, AuditLog


@admin.register(TradeJournal)
class TradeJournalAdmin(admin.ModelAdmin):
    list_display = ["symbol", "side", "quantity", "entry_price", "status", "pnl", "risk_approved", "created_at"]
    list_filter = ["status", "side", "trade_date", "risk_approved"]
    search_fields = ["symbol", "reasoning"]
    readonly_fields = ["created_at", "updated_at"]


@admin.register(StrategyDoc)
class StrategyDocAdmin(admin.ModelAdmin):
    list_display = ["title", "category", "is_active", "updated_at"]
    list_filter = ["category", "is_active"]


@admin.register(PortfolioSnapshot)
class PortfolioSnapshotAdmin(admin.ModelAdmin):
    list_display = ["snapshot_date", "capital", "available_cash", "daily_pnl", "total_pnl", "daily_loss", "open_positions", "last_updated"]
    list_filter = ["snapshot_date"]
    readonly_fields = ["created_at", "last_updated"]


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ["event_type", "symbol", "model_name", "latency_ms", "created_at"]
    list_filter = ["event_type", "model_name"]
    search_fields = ["symbol", "prompt", "response"]
    readonly_fields = ["created_at"]
    raw_id_fields = ["trade_journal"]
