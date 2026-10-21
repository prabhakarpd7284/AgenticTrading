from django.apps import AppConfig


class SystemConfig(AppConfig):
    name = "apps.system"
    verbose_name = "System control"

    def ready(self) -> None:
        # AgentRun / Backtest rows mirror into RunReceipts (UX redesign R1 follow-up).
        from apps.system.services.detail_receipts import connect_signals

        connect_signals()
