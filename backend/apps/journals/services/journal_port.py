from __future__ import annotations


class JournalAdapter:
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id

    def record(self, entry: dict) -> None:
        from apps.journals.models import JournalEntry
        JournalEntry.objects.create(tenant_id=self.tenant_id, **entry)
