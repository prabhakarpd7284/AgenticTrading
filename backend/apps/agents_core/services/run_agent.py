"""Orchestrates: load strategy from registry -> build context -> run graph -> stream events."""
from __future__ import annotations

import asyncio
from uuid import UUID

import structlog
from channels.layers import get_channel_layer
from django.utils import timezone

from apps.agents_core.domain.contracts import AgentContext, AgentEvent
from apps.agents_core.models import AgentRun, AgentStep
from apps.agents_core.registry import strategy_registry

log = structlog.get_logger()


class ChannelsPublisher:
    def __init__(self, run_id: UUID, tenant_id: UUID):
        self.run_id = run_id
        self.tenant_id = tenant_id
        self._seq = 0
        self._layer = get_channel_layer()

    def emit(self, event: AgentEvent) -> None:
        # Persist and publish. Never block on DB failure.
        try:
            AgentStep.objects.create(
                run_id=self.run_id,
                seq=event.seq,
                node=event.node,
                event_type=event.type,
                payload=event.payload,
            )
        except Exception:  # noqa: BLE001
            log.exception("agentstep.save_failed", run_id=str(self.run_id))

        # Stamp the wire payload with a server-side timestamp so the UI can
        # render per-event clock times + inter-step latencies without trusting
        # the operator's local clock. We attach it at emit time (not on the
        # AgentEvent contract) so plugin nodes don't have to construct it —
        # AgentStep.created_at remains the canonical persisted timestamp.
        payload = event.model_dump()
        payload["ts"] = timezone.now().isoformat()

        group = f"agent.{self.tenant_id}.{self.run_id}"
        try:
            asyncio.run(
                self._layer.group_send(group, {"type": "agent.event", "event": payload})
            )
        except Exception:  # noqa: BLE001
            log.exception("channel.group_send_failed", run_id=str(self.run_id))

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq


def build_context(run: AgentRun, publisher: ChannelsPublisher) -> AgentContext:
    from apps.market_data.services.data_port import DefaultMarketData
    from apps.trading.services.risk_engine import RiskEngine
    from apps.rag.services.router import DefaultRAGRouter
    from apps.events.services.journal_port import JournalAdapter

    # Canonical 10-criterion risk engine; wrapped in an adapter so plugin
    # nodes can keep calling `ctx.risk.validate(draft_dict)`.
    risk_port = RiskEngine().as_risk_port(portfolio_id=run.portfolio_id)

    return AgentContext(
        run_id=run.id,
        tenant_id=run.tenant_id,
        user_id=run.triggered_by_id,
        portfolio_id=run.portfolio_id,
        market_data=DefaultMarketData(run.tenant_id),
        rag=DefaultRAGRouter(run.tenant_id),
        risk=risk_port,
        journal=JournalAdapter(run.tenant_id),
        publisher=publisher,
        config=run.config or {},
    )


def run_agent(run: AgentRun) -> None:
    """Synchronous entry — called from the Celery task."""
    publisher = ChannelsPublisher(run.id, run.tenant_id)
    try:
        run.status = AgentRun.Status.RUNNING
        run.started_at = timezone.now()
        run.save(update_fields=["status", "started_at"])

        strategy = strategy_registry.get(run.strategy_name)
        ctx = build_context(run, publisher)
        publisher.emit(AgentEvent(seq=publisher.next_seq(), node="init",
                                  type="info", payload={"strategy": run.strategy_name}))

        graph = strategy.build_graph(ctx)
        state = asyncio.run(graph.ainvoke({"config": run.config}))

        run.result = state
        run.status = AgentRun.Status.SUCCEEDED
    except Exception as e:  # noqa: BLE001
        run.status = AgentRun.Status.FAILED
        run.error = str(e)
        publisher.emit(AgentEvent(seq=publisher.next_seq(), node="error",
                                  type="error", payload={"detail": str(e)}))
        log.exception("agent_run.failed", run_id=str(run.id))
    finally:
        run.completed_at = timezone.now()
        run.save()
