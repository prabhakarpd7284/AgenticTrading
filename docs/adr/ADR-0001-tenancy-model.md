# ADR-0001: Tenancy Model

- Status: Accepted
- Date: 2026-04-18
- Deciders: Platform eng

## Context
AlphaDesk targets retail individual users, B2B advisors managing many client books, and
prop desks with multiple traders. We need data isolation strong enough for SEBI-grade
audit but cheap enough to scale to 10k+ tenants in v1.

## Decision
**Single database, shared schema, row-level isolation by `tenant_id`.**

- Every tenant-owned model inherits `TenantModel` (adds indexed `tenant` FK).
- Tenant resolved from JWT in middleware → `request.tenant`.
- DRF base ViewSet filters all querysets by `request.tenant`.
- A pytest linter scans `apps/*/api/views.py` for ViewSets that bypass the filter.
- Admin uses an explicit tenant switcher (cookie + audit log entry).

## Alternatives considered
- **Database-per-tenant.** Rejected: ops cost (10k DBs), migration story is brutal,
  cross-tenant analytics impossible.
- **Schema-per-tenant (Postgres schemas).** Rejected: still pays per-schema connection cost,
  and `django-tenants` adds significant complexity. Reconsider at >100k tenants.
- **Separate cluster per tier (retail / B2B).** Deferred — possible at Stage-3 if enterprise
  customers demand it. Same code, different deploy.

## Consequences
- Pros: simple ops, single migration, fast cross-tenant analytics.
- Cons: noisy-neighbor risk on hot tables → mitigated by read replicas + per-tenant rate limit.
- Cons: a query bug could leak data → mitigated by middleware + linter + integration test
  that creates two tenants and verifies isolation.

## Compliance addendum
- B2B advisor tenants get their own KMS key for column-level encryption of client PII.
- Prop-desk tenants can opt into `dedicated=true` flag → routed to a separate Postgres
  cluster via Django DB router. Same code, different connection.
