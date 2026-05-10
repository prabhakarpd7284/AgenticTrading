## What
<!-- 1–2 sentences: what does this PR do? -->

## Why
<!-- Link the ticket / problem statement -->

## How (notable design decisions)
<!-- Anything reviewers can't see from the diff alone -->

## Risk + rollout
- [ ] DB migration (forward-only? reversible?)
- [ ] Feature-flagged
- [ ] Touches @RiskGuard / order placement path
- [ ] Touches multi-tenant isolation
- [ ] Schema/contract change (OpenAPI / WS) — version bumped

## Testing
- [ ] `pytest` green
- [ ] `npm run lint && npm run typecheck && npm run build` green
- [ ] Manual sanity in dev

## Checklist
- [ ] No secrets in code
- [ ] Tenant scoping verified for any new model/endpoint
- [ ] Idempotency-Key respected on any new mutating endpoint
- [ ] Telemetry: structured log + audit entry where appropriate
