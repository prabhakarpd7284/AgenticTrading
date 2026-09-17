# Deployment Options — Pros, Cons & Recommendation

This compares four realistic targets for the AlphaDesk stack:
**Django REST + Channels (WebSockets) + Celery workers + Postgres + Redis + React/Vite frontend.**

The workload has three traits that matter for the decision:
1. **Long-running LangGraph agent runs** (5–60 s) that stream tokens.
2. **Persistent WebSockets** (live ticks + agent stream).
3. **Background workers** (Celery beat schedules, broker WS ingest, order outbox).

---

## Option A — Amplify (FE) + ECS Fargate + RDS (BE) ✅ **Recommended for v1**

**Shape:** Frontend on AWS Amplify (CI-built static + global CDN). Backend in containers on
ECS Fargate behind ALB. RDS Postgres, ElastiCache Redis, Secrets Manager, S3, CloudWatch.

| Dimension | Verdict |
|-----------|---------|
| Fits LangGraph long runs | **Yes** — long-lived ASGI workers, no 15-min timeouts. |
| Fits WebSockets | **Yes** — ALB supports WS; tasks are persistent. |
| Ops effort | Moderate — managed control plane, no nodes to patch. |
| Scaling | Horizontal autoscaling on CPU/memory or custom metric. |
| Cost (steady, ~10k DAU) | ~$700–1,200 / mo prod (2× tasks REST, 2× WS, 2× workers, db.r6g.large, cache.t4g.medium). |
| Time to first deploy | 1–2 weeks. |
| Multi-AZ HA | First-class. |
| Future K8s migration | Easy — same Dockerfiles. |

**Pros**
- Identical local dev (Docker Compose) and prod (containers).
- ALB → target groups split nicely: `/api/*` → REST tasks, `/ws/*` → Channels tasks.
- Amplify gives PR previews and a beautiful frontend deploy story (custom domain, SSL, atomic deploys).
- Secrets Manager + IAM task roles → no secrets in env files.
- Easy to add a second region behind Route 53 latency routing.

**Cons**
- Fargate cold-start on scale-up is slow (~30–60s); use min-tasks > 0.
- WS tasks need sticky sessions or use Redis as the channel layer (we do).
- Slightly more expensive than EC2 at low load, but you're paying for not running EC2.

---

## Option B — Amplify (FE) + API Gateway + Lambda (BE)

**Shape:** Frontend on Amplify. Backend = a fleet of Lambdas behind API Gateway (HTTP API for REST, WebSocket API for sockets). DynamoDB or RDS Proxy → RDS.

| Dimension | Verdict |
|-----------|---------|
| Fits LangGraph long runs | **No, painful.** Lambda timeout 15 min ceiling. Async streaming via Lambda is awkward (Response Streaming exists but limited). Need to refactor every long graph into Step Functions. |
| Fits WebSockets | Possible via API Gateway WebSocket API — but each message is a separate Lambda invocation; you lose in-process state and Channels groups. |
| Ops effort | Very low for compute, very high for refactor — Django doesn't fit Lambda well. |
| Scaling | Infinite, automatic. |
| Cost (steady, ~10k DAU) | Cheap at low traffic; gets expensive when sustained agent-run concurrency rises (per-invocation + duration billing). |
| Time to first deploy | 4–6 weeks (significant rewrite). |

**Pros**
- Cheapest at very low traffic. Scales to zero.
- Zero server management.
- API Gateway adds throttling, caching, JWT authorizers for free.

**Cons**
- **Django on Lambda (Mangum/Zappa) is a known anti-pattern** for our shape — we'd want to rewrite as FastAPI or split into many micro-Lambdas.
- LangGraph nodes that take 30s and stream tokens fight the Lambda model.
- Channels can't really run on Lambda — we'd build our own WS state machine on DynamoDB. That's months.
- Cold starts hurt agent latency.

**When this would win:** if AlphaDesk were a stateless pricing/quote API with bursty traffic and no agentic streaming. Not us.

---

## Option C — Single EC2 (FE+BE) via GitHub Actions over SSH

**Shape:** One `t3.large` (or `c6i.xlarge`) EC2. Nginx reverse-proxy. Docker Compose runs Postgres, Redis, backend, worker, beat, frontend container. GHA SSHes in and runs `docker compose pull && up -d`.

| Dimension | Verdict |
|-----------|---------|
| Fits LangGraph long runs | Yes — same container shape. |
| Fits WebSockets | Yes — Nginx supports `proxy_pass` upgrade. |
| Ops effort | Low until something breaks; then high (SSH, single point of failure). |
| Scaling | Vertical only; downtime to resize. |
| Cost | ~$70–150 / mo for one box. **Cheapest by far.** |
| Time to first deploy | < 3 days. |
| Multi-AZ HA | None. |

**Pros**
- Fastest path to a live demo URL.
- All logs in one place.
- Zero cloud-account complexity. Great for **MVP / investor demo / first 100 users**.

**Cons**
- Single AZ; one instance reboot = downtime.
- DB backups are your problem (snapshots + pg_dump cron).
- No autoscaling — a viral moment kills you.
- No PR previews.
- Secrets sit on disk.

**Use it for:** dev/staging always; production until paying customers warrant Option A.

---

## Option D — Kubernetes (EKS)

**Shape:** EKS cluster (Karpenter for nodes), Helm charts, ingress-nginx, cert-manager,
ArgoCD, RDS, ElastiCache. Same containers as Option A but orchestrated by Kubernetes.

| Dimension | Verdict |
|-----------|---------|
| Fits LangGraph + WS | Yes. |
| Ops effort | High — needs a Kubernetes-fluent person. |
| Cost | EKS control plane $73/mo + nodes. Worth it past ~50 services. |
| Time to first deploy | 4–8 weeks for prod-grade. |

**Pros**
- Industry-standard, portable across clouds.
- Best fine-grained autoscaling story (HPA, VPA, KEDA on Redis queue depth).
- Ecosystem (Istio, OpenTelemetry operators, etc.).

**Cons**
- Overkill for v1.
- High operational tax — patching, IAM-Roles-for-Service-Accounts, networking gotchas.
- Slows shipping at the early-stage when shipping > scale.

**Use it for:** v3, when we're at 100k MAU or need on-prem deploys for prop desks.

---

## Recommendation: stage-gated path

```
 Stage 0   →   Stage 1   →   Stage 2   →   Stage 3
 (today)       (MVP/beta)    (paid GA)    (scale-out)

 Local         Single EC2    Amplify+ECS   Multi-region
 Compose       (Option C)    (Option A)    EKS (Option D)
```

| Stage | Trigger to advance | Effort |
|-------|--------------------|--------|
| 0 → 1 | First external user | 2 days (this repo will ship this) |
| 1 → 2 | First paying tenant OR > 200 DAU | 1–2 weeks |
| 2 → 3 | > 50k DAU OR enterprise deal requiring on-prem | 1 quarter |

**We will scaffold infra for both Stage 1 and Stage 2 in this build:**
- `docker-compose.yml` — runs full stack locally; also the Stage-1 production layout.
- `infra/terraform/` — Stage-2 ECS+RDS+ElastiCache+Amplify modules, parameterized by env.
- `.github/workflows/` — CI for both: SSH-deploy for Stage 1, ECR+ECS deploy for Stage 2.

**Why not Lambda?** Our agent runs and WebSockets fight the FaaS model. We'd spend 6 weeks
fighting the platform to save $200/mo. Wrong trade.

**Why not EKS now?** A two-person team should not be patching Kubernetes nodes when we
haven't found product-market fit. Container-shape is K8s-ready when we get there.

---

## Cost ballpark (production, monthly)

| Item | Stage 1 (EC2) | Stage 2 (ECS) | Stage 3 (EKS) |
|------|--------------|----------------|---------------|
| Compute | $80 | $400 | $900 |
| DB | self-hosted | $250 (RDS r6g.large) | $500 (multi-AZ) |
| Cache | self-hosted | $60 | $150 |
| Frontend hosting | Nginx | $20 (Amplify) | $20 |
| Secrets/KMS | $0 | $5 | $10 |
| Logs/metrics | $20 | $80 | $250 |
| ALB / API GW | n/a | $25 | $25 |
| **Total** | **~$120** | **~$840** | **~$1,855** |

(Indicative, ap-south-1, excluding LLM and broker fees.)
