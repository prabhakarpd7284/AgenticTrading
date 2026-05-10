# ADR-0005: Deployment target

- Status: Accepted
- Date: 2026-04-18

## Context
See `docs/infra/DEPLOYMENT_COMPARISON.md` for the four-way analysis.

## Decision
- **Stage 1 (MVP / first 200 users):** single EC2 + Docker Compose + Nginx, deploy via
  GitHub Actions over SSH. Same compose file as local dev.
- **Stage 2 (paid GA):** AWS Amplify (FE) + ECS Fargate (BE) + RDS + ElastiCache. Deploy
  via GitHub Actions → ECR → ECS rolling deploy. Terraform for IaC.
- **Stage 3 (>50k DAU or enterprise on-prem):** EKS, multi-region, ArgoCD GitOps.

We scaffold infra for **both Stage 1 and Stage 2** in this build.

## Rationale
- Lambda is wrong for our workload (long agent runs, persistent WebSockets).
- EKS is right but premature; the same containers move to EKS in a quarter when needed.
- Amplify+ECS gives us the cleanest split between a CDN-hosted SPA and a long-running
  ASGI backend, without paying the K8s tax.

## Consequences
- Two CI pipelines (SSH-deploy and ECR-deploy). We accept the small duplication for the
  ability to demo on cheap infra and scale on real infra.
- Terraform state in S3 + DynamoDB lock from day one.
- Secrets in AWS Secrets Manager from Stage 2 onward; in `.env` files for Stage 1
  (with `chmod 600` and never in git).
