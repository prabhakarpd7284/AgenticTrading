# AlphaDesk — Terraform (Stage 2 infra)

This tree provisions the Stage-2 target architecture described in
`docs/infra/DEPLOYMENT_COMPARISON.md` and `docs/adr/ADR-0005-deployment-target.md`:

    Amplify (frontend) ──┐
                         ├─► ALB ──► ECS Fargate (api + ws + workers + beat)
                         │             │
                         │             ├─► RDS Postgres 16 (pgvector)
                         │             └─► ElastiCache Redis 7
                         └─► Secrets Manager (broker creds, Django secret key)

Stage 1 (single EC2 running docker-compose) does **not** use Terraform — it is
bootstrapped by a cloud-init script and deployed via GitHub Actions over SSH.
See `infra/ec2/` for that path.

## Layout

```
infra/terraform/
├── modules/
│   ├── network/          VPC, public + private subnets, NAT, routes
│   ├── alb/              ALB + TLS listener + target groups (/api, /ws)
│   ├── ecs_service/      Cluster, task def, service, autoscaling
│   ├── rds/              Postgres 16 with pgvector parameter group
│   ├── redis/            ElastiCache Redis 7 replication group
│   ├── amplify/          Amplify app + branch + env vars
│   ├── ecr/              ECR repos for backend + worker images
│   ├── secrets/          Secrets Manager entries
│   └── state_backend/    S3 + DynamoDB for remote state (bootstrap once)
└── envs/
    ├── dev/              dev.tfvars, backend.hcl
    └── prod/             prod.tfvars, backend.hcl
```

## Bootstrap (one-time)

```bash
# 1. Create the remote state backend (run from a workstation with AdminAccess)
cd modules/state_backend
terraform init && terraform apply

# 2. Init an environment against that backend
cd ../../envs/dev
terraform init -backend-config=backend.hcl
terraform plan  -var-file=dev.tfvars
terraform apply -var-file=dev.tfvars
```

## Conventions

- All resources tagged with `Project=alphadesk`, `Env=<dev|prod>`, `ManagedBy=terraform`.
- Secrets are **never** committed. `*.tfvars` in this tree contain only non-secret
  values; secrets are sourced from AWS Secrets Manager at container start.
- RDS and Redis live in private subnets; only the ECS security group can reach them.
- Container images are pulled from ECR — pushed by GitHub Actions on merge to `main`.
