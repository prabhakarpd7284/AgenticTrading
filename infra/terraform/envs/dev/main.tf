// Dev environment root module.
// Calls network → ecr → secrets → alb → rds → redis → ecs_service → amplify.

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.50" }
  }
  backend "s3" {}   // filled in via backend.hcl
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "alphadesk", Env = var.env, ManagedBy = "terraform" }
  }
}

# ---------- variables ----------
variable "region" {
  type = string
  default = "ap-south-1"
}
variable "env" {
  type = string
  default = "dev"
}
variable "name" {
  type = string
  default = "alphadesk"
}
variable "certificate_arn" {
  type = string
  default = ""
}
variable "repository_url"   { type = string }          // https://github.com/<org>/<repo>
variable "github_token_arn" {
  type = string
}
variable "db_password" {
  type = string
  sensitive = true
}
variable "backend_image"    { type = string }          // ECR URI:tag (seeded after first push)
variable "worker_image" {
  type = string
}
# ---------- layers ----------
module "network" {
  source = "../../modules/network"
  name   = var.name
  env    = var.env
}

module "ecr" {
  source = "../../modules/ecr"
  name   = var.name
  env    = var.env
}

module "secrets" {
  source = "../../modules/secrets"
  name   = var.name
  env    = var.env
}

module "alb" {
  source            = "../../modules/alb"
  name              = var.name
  env               = var.env
  vpc_id            = module.network.vpc_id
  public_subnet_ids = module.network.public_subnet_ids
  certificate_arn   = var.certificate_arn
}

module "ecs" {
  source             = "../../modules/ecs_service"
  name               = var.name
  env                = var.env
  vpc_id             = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  alb_sg_id          = module.alb.alb_sg_id
  tg_api_arn         = module.alb.tg_api_arn
  tg_ws_arn          = module.alb.tg_ws_arn
  backend_image      = var.backend_image
  worker_image       = var.worker_image
  env_vars = {
    DJANGO_SETTINGS_MODULE = "config.settings.prod"
    ALLOWED_HOSTS          = "*"
    CORS_ALLOWED_ORIGINS   = "https://${var.name}.example.com"
  }
  secret_arns = module.secrets.secret_arns
}

module "rds" {
  source             = "../../modules/rds"
  name               = var.name
  env                = var.env
  vpc_id             = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  ecs_sg_id          = module.ecs.ecs_sg_id
  password           = var.db_password
  multi_az           = false
}

module "redis" {
  source             = "../../modules/redis"
  name               = var.name
  env                = var.env
  vpc_id             = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  ecs_sg_id          = module.ecs.ecs_sg_id
  replicas           = 0
}

module "amplify" {
  source           = "../../modules/amplify"
  name             = var.name
  env              = var.env
  repository_url   = var.repository_url
  branch           = "develop"
  github_token_arn = var.github_token_arn
  # Names MUST match what the bundle reads (src/lib/api.ts: VITE_API_URL,
  # src/lib/ws.ts: VITE_WS_URL). The old *_BASE_URL names were ignored, so the
  # build fell back to a relative /api and window.origin → broken in prod.
  vite_env = {
    VITE_API_URL     = "https://${module.alb.alb_dns_name}"
    VITE_WS_URL      = "wss://${module.alb.alb_dns_name}"
    VITE_MONTHLY_LIVE = "1"
  }
}

# ---------- outputs ----------
output "alb_dns" {
  value = module.alb.alb_dns_name
}
output "rds_endpoint" {
  value = module.rds.endpoint
  sensitive = true
}
output "redis_endpoint" {
  value = module.redis.primary_endpoint
  sensitive = true
}
output "ecr_repos" {
  value = module.ecr.repository_urls
}
output "amplify_domain" {
  value = module.amplify.default_domain
}