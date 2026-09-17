// Prod environment root — mirrors dev but with Multi-AZ RDS, Redis replica, bigger tasks.

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.50" }
  }
  backend "s3" {}
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "alphadesk", Env = var.env, ManagedBy = "terraform" }
  }
}

variable "region" {
  type = string
  default = "ap-south-1"
}
variable "env" {
  type = string
  default = "prod"
}
variable "name" {
  type = string
  default = "alphadesk"
}
variable "certificate_arn" {
  type = string
}
variable "repository_url" {
  type = string
}
variable "github_token_arn" {
  type = string
}
variable "db_password" {
  type = string
  sensitive = true
}
variable "backend_image" {
  type = string
}
variable "worker_image" {
  type = string
}
module "network" {
  source = "../../modules/network"
  name = var.name
  env = var.env
}
module "ecr" {
  source = "../../modules/ecr"
  name = var.name
  env = var.env
}
module "secrets" {
  source = "../../modules/secrets"
  name = var.name
  env = var.env
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
  cpu                = 1024
  memory             = 2048
  desired_api        = 3
  desired_ws         = 3
  desired_worker     = 3
  env_vars = {
    DJANGO_SETTINGS_MODULE = "config.settings.prod"
    ALLOWED_HOSTS          = "api.alphadesk.example.com"
    CORS_ALLOWED_ORIGINS   = "https://app.alphadesk.example.com"
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
  instance_class     = "db.t4g.medium"
  multi_az           = true
}

module "redis" {
  source             = "../../modules/redis"
  name               = var.name
  env                = var.env
  vpc_id             = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  ecs_sg_id          = module.ecs.ecs_sg_id
  node_type          = "cache.t4g.medium"
  replicas           = 1
}

module "amplify" {
  source           = "../../modules/amplify"
  name             = var.name
  env              = var.env
  repository_url   = var.repository_url
  branch           = "main"
  github_token_arn = var.github_token_arn
  # Names MUST match what the bundle reads (src/lib/api.ts: VITE_API_URL,
  # src/lib/ws.ts: VITE_WS_URL). Replace the example.com placeholder with the
  # real backend hostname before applying.
  vite_env = {
    VITE_API_URL      = "https://api.alphadesk.example.com"
    VITE_WS_URL       = "wss://api.alphadesk.example.com"
    VITE_MONTHLY_LIVE = "1"
  }
}

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