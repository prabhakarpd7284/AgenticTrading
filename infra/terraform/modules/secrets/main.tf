// Secrets Manager entries. Values are seeded OUTSIDE terraform (console / CI step);
// we only declare the containers so the ECS task can reference them by ARN.

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.50" }
  }
}

variable "name" {
  type = string
}
variable "env" {
  type = string
}
locals {
  tags   = { Project = "alphadesk", Env = var.env, ManagedBy = "terraform" }
  names  = [
    "django_secret_key",
    "database_url",
    "redis_url",
    "anthropic_api_key",
    "smartapi_key",
    "smartapi_username",
    "smartapi_password",
    "smartapi_totp_secret",
  ]
}

resource "aws_secretsmanager_secret" "this" {
  for_each                = toset(local.names)
  name                    = "${var.name}/${var.env}/${each.key}"
  recovery_window_in_days = 7
  tags                    = local.tags
}

output "secret_arns" {
  value = { for k, s in aws_secretsmanager_secret.this : k => s.arn }
}
