// ElastiCache Redis 7 — single node in dev, replicated in prod.

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
variable "vpc_id" {
  type = string
}
variable "private_subnet_ids" {
  type = list(string)
}
variable "ecs_sg_id" {
  type = string
}
variable "node_type" {
  type = string
  default = "cache.t4g.small"
}
variable "replicas" {
  type = number
  default = 0
}
locals {
  tags = { Project = "alphadesk", Env = var.env, ManagedBy = "terraform" }
}

resource "aws_security_group" "redis" {
  name        = "${var.name}-redis"
  description = "Redis — only ECS tasks in"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [var.ecs_sg_id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = local.tags
}

resource "aws_elasticache_subnet_group" "this" {
  name       = "${var.name}-redis"
  subnet_ids = var.private_subnet_ids
}

resource "aws_elasticache_replication_group" "this" {
  replication_group_id       = "${var.name}-${var.env}"
  description                = "alphadesk ${var.env}"
  engine                     = "redis"
  engine_version             = "7.1"
  node_type                  = var.node_type
  num_cache_clusters         = var.replicas + 1
  parameter_group_name       = "default.redis7"
  subnet_group_name          = aws_elasticache_subnet_group.this.name
  security_group_ids         = [aws_security_group.redis.id]
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  automatic_failover_enabled = var.replicas > 0
  multi_az_enabled           = var.replicas > 0
  tags                       = local.tags
}

output "primary_endpoint" {
  value = aws_elasticache_replication_group.this.primary_endpoint_address
}
output "redis_sg_id" {
  value = aws_security_group.redis.id
}