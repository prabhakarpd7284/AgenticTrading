// RDS Postgres 16 with pgvector extension. Single-AZ in dev, Multi-AZ in prod.

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
variable "instance_class" {
  type = string
  default = "db.t4g.small"
}
variable "multi_az" {
  type = bool
  default = false
}
variable "username" {
  type = string
  default = "alphadesk"
}
variable "password" {
  type = string
  sensitive = true
}
locals {
  tags = { Project = "alphadesk", Env = var.env, ManagedBy = "terraform" }
}

resource "aws_security_group" "db" {
  name        = "${var.name}-db"
  description = "Postgres — only ECS tasks in"
  vpc_id      = var.vpc_id

  ingress {
    description     = "Postgres from ECS"
    from_port       = 5432
    to_port         = 5432
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

resource "aws_db_subnet_group" "this" {
  name       = "${var.name}-db"
  subnet_ids = var.private_subnet_ids
  tags       = local.tags
}

resource "aws_db_parameter_group" "pg16_vec" {
  name   = "${var.name}-pg16-vec"
  family = "postgres16"
  parameter {
    name         = "shared_preload_libraries"
    value        = "pg_stat_statements"
    apply_method = "pending-reboot"
  }
  tags = local.tags
}

resource "aws_db_instance" "this" {
  identifier             = "${var.name}-${var.env}"
  engine                 = "postgres"
  engine_version         = "16.3"
  instance_class         = var.instance_class
  allocated_storage      = 50
  max_allocated_storage  = 500
  storage_type           = "gp3"
  storage_encrypted      = true
  db_name                = "alphadesk"
  username               = var.username
  password               = var.password
  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [aws_security_group.db.id]
  parameter_group_name   = aws_db_parameter_group.pg16_vec.name
  multi_az               = var.multi_az
  publicly_accessible    = false
  skip_final_snapshot    = var.env != "prod"
  deletion_protection    = var.env == "prod"
  backup_retention_period = var.env == "prod" ? 14 : 3
  performance_insights_enabled = true
  tags = local.tags
}

output "endpoint" {
  value = aws_db_instance.this.endpoint
}
output "db_sg_id" {
  value = aws_security_group.db.id
}