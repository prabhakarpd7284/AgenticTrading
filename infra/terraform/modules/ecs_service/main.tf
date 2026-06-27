// ECS Fargate cluster + three services (api, ws, worker) + beat as a single-task service.
// All four run the same image; the command differs.

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
variable "alb_sg_id" {
  type = string
}
variable "tg_api_arn" {
  type = string
}
variable "tg_ws_arn" {
  type = string
}
variable "backend_image"      { type = string }    // ECR URI : tag
variable "worker_image" {
  type = string
}
variable "env_vars" {
  type    = map(string)
  default = {}
}
variable "secret_arns" {
  type    = map(string)
  default = {}
}
variable "cpu" {
  type = number
  default = 512
}
variable "memory" {
  type = number
  default = 1024
}
variable "desired_api" {
  type = number
  default = 2
}
variable "desired_ws" {
  type = number
  default = 2
}
variable "desired_worker" {
  type = number
  default = 2
}
variable "desired_worker_orders" {
  type = number
  default = 2
}
locals {
  tags = { Project = "alphadesk", Env = var.env, ManagedBy = "terraform" }
  env_list    = [for k, v in var.env_vars    : { name = k, value = v }]
  secret_list = [for k, v in var.secret_arns : { name = upper(k), valueFrom = v }]
}

# ----- IAM -----
data "aws_iam_policy_document" "assume_ecs" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "task_exec" {
  name               = "${var.name}-${var.env}-exec"
  assume_role_policy = data.aws_iam_policy_document.assume_ecs.json
  tags               = local.tags
}
resource "aws_iam_role_policy_attachment" "task_exec_mgd" {
  role       = aws_iam_role.task_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}
resource "aws_iam_role_policy" "task_exec_secrets" {
  role = aws_iam_role.task_exec.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Action = ["secretsmanager:GetSecretValue"],
      Resource = values(var.secret_arns)
    }]
  })
}

resource "aws_iam_role" "task" {
  name               = "${var.name}-${var.env}-task"
  assume_role_policy = data.aws_iam_policy_document.assume_ecs.json
  tags               = local.tags
}

# ----- security group -----
resource "aws_security_group" "ecs" {
  name        = "${var.name}-ecs"
  description = "ECS tasks — ALB in, egress anywhere"
  vpc_id      = var.vpc_id
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [var.alb_sg_id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = local.tags
}

# ----- cluster + logs -----
resource "aws_ecs_cluster" "this" {
  name = "${var.name}-${var.env}"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = local.tags
}

resource "aws_cloudwatch_log_group" "lg" {
  for_each          = toset(["api", "ws", "worker", "worker-orders", "beat"])
  name              = "/ecs/${var.name}/${var.env}/${each.key}"
  retention_in_days = var.env == "prod" ? 30 : 14
  tags              = local.tags
}

# ----- helper: container definition factory -----
locals {
  def = {
    api    = { cmd = ["gunicorn","config.wsgi:application","--bind","0.0.0.0:8000","--workers","3","--timeout","60"],    image = var.backend_image, expose = true  }
    ws     = { cmd = ["daphne","-b","0.0.0.0","-p","8000","config.asgi:application"],                                      image = var.backend_image, expose = true  }
    # Order placement runs on its OWN service so a multi-minute backtest can
    # never block `process_outbox`. orders ONLY here; never on the general pool.
    "worker-orders" = { cmd = ["celery","-A","config","worker","-Q","orders","-l","info","--concurrency","4"],            image = var.worker_image,  expose = false }
    # General pool — the long/heavy jobs. Deliberately does NOT consume `orders`.
    worker = { cmd = ["celery","-A","config","worker","-Q","default,agents,backtests","-l","info"],                       image = var.worker_image,  expose = false }
    beat   = { cmd = ["celery","-A","config","beat","-l","info"],                                                          image = var.worker_image,  expose = false }
  }
}

resource "aws_ecs_task_definition" "td" {
  for_each                 = local.def
  family                   = "${var.name}-${var.env}-${each.key}"
  cpu                      = var.cpu
  memory                   = var.memory
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  execution_role_arn       = aws_iam_role.task_exec.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name         = each.key
    image        = each.value.image
    essential    = true
    command      = each.value.cmd
    portMappings = each.value.expose ? [{ containerPort = 8000, protocol = "tcp" }] : []
    environment  = local.env_list
    secrets      = local.secret_list
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.lg[each.key].name
        awslogs-region        = data.aws_region.current.name
        awslogs-stream-prefix = each.key
      }
    }
  }])
  tags = local.tags
}

data "aws_region" "current" {}

# ----- services -----
resource "aws_ecs_service" "api" {
  name            = "${var.name}-${var.env}-api"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.td["api"].arn
  desired_count   = var.desired_api
  launch_type     = "FARGATE"
  network_configuration {
    subnets         = var.private_subnet_ids
    security_groups = [aws_security_group.ecs.id]
  }
  load_balancer {
    target_group_arn = var.tg_api_arn
    container_name   = "api"
    container_port   = 8000
  }
  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200
  tags = local.tags
}

resource "aws_ecs_service" "ws" {
  name            = "${var.name}-${var.env}-ws"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.td["ws"].arn
  desired_count   = var.desired_ws
  launch_type     = "FARGATE"
  network_configuration {
    subnets         = var.private_subnet_ids
    security_groups = [aws_security_group.ecs.id]
  }
  load_balancer {
    target_group_arn = var.tg_ws_arn
    container_name   = "ws"
    container_port   = 8000
  }
  tags = local.tags
}

resource "aws_ecs_service" "worker" {
  name            = "${var.name}-${var.env}-worker"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.td["worker"].arn
  desired_count   = var.desired_worker
  launch_type     = "FARGATE"
  network_configuration {
    subnets         = var.private_subnet_ids
    security_groups = [aws_security_group.ecs.id]
  }
  tags = local.tags
}

resource "aws_ecs_service" "worker_orders" {
  name            = "${var.name}-${var.env}-worker-orders"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.td["worker-orders"].arn
  desired_count   = var.desired_worker_orders
  launch_type     = "FARGATE"
  network_configuration {
    subnets         = var.private_subnet_ids
    security_groups = [aws_security_group.ecs.id]
  }
  tags = local.tags
}

resource "aws_ecs_service" "beat" {
  name            = "${var.name}-${var.env}-beat"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.td["beat"].arn
  desired_count   = 1            // singleton
  launch_type     = "FARGATE"
  network_configuration {
    subnets         = var.private_subnet_ids
    security_groups = [aws_security_group.ecs.id]
  }
  tags = local.tags
}

# ----- autoscaling for api -----
resource "aws_appautoscaling_target" "api" {
  max_capacity       = 8
  min_capacity       = var.desired_api
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "api_cpu" {
  name               = "${var.name}-${var.env}-api-cpu"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.api.resource_id
  scalable_dimension = aws_appautoscaling_target.api.scalable_dimension
  service_namespace  = aws_appautoscaling_target.api.service_namespace
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 60
  }
}

# ----- autoscaling for the order worker -----
# Order placement must keep up under bursts. CPU target-tracking mirrors the api
# policy; true queue-depth scaling would need a custom CloudWatch metric for the
# Celery `orders` queue length (not published by default).
resource "aws_appautoscaling_target" "worker_orders" {
  max_capacity       = 6
  min_capacity       = var.desired_worker_orders
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.worker_orders.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "worker_orders_cpu" {
  name               = "${var.name}-${var.env}-worker-orders-cpu"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.worker_orders.resource_id
  scalable_dimension = aws_appautoscaling_target.worker_orders.scalable_dimension
  service_namespace  = aws_appautoscaling_target.worker_orders.service_namespace
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 50
  }
}

# ----- autoscaling for the general worker (backtests/agents) -----
resource "aws_appautoscaling_target" "worker" {
  max_capacity       = 6
  min_capacity       = var.desired_worker
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.worker.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "worker_cpu" {
  name               = "${var.name}-${var.env}-worker-cpu"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  service_namespace  = aws_appautoscaling_target.worker.service_namespace
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 65
  }
}

output "ecs_sg_id" {
  value = aws_security_group.ecs.id
}
output "cluster_name" {
  value = aws_ecs_cluster.this.name
}