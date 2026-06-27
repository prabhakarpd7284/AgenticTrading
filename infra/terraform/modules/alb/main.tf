// Internet-facing ALB with TLS, routing /api/* and /ws/* to separate target groups.

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
variable "public_subnet_ids" {
  type = list(string)
}
variable "certificate_arn" {
  type = string
  default = ""
}
locals {
  tags = { Project = "alphadesk", Env = var.env, ManagedBy = "terraform" }
}

resource "aws_security_group" "alb" {
  name        = "${var.name}-alb"
  description = "Public ALB — 80/443 in, ECS targets out"
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = local.tags
}

resource "aws_lb" "this" {
  name               = "${var.name}-alb"
  load_balancer_type = "application"
  subnets            = var.public_subnet_ids
  security_groups    = [aws_security_group.alb.id]
  idle_timeout       = 300    // keep WS alive
  tags               = local.tags
}

resource "aws_lb_target_group" "api" {
  name        = "${var.name}-tg-api"
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id
  deregistration_delay = 30
  health_check {
    path                = "/healthz/live"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 15
    timeout             = 5
    matcher             = "200"
  }
  tags = local.tags
}

resource "aws_lb_target_group" "ws" {
  name        = "${var.name}-tg-ws"
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id
  deregistration_delay = 30
  stickiness {
    type    = "lb_cookie"
    enabled = true
  }
  health_check {
    path                = "/healthz/live"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 15
    matcher             = "200"
  }
  tags = local.tags
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

resource "aws_lb_listener" "https" {
  count             = var.certificate_arn == "" ? 0 : 1
  load_balancer_arn = aws_lb.this.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

resource "aws_lb_listener_rule" "ws" {
  count        = var.certificate_arn == "" ? 0 : 1
  listener_arn = aws_lb_listener.https[0].arn
  priority     = 10
  condition {
    path_pattern { values = ["/ws/*"] }
  }
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ws.arn
  }
}

output "alb_sg_id" {
  value = aws_security_group.alb.id
}
output "alb_dns_name" {
  value = aws_lb.this.dns_name
}
output "tg_api_arn" {
  value = aws_lb_target_group.api.arn
}
output "tg_ws_arn" {
  value = aws_lb_target_group.ws.arn
}