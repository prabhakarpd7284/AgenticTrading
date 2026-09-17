// AWS Amplify hosting for the Vite SPA. Connected to GitHub; builds on push.

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
variable "repository_url" {
  type = string
}
variable "branch" {
  type = string
  default = "main"
}
variable "github_token_arn" { type = string }    // Secrets Manager ARN with a PAT
variable "vite_env" {
  type    = map(string)
  default = {}
}

locals {
  tags = { Project = "alphadesk", Env = var.env, ManagedBy = "terraform" }
}

data "aws_secretsmanager_secret_version" "gh" {
  secret_id = var.github_token_arn
}

resource "aws_amplify_app" "this" {
  name         = "${var.name}-${var.env}"
  repository   = var.repository_url
  access_token = data.aws_secretsmanager_secret_version.gh.secret_string
  platform     = "WEB"
  enable_auto_branch_creation = false
  enable_branch_auto_build    = true
  enable_branch_auto_deletion = false

  build_spec = <<YAML
version: 1
applications:
  - appRoot: frontend
    frontend:
      phases:
        preBuild:
          commands:
            # The frontend ships ONLY pnpm-lock.yaml (no package-lock.json), so
            # `npm ci` fails. Use the lockfile's package manager via corepack.
            - corepack enable
            - corepack prepare pnpm@9 --activate
            - pnpm install --frozen-lockfile
        build:
          commands:
            - pnpm build
      artifacts:
        baseDirectory: dist
        files:
          - '**/*'
      cache:
        paths:
          - node_modules/**/*
          - .pnpm-store/**/*
YAML

  custom_rule {
    source = "</^[^.]+$|\\.(?!(css|gif|ico|jpg|js|png|txt|svg|woff|woff2|ttf|map|json)$)([^.]+$)/>"
    status = "200"
    target = "/index.html"
  }

  environment_variables = merge(
    var.vite_env,
    { AMPLIFY_MONOREPO_APP_ROOT = "frontend" }
  )

  tags = local.tags
}

resource "aws_amplify_branch" "branch" {
  app_id      = aws_amplify_app.this.id
  branch_name = var.branch
  stage       = var.env == "prod" ? "PRODUCTION" : "DEVELOPMENT"
  framework   = "React"
  tags        = local.tags
}

output "default_domain" {
  value = aws_amplify_app.this.default_domain
}
output "app_id" {
  value = aws_amplify_app.this.id
}