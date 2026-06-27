// One-time bootstrap: S3 bucket + DynamoDB table for remote Terraform state.
// Run `terraform apply` here BEFORE any env directory is initialised.

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.50" }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type = string
  default = "ap-south-1"
}
variable "bucket_name" {
  type = string
  default = "alphadesk-tfstate"
}
variable "lock_table" {
  type = string
  default = "alphadesk-tflock"
}
resource "aws_s3_bucket" "tfstate" {
  bucket        = var.bucket_name
  force_destroy = false
  tags = { Project = "alphadesk", ManagedBy = "terraform" }
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "tflock" {
  name         = var.lock_table
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"
  attribute {
    name = "LockID"
    type = "S"
  }
  tags = { Project = "alphadesk", ManagedBy = "terraform" }
}

output "bucket" {
  value = aws_s3_bucket.tfstate.id
}
output "table" {
  value = aws_dynamodb_table.tflock.id
}