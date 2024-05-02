

variable "repository" {
  type        = string
  description = "github repo url"
  default     = "https://github.com/RIT-SWEN-343/2235-1a-SPECIAL-REPO-Cheemsburger"
}

variable "app_name" {
  type        = string
  description = "AWS Amplify App Name"
  default     = "FoodLensApp"
}

variable "branch_name" {
  type        = string
  description = "AWS Amplify App Repo Branch Name"
  default     = "main"
}


variable "domain_name" {
  type        = string
  default     = "amplifyapp.com"
  description = "AWS Amplify Domain Name"
}

variable "github_access_token" {
  type        = string
  description = "github token to connect github repo"
}
variable "region" {
  description = "The AWS region to deploy resources in"
  type        = string
  default = "us-east-1"
}

variable "aws_access_key" {
  type    = string
  description = "Access Key for the AWS account"
}

variable "aws_secret_access_key" {
  type    = string
  description = "Secret Access Key for the AWS account"
}

