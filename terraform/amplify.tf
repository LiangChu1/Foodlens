
resource "aws_amplify_app" "amplify_app" {
  name       = var.app_name
  repository = var.repository
  oauth_token = var.github_access_token
  build_spec = file("amplify.yml") 
}

resource "aws_amplify_backend_environment" "amiplify_backend"{
  app_id           = aws_amplify_app.amplify_app.id
  environment_name = "backend"

  deployment_artifacts = "app-example-deployment"
  stack_name           = "amplify-app-example"
}

resource "aws_amplify_branch" "amplify_branch" {
  app_id      = aws_amplify_app.amplify_app.id
  branch_name = var.branch_name
}

resource "aws_amplify_domain_association" "domain_association" {
  app_id      = aws_amplify_app.amplify_app.id
  domain_name = var.domain_name
  wait_for_verification = false

  sub_domain {
    branch_name = aws_amplify_branch.amplify_branch.branch_name
    prefix      = var.branch_name
  }
}

