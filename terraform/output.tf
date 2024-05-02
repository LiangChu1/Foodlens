output "api_url" {
  value = aws_api_gateway_deployment.api_deployment.invoke_url
}

output "amplify_Domain" {
  value = format("https://%s.%s.%s", aws_amplify_branch.amplify_branch.branch_name, aws_amplify_app.amplify_app.id, aws_amplify_domain_association.domain_association.domain_name)
}

resource "local_file" "output" {
  content  = aws_api_gateway_deployment.api_deployment.invoke_url
  filename = "../foodlens/src/output.txt"
}
