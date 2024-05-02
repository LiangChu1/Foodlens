//in ./foodlens 
//amplify init
//amplify push

//terraform init
//terraform plan
//terraform apply
//terraform output -json invoke_url

//terraform destroy

//author note: We know this is not the best way to structure terraform
//files, but it makes it easier on our end to create seperate resources
//and keep everything organized. Terraform can't tell the difference
//if the files are seperate or all in main.tf, as it reads every file
//in the directory when applied

provider "aws" {
  region = var.region

  access_key = var.aws_access_key
  secret_key = var.aws_secret_access_key
}



