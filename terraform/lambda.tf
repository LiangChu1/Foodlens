
resource "aws_lambda_function" "image_classification" {
  function_name = "lambda-image-classification-model-image"
  
  # Docker image URI
  # image_uri = "public.ecr.aws/y1u8l9r7/image-classification-model-image-2-public:latest"
  image_uri = "590183726950.dkr.ecr.us-east-1.amazonaws.com/image-classification-model-image-2:latest"


  package_type = "Image"
  role = aws_iam_role.lambda_exec.arn

  memory_size = 1024
  timeout = 120
}

resource "aws_iam_role" "lambda_exec" {
  name = "lambda_exec_role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Principal = {
          Service = "lambda.amazonaws.com",
        },
        Effect = "Allow",
        Sid = "",
      },
    ],
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "lambda_exec_policy"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action: [
          "s3:Get*",
          "s3:List*",
          "s3:Describe*",
          "s3-object-lambda:Get*",
          "s3-object-lambda:List*",
          #we added every single ecr policy because we arent sure which to use. Make sure to refactor
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
          "ecr:DescribeImages",
          "ecr:BatchGetImage",
          "ecr:GetLifecyclePolicy",
          "ecr:GetLifecyclePolicyPreview",
          "ecr:ListTagsForResource",
          "ecr:DescribeImageScanFindings",
          "ecr:SetRepositoryPolicy",
          "ecr:GetRepositoryPolicy"
        ],
        Resource: "*"
      },
      {
        Effect: "Allow",
        Action: [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Resource: "arn:aws:logs:us-east-1:590183726950:*"
      }
    ]
  })
}

resource "aws_iam_policy" "lambda_ecr_pull_policy" {
  name = "lambda_ecr_pull_policy"
  description = "Allows lambda function to pull images from ECR"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action   = "ecr:BatchCheckLayerAvailability",
        Effect   = "Allow",
        Resource = "*"
      },
      {
        Action   = "ecr:GetDownloadUrlForLayer",
        Effect   = "Allow",
        Resource = "*"
      },
      {
        Action   = "ecr:BatchGetImage",
        Effect   = "Allow",
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_ecr_pull_attach" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = aws_iam_policy.lambda_ecr_pull_policy.arn
}

#--------------------------------------------------------------------------------
resource "aws_lambda_function" "upload_image"{
  function_name = "upload-image-to-lambda"
  role = aws_iam_role.lambda_role.arn
  handler = "index.handler" // Name of python file and handler function
  runtime = "python3.8"

  memory_size = 1024
  timeout = 120
  
  // Uploading an image to bucket
  filename         = "uploadToS3.zip" // Path to the zip file containing the Lambda function code
  source_code_hash = filebase64sha256("uploadToS3.zip") // Base64-encoded SHA-256 hash of the zip file

  // Environment variables
  environment {
    variables = {
      DESTINATION_BUCKET = aws_s3_bucket.destination_bucket.bucket // Pass the destination bucket name to the Lambda function
    }
  }
}

// IAM role for the Lambda function
resource "aws_iam_role" "lambda_role" {
  name               = "generalUser" // Replace with your desired IAM role name
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = {
        Service = "lambda.amazonaws.com"
      },
      Action    = "sts:AssumeRole"
    }]
  })
}

// IAM role policy for the Lambda function
resource "aws_iam_role_policy" "lambda_role_policy" {
  name   = "LambdaRolePolicy"
  role   = aws_iam_role.lambda_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect   = "Allow",
      Action   = [
        "s3:PutObject",
        "s3:GetBucketLocation",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      Resource = [
        "${aws_s3_bucket.destination_bucket.arn}/*"
      ]
    }]
  })
}

#------------------------------------------------------------------------
resource "aws_lambda_function" "web_scraper" {
  filename         = "NutritionWebscraper.zip"
  function_name    = "web_scraper_function"
  role             = aws_iam_role.lambda_nutrition_exec.arn
  handler          = "index.handler"
  runtime          = "python3.8"
  source_code_hash = filebase64sha256("NutritionWebscraper.zip") // Base64-encoded SHA-256 hash of the zip file
  timeout = 300
  memory_size = 1000
}

resource "aws_iam_role" "lambda_nutrition_exec" {
  name = "lambda_exec_role_2"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
}

resource "aws_iam_policy_attachment" "lambda_basic_execution" {
  name       = "lambda_basic_execution"
  roles      = [aws_iam_role.lambda_nutrition_exec.name]
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}