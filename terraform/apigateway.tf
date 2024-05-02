resource "aws_api_gateway_rest_api" "root_api" {
  name = "rootAPI"
  description = "API Gateway"
}

## Upload Image (post)
resource "aws_api_gateway_resource" "upload_resource" {
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  parent_id = aws_api_gateway_rest_api.root_api.root_resource_id
  path_part = "upload"
}

resource "aws_api_gateway_method" "upload_method" {
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  resource_id = aws_api_gateway_resource.upload_resource.id
  http_method = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_method_response" "upload_method_response" {
  depends_on = [aws_api_gateway_method.upload_method]
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  resource_id = aws_api_gateway_resource.upload_resource.id
  http_method = aws_api_gateway_method.upload_method.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = true,
    "method.response.header.Access-Control-Allow-Headers" = true,
    "method.response.header.Access-Control-Allow-Methods" = true
  }
}

resource "aws_api_gateway_integration" "upload_integration" {
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  resource_id = aws_api_gateway_resource.upload_resource.id
  http_method = aws_api_gateway_method.upload_method.http_method
  integration_http_method = "POST"
  type = "AWS_PROXY"
  uri = aws_lambda_function.upload_image.invoke_arn

  request_templates = {
    "application/json" = "$input.json('$')"
  }

  passthrough_behavior = "WHEN_NO_MATCH"
  content_handling = "CONVERT_TO_TEXT"
}

resource "aws_api_gateway_integration_response" "upload_integration_response" {
  depends_on = [aws_api_gateway_integration.upload_integration]
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  resource_id = aws_api_gateway_resource.upload_resource.id
  http_method = aws_api_gateway_method.upload_method.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = "'*'",
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,X-Requested-With'",
    "method.response.header.Access-Control-Allow-Methods" = "'*'"
  }
}

resource "aws_lambda_permission" "upload_lambda_permission" {
  statement_id = "AllowAPIGatewayInvoke"
  action = "lambda:InvokeFunction"
  function_name = aws_lambda_function.upload_image.function_name
  principal = "apigateway.amazonaws.com"
  source_arn = "${aws_api_gateway_rest_api.root_api.execution_arn}/*/POST/upload"
}

resource "aws_api_gateway_method" "options_method_upload" {
  rest_api_id   = aws_api_gateway_rest_api.root_api.id
  resource_id   = aws_api_gateway_resource.upload_resource.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_method_response" "options_200_upload" {
  rest_api_id   = aws_api_gateway_rest_api.root_api.id
  resource_id   = aws_api_gateway_resource.upload_resource.id
  http_method   = aws_api_gateway_method.options_method_upload.http_method
  status_code   = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true,
    "method.response.header.Access-Control-Allow-Methods" = true,
    "method.response.header.Access-Control-Allow-Origin" = true
  }
}

resource "aws_api_gateway_integration" "options_integration_upload" {
  rest_api_id   = aws_api_gateway_rest_api.root_api.id
  resource_id   = aws_api_gateway_resource.upload_resource.id
  http_method   = aws_api_gateway_method.options_method_upload.http_method
  type          = "MOCK"
  
  request_templates = {
    "application/json" = "{ \"statusCode\": 200 }"
  }

  passthrough_behavior = "WHEN_NO_MATCH"
  content_handling     = "CONVERT_TO_TEXT"
}

resource "aws_api_gateway_integration_response" "options_integration_response_upload" {
  rest_api_id   = aws_api_gateway_rest_api.root_api.id
  resource_id   = aws_api_gateway_resource.upload_resource.id
  http_method   = aws_api_gateway_method.options_method_upload.http_method
  status_code   = aws_api_gateway_method_response.options_200_upload.status_code
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'*'",
    "method.response.header.Access-Control-Allow-Methods" = "'*'",
    "method.response.header.Access-Control-Allow-Origin" = "'*'"
  }
  response_templates = {
    "application/json" = ""
  }
}


## Classification
resource "aws_api_gateway_resource" "classify_resource" {
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  parent_id = aws_api_gateway_rest_api.root_api.root_resource_id
  path_part = "classify"
}

resource "aws_api_gateway_method" "classify_method" {
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  resource_id = aws_api_gateway_resource.classify_resource.id
  http_method = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_method_response" "classify_method_response" {
  depends_on = [aws_api_gateway_method.classify_method]
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  resource_id = aws_api_gateway_resource.classify_resource.id
  http_method = aws_api_gateway_method.classify_method.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = true,
    "method.response.header.Access-Control-Allow-Headers" = true,
    "method.response.header.Access-Control-Allow-Methods" = true
  }
}


resource "aws_api_gateway_integration" "classify_integration" {
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  resource_id = aws_api_gateway_resource.classify_resource.id
  http_method = aws_api_gateway_method.classify_method.http_method
  integration_http_method = "POST"
  type = "AWS_PROXY"
  uri = aws_lambda_function.image_classification.invoke_arn

  request_templates = {
    "application/json" = "$input.json('$')"
  }

  passthrough_behavior = "WHEN_NO_MATCH"
  content_handling = "CONVERT_TO_TEXT"
}

resource "aws_api_gateway_integration_response" "classify_integration_response" {
  depends_on = [aws_api_gateway_integration.classify_integration]
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  resource_id = aws_api_gateway_resource.classify_resource.id
  http_method = aws_api_gateway_method.classify_method.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = "'*'",
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,X-Requested-With'",
    "method.response.header.Access-Control-Allow-Methods" = "'*'"
  }
}

resource "aws_lambda_permission" "classify_lambda_permission" {
  statement_id = "AllowAPIGatewayInvoke"
  action = "lambda:InvokeFunction"
  function_name = aws_lambda_function.image_classification.function_name
  principal = "apigateway.amazonaws.com"
  source_arn = "${aws_api_gateway_rest_api.root_api.execution_arn}/*/POST/classify"
}

resource "aws_api_gateway_method" "options_method_classify" {
  rest_api_id   = aws_api_gateway_rest_api.root_api.id
  resource_id   = aws_api_gateway_resource.classify_resource.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_method_response" "options_200_classify" {
  rest_api_id   = aws_api_gateway_rest_api.root_api.id
  resource_id   = aws_api_gateway_resource.classify_resource.id
  http_method   = aws_api_gateway_method.options_method_classify.http_method
  status_code   = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true,
    "method.response.header.Access-Control-Allow-Methods" = true,
    "method.response.header.Access-Control-Allow-Origin" = true
  }
}

resource "aws_api_gateway_integration" "options_integration_classify" {
  rest_api_id   = aws_api_gateway_rest_api.root_api.id
  resource_id   = aws_api_gateway_resource.classify_resource.id
  http_method   = aws_api_gateway_method.options_method_classify.http_method
  type          = "MOCK"
  
  request_templates = {
    "application/json" = "{ \"statusCode\": 200 }"
  }

  passthrough_behavior = "WHEN_NO_MATCH"
  content_handling     = "CONVERT_TO_TEXT"
}

resource "aws_api_gateway_integration_response" "options_integration_response_classify" {
  rest_api_id   = aws_api_gateway_rest_api.root_api.id
  resource_id   = aws_api_gateway_resource.classify_resource.id
  http_method   = aws_api_gateway_method.options_method_classify.http_method
  status_code   = aws_api_gateway_method_response.options_200_classify.status_code
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'*'",
    "method.response.header.Access-Control-Allow-Methods" = "'*'",
    "method.response.header.Access-Control-Allow-Origin" = "'*'"
  }
  response_templates = {
    "application/json" = ""
  }
}

## Nutrition Informaton
resource "aws_api_gateway_resource" "nutrition_resource" {
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  parent_id = aws_api_gateway_rest_api.root_api.root_resource_id
  path_part = "nutrition"
}

resource "aws_api_gateway_method" "nutrition_method" {
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  resource_id = aws_api_gateway_resource.nutrition_resource.id
  http_method = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_method_response" "nutrition_method_response" {
  depends_on = [ aws_api_gateway_method.nutrition_method ]
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  resource_id = aws_api_gateway_resource.nutrition_resource.id
  http_method = aws_api_gateway_method.nutrition_method.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = true,
    "method.response.header.Access-Control-Allow-Headers" = true,
    "method.response.header.Access-Control-Allow-Methods" = true
  }
}

resource "aws_api_gateway_integration" "nutrition_integration" {
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  resource_id = aws_api_gateway_resource.nutrition_resource.id
  http_method = aws_api_gateway_method.nutrition_method.http_method
  integration_http_method = "POST"
  type = "AWS_PROXY"
  uri = aws_lambda_function.web_scraper.invoke_arn

  request_templates = {
    "application/json" = "$input.json('$')"
  }

  passthrough_behavior = "WHEN_NO_MATCH"
  content_handling = "CONVERT_TO_TEXT"
}

resource "aws_api_gateway_integration_response" "nutrition_integration_response" {
  depends_on = [aws_api_gateway_integration.nutrition_integration]
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  resource_id = aws_api_gateway_resource.nutrition_resource.id
  http_method = aws_api_gateway_method.nutrition_method.http_method
  status_code = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Origin" = "'*'"
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,X-Requested-With'",
    "method.response.header.Access-Control-Allow-Methods" = "'*'"
  }
}

resource "aws_lambda_permission" "nutrition_lambda_permission" {
  statement_id = "AllowAPIGatewayInvoke"
  action = "lambda:InvokeFunction"
  function_name = aws_lambda_function.web_scraper.function_name
  principal = "apigateway.amazonaws.com"
  source_arn = "${aws_api_gateway_rest_api.root_api.execution_arn}/*/POST/nutrition"
}

resource "aws_api_gateway_method" "options_method_nutrition" {
  rest_api_id   = aws_api_gateway_rest_api.root_api.id
  resource_id   = aws_api_gateway_resource.nutrition_resource.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_method_response" "options_200_nutrition" {
  rest_api_id   = aws_api_gateway_rest_api.root_api.id
  resource_id   = aws_api_gateway_resource.nutrition_resource.id
  http_method   = aws_api_gateway_method.options_method_nutrition.http_method
  status_code   = "200"
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true,
    "method.response.header.Access-Control-Allow-Methods" = true,
    "method.response.header.Access-Control-Allow-Origin" = true
  }
}

resource "aws_api_gateway_integration" "options_integration_nutrition" {
  rest_api_id   = aws_api_gateway_rest_api.root_api.id
  resource_id   = aws_api_gateway_resource.nutrition_resource.id
  http_method   = aws_api_gateway_method.options_method_nutrition.http_method
  type          = "MOCK"
  
  request_templates = {
    "application/json" = "{ \"statusCode\": 200 }"
  }

  passthrough_behavior = "WHEN_NO_MATCH"
  content_handling     = "CONVERT_TO_TEXT"
}

resource "aws_api_gateway_integration_response" "options_integration_response_nutrition" {
  rest_api_id   = aws_api_gateway_rest_api.root_api.id
  resource_id   = aws_api_gateway_resource.nutrition_resource.id
  http_method   = aws_api_gateway_method.options_method_nutrition.http_method
  status_code   = aws_api_gateway_method_response.options_200_nutrition.status_code
  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'*'",
    "method.response.header.Access-Control-Allow-Methods" = "'*'",
    "method.response.header.Access-Control-Allow-Origin" = "'*'"
  }
  response_templates = {
    "application/json" = ""
  }
}



## Deployment
resource "aws_api_gateway_deployment" "api_deployment" {
  depends_on = [
    aws_api_gateway_integration.upload_integration,
    aws_api_gateway_integration_response.upload_integration_response,
    aws_api_gateway_integration.options_integration_upload,
    aws_api_gateway_integration_response.options_integration_response_upload,
    aws_api_gateway_integration.classify_integration,
    aws_api_gateway_integration_response.classify_integration_response,
    aws_api_gateway_integration.options_integration_classify,
    aws_api_gateway_integration_response.options_integration_response_classify,
    aws_api_gateway_integration.nutrition_integration,
    aws_api_gateway_integration_response.nutrition_integration_response,
    aws_api_gateway_integration.options_integration_nutrition,
    aws_api_gateway_integration_response.options_integration_response_nutrition,
  ]
  rest_api_id = aws_api_gateway_rest_api.root_api.id
  stage_name = "prod"
}

#Valid url endings
#classify       - to process image and get food label
#upload-image   - to upload image to S3 bucket
#get-macros     - to get nutritional content from nutritionX
#
#
#
#
#
#
