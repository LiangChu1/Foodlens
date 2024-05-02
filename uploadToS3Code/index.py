import json
import os
import uuid
import boto3
from io import BytesIO
import base64

destination_bucket = os.environ.get('DESTINATION_BUCKET')

def uploadFileToS3(file_name, file_content):
    s3 = boto3.client("s3")
    try:
        s3.put_object(
            Bucket=destination_bucket,
            Key=file_name,
            Body=file_content.getvalue(),  # Pass the raw bytes
            ContentType="image/jpeg"  # Set the content type based on the image format
        )
        print(f"File '{file_name}' uploaded to S3 bucket '{destination_bucket}'")
    except Exception as e:
        print(f"Error uploading file to S3: {str(e)}")

def handler(event, context):
    try:
        # Parse the JSON body
        body = json.loads(event.get('body', '{}'))
        # Get the image data
        actual_body = body.get('body')
        image_data = actual_body.get('imageData')

        if image_data:
            # Process the image
            image_content = base64.b64decode(image_data)

            # Upload image to S3
            file_name = str(uuid.uuid4()) + ".jpg"  # Generate a unique file name
            blob = BytesIO(image_content)
            uploadFileToS3(file_name, blob)

            # Get the URL of the uploaded file
            s3_url = f"https://{destination_bucket}.s3.amazonaws.com/{file_name}"

            response = {
                "statusCode": 200,
                "headers": {
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                },
                "body": json.dumps({"message": "Image uploaded successfully!", "url": s3_url}),
            }
        else:
            raise Exception("'imageData' field is missing or empty in the request body")

    except Exception as e:
        response = {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

    return response