
// S3 bucket where images will be uploaded
resource "random_integer" "six_digit_number" {
  min = 100000
  max = 999999
}

resource "aws_s3_bucket" "destination_bucket" {
  bucket = "foodlensbucket${random_integer.six_digit_number.result}-dev"
  force_destroy = true
}