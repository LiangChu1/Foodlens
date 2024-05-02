# Define download URL
#version = latest relase
#OS = 
#PROC = amd64/32
#FILENAME="terraform_${VERSION}_${OS}_${PROC}.zip"
#LINK="https://releases.hashicorp.com/terraform/${VERSION}/${FILENAME}"

$OS = 'windows'
$PROC = 'amd64'
$VERSION = '1.7.5'
$FILENAME = "terraform_${VERSION}_${OS}_${PROC}.zip"
$downloadUrl = "https://releases.hashicorp.com/terraform/${VERSION}/${FILENAME}"

# Define destination directory
$destinationDirectory = "C:\Program Files\Terraform"

# Create destination directory if it doesn't exist
if (-not (Test-Path -Path $destinationDirectory)) {
    New-Item -ItemType Directory -Path $destinationDirectory | Out-Null
}

# Download Terraform ZIP file
$zipFile = Join-Path -Path $destinationDirectory -ChildPath "terraform_$VERSION.zip"
Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile

# Unzip Terraform ZIP file
Expand-Archive -Path $zipFile -DestinationPath $destinationDirectory -Force

# Clean up ZIP file
Remove-Item -Path $zipFile

Write-Output "Terraform $VERSION installed successfully to $destinationDirectory"


#Get the current user's environment variable "Path"
$currentPath = [System.Environment]::GetEnvironmentVariable("Path", "User")

#Check if the Terraform path is already in the "Path" variable
if ($currentPath -notlike "$destinationDirectory") {
    # Add the Terraform path to the "Path" variable
    $newPath = "$currentPath;$destinationDirectory"
    [System.Environment]::SetEnvironmentVariable("Path", $newPath, "User")

#Notify the user
    Write-Output "Terraform path added to the environment variables."
} else {
    Write-Output "Terraform path is already in the environment variables."
}