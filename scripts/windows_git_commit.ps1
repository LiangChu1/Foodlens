# Change directory to the specified path
Set-Location "../foodlens/src"

# Pull changes from the Git repository
git pull

# Add the output.txt file
git add output.txt

# Commit the changes
git commit -m "Auto-commit from Terraform apply"

# Push the changes to GitHub
git push

Set-Location "../../scripts"