#!/bin/bash

cd ../foodlens/src

git pull

# Add the output.txt file
git add output.txt

# Commit the changes
git commit -m "Auto-commit from Terraform apply"

# Push the changes to GitHub
git push

cd "../../scripts"
