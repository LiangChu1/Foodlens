# Foodlens
## Concept summary
Foodlens is a project that makes nutrition easy and fun. Our application allows our users to take a picture of their meal, or upload a picture of their meal, and see all the nutritional information of the meal. We intended Foodlens to be an application that anyone could use, no matter the end user's previous exposure to technology, age, or even economic status. Our application is a completely free website that anyone can access and quickly gather the information they desire. There is no need to input personal information, or pay for access. Just access our domain, upload or take a picture of your food, and read about the nutritional information.

  
## Team

- Glenn
- Erik
- Liang
- Craig

## Prerequisites
1. AWS Terraform
  In regards to installing terraform, we recommend doing it locally. However, we have a set of scripts that installs terraform seamlessly, thus avoiding the need to install it manually. The steps on how to run this script is given in the instructions for “how to deploy it”.

2. AWS Free Tier Account
  Link to sign-in to sign-up to an AWS Free Tier: https://aws.amazon.com/free/?all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all

3. AWS Root Access Key and Secret Key
  - Click on your account username in the top right corner of the page. From there click on security credentials.
  - Scroll down to the access key section. Click on Create access key. Then check off that you “understand creating a root access key is not a best practice, but I still want to create one.”. Then click on the Create access key button again.
  - It will then bring you to a page where it will give you an access key and secret access key. Keep this information safe with you.


4. Github account
  Link to sign-in/sign-up to github: https://github.com/

5. Github authentication token for the repo
  - Click on your user icon on the top right of the page when you signed-in. Then click on settings.
  - Then scroll down to developer settings (as shown through the image below):
  - Then click on personal access tokens. Then click on generate new tokens. Choose the classic option.
  - Once you click on that option, add a name to the Note and enable all of the scope options. Then click the Generate token button at the bottom of the page. It will then give you an access token (as shown below). Keep this information safe with you.

6. Admin access to the github organization
  In order to be able to view our frontend page through the amplify domain link, you’ll need to be an admin to our github organization as otherwise, you’ll get an error message stating that you’ll won’t have the right permissions to access the link as the code associated with it is connected to a repository which is housed inside of an organization. 


## How to deploy Foodlens on Windows
1. Open powershell as admin 
2. In the desired directory, run git clone “github clone url”
3. If you already have terraform installed, skip steps 4-7
4. Change directory to be in “scripts”
5. Run this command: “./windows_install_terraform.ps1”
6. If this command is refused, type this command into powershell: “set-executionpolicy remotesigned”
7. Reopen PowerShell as administrator and go back to the directory storing Foodlens
8. Go to the “terraform” directory by typing this command: “../terraform”
9. Type: “terraform init”
10. Type: “terraform apply”
11. You will be prompted to enter in personal credentials(github oauth token, access key, secret key). Copy them where they are prompted.
12. When prompted if you would like to continue, type “yes”
13. Save the “amplify_domain” link
14. Go back to “scripts” by typing: “cd ../scripts”
15. Type: “./windows_git_commit.ps1”
16. Search for the “amplify_domain” link in your preferred browser
17. If Foodlens does not appear, wait a few minutes and refresh the page
18. You should now have access to Foodlens

## How to deploy Foodlens on MacOS: 
1. Open terminal.
2. In the desired directory, run git clone “github clone url”.
3. Then change the directory to be at the root of the repository.
4. If you have terraform already installed on your local machine, skip to step 7. Otherwise, continue to step 5.
5. Change to the scripts directory and then type in “chmod +x install_terraform.sh” to enable permissions to the script file.
6. Type in “./install_terraform.sh” to run the script to install terraform. It may prompt you to type in your mac login password to confirm the installation.
7. Then change to the terraform directory of the repository and type in the command “terraform init”.
8. Then type in “terraform apply”.
9. It will then prompt you to type in your aws access key, secret access key, and github access token. 
10. Then type in “yes” to confirm the actions performed by terraform.
11. After “terraform apply”, it will give you a “amplify_Domain” link as an output. Save this link for later.
12. Then go back to the scripts directory.
13. Type in “chmod +x git_commit.sh” to enable permissions to the script file that pushes an output.txt file to our github repository (this is needed in order for our frontend application to access the backend).
14. Type in “./git_commit.sh” to run the script.  It may prompt you to type in your mac login password to confirm the installation again.
15. Now get the “amplify_Domain” link and place it into your favorite browser (this may take a few minutes for the site to show up).


## Known bugs and disclaimers
 - when you initally deploy the infrastructure using `terraform apply`, there will sometimes be an error of uploading a new image for the first time as currently the model is undergoing a cold start. If you upload that same image after a few minutes, it should now give you the right information.

## Data sources and other dependencies
  - [nutrionx](https://www.nutritionix.com/)  API is used in order to get nutritional information about detected meals.

## How to test/run/access/use it
  - Now that you have access to the page, you can start by either taking a photo through the application or choose an image file within your local machine. From there, you would click on the “Upload new Image” button where it would then proceed to detect what meal is within that image and get you the nutritional information of said meal.


## How to destroy Foodlens on Windows:
1. When you are ready to destroy, go back to PowerShell and type: “cd ../terraform”
2. Type: “terraform destroy”
3. When you are prompted to supply your credentials, do so, and type “yes” when prompted
4. In the bottom right of your screen, search: “path”
5. Select “edit the system environment variables”
6. A new window will open, click “Environment Variables” in the bottom right
7. A new window will appear, one labeled: “User variables for <user>” and one labeled: “System Variables”
8. Under “System Variables” click the “Path” variable
9. Click “Edit…” in the bottom right
10. Scroll until you find: “C:\\Program Files\Terraform\”
11. Click on that path and click delete on the right side of the window

## How to destroy Foodlens on Mac:
1. When you're ready to destroy the deployment of our system, cd back into the terraform directory.
2. type in the command “terraform destroy”. 
3. From here, the terminal will prompt you to type in your AWS access key, secret access key, and github access token like before when you did “terraform apply”. 
4. Then it will ask you to confirm if you “want to perform these actions”. Type in “yes” to confirm the destruction of the aws technologies. 
5. Then if you want to remove the installation instance of terraform within your local machine, go back to the script directory, type in “chmod -x install_terraform.sh” to remove permission access to the script file first.
6. Then “rm /usr/local/bin/terraform” (add sudo in the beginning of the “rm” command if on Mac and proceed to type in your mac login password if prompted). 
7. Then type in “chmod -x git_commit.sh” to remove permission access to the script file first. Then your Done!



## License

MIT License

See LICENSE for details.
