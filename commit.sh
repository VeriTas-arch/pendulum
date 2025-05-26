echo "Auto synchronize changes from remote repository."
git add .

echo "Please enter commit message:"
read message

git commit -m "$message"
git pull
git push
echo "Commit success (?)"
echo "Press any key to continue..."
read input

