echo "Auto synchronize changes to remote repository."
cd "$(dirname "$0")/.."
git add .

echo "Please enter commit message:"
read message

git commit -m "$message"
git push
echo "Commit success (?)"
echo "Press any key to continue..."
read input
