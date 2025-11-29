#!/bin/bash

# Define the URLs for the two repos
REPO_1="https://github.com/singlaayush/barttender.git"
REPO_2="https://github.com/singlaayush/CardiomegalyBiomarkers.git"

# Clone the first repository
echo "Cloning the first repository..."
git clone $REPO_1

# Navigate into the first repository's folder
cd barttender

# Clone the second repository inside the first repo folder
echo "Cloning the second repository..."
git clone $REPO_2

echo "Repositories cloned successfully!"

cd ..