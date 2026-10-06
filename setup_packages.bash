#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# # git update
# echo "Pulling latest updates..."
# git pull

# Install necessary packages and set up the environment
echo "Installing required packages..."
# sudo apt -y install python3.11-venv python3.11-dev
# python3.11 -m venv venv

echo "Installing Python dependencies..."
# python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
#python -m pip install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html


echo "Setup complete!"