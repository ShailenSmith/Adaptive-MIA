#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Filesystem setup
echo "Cloning required repositories..."
git clone https://github.com/ShailenSmith/Defend_MI.git
git clone https://github.com/ShailenSmith/RoLSS.git
git clone https://github.com/RasmusTorp/Plug_and_Play_Attacks.git
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
git clone https://github.com/RasmusTorp/IF_GMI.git
git clone https://github.com/ShailenSmith/hcrbounds.git
git clone https://github.com/ShailenSmith/autoattack.git
git clone https://github.com/ShailenSmith/PPDG_MI.git


# Clean up the stylegan2-ada-pytorch repository
rm -rf stylegan2-ada-pytorch/.git/
rm -rf stylegan2-ada-pytorch/.github/
rm -f stylegan2-ada-pytorch/.gitignore

# Download pretrained file for StyleGAN2
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -P stylegan2-ada-pytorch/
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl -P stylegan2-ada-pytorch/
