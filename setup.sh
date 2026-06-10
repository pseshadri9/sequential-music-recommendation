#!/bin/bash

set -e  # Exit immediately on error

# === CONFIG ===
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER"
INSTALL_DIR="$HOME/miniconda3"

echo "Downloading Miniconda installer..."
wget -O "$MINICONDA_INSTALLER" "$MINICONDA_URL"

echo "Installing Miniconda to $INSTALL_DIR..."
bash "$MINICONDA_INSTALLER" -b -p "$INSTALL_DIR"

# Initialize conda
echo "Initializing conda..."
eval "$($INSTALL_DIR/bin/conda shell.bash hook)"
conda init

# Activate conda and create env
echo "Creating conda environment from requirements.yml..."
conda env create -f requirements.yml

echo "Setup complete! To activate your environment, run:"
echo "  conda activate <env-name-from-yml>"

pip install torch torchvision torchaudio

# Cleanup
rm "$MINICONDA_INSTALLER"