#!/bin/bash

# to use this script, you'll need to have conda installed.
# you can install conda here: https://docs.anaconda.com/miniconda/

# to run:
#     chmod +x env_setup.sh
#     ./env_setup.sh

ENV_NAME="ad_doc_ranker"
PYTHON_VERSION="3.9"

echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "Activating environment..."
# Source the conda.sh script to enable conda in this shell session
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Installing pip dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Environment '$ENV_NAME' is ready!"
echo "To activate the environment later, run: conda activate $ENV_NAME"