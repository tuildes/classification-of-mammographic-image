#!/bin/bash

ENV_DIR=".venv"

python3 -m venv $ENV_DIR
source $ENV_DIR/bin/activate
pip install --upgrade pip

pip install -r requirements.txt
pip install git+https://github.com/AIM-Harvard/pyradiomics # Solution for Pyradiomics package

echo "Virtual ambient setup finished. Run the command below:"
echo "source $ENV_DIR/bin/activate"
