#!/bin/bash

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install ipykernel and create kernel for Jupyter
pip install ipykernel
python -m ipykernel install --user --name=balance_predictions --display-name "Python (balance_predictions)"

echo "Virtual environment setup complete. To activate it, run:"
echo "source .venv/bin/activate"
