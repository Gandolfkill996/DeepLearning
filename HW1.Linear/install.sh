#!/bin/bash

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
  echo "requirements.txt not found!"
  exit 1
fi

# Create virtual environment
if [ ! -d venv ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "All dependencies installed successfully!"
