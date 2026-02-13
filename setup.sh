#!/bin/bash

# Quick setup script for DSPy demo project

set -e  # Exit on error

echo "=========================================="
echo "DSPy Small-to-SOTA Demo - Quick Setup"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi
echo "✓ Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"

# Install requirements
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create .env if it doesn't exist
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo "⚠ Please edit .env and add your API keys/tokens"
else
    echo "✓ .env file already exists"
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/gsm8k data/hotpotqa results models dspy_cache
echo "✓ Directories created"

# Download datasets (optional, will be auto-downloaded on first run)
echo ""
read -p "Download datasets now? (y/n) [n]: " download_data
if [ "$download_data" = "y" ]; then
    echo "Downloading GSM8K..."
    python data/gsm8k_loader.py
    echo "Downloading HotPotQA..."
    python data/hotpotqa_loader.py
    echo "✓ Datasets downloaded"
else
    echo "⏭ Skipping dataset download (will auto-download when needed)"
fi

# Models
echo ""
read -p "Download models now? (y/n) [n]: " download_models
if [ "$download_models" = "y" ]; then
    echo "Which models to download?"
    echo "1) Small models only (Phi-2) - ~5 GB"
    echo "2) All models - ~150+ GB"
    read -p "Choice [1]: " model_choice
    
    if [ "$model_choice" = "2" ]; then
        python setup_models.py --models all
    else
        python setup_models.py --models all-small
    fi
    echo "✓ Models downloaded"
else
    echo "⏭ Skipping model download"
    echo "   You can use API models or download later with: python setup_models.py"
fi

# Final instructions
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Edit .env and add your API keys (if using API models)"
echo "  3. Start Jupyter: jupyter notebook"
echo "  4. Open notebooks/gsm8k_demo.ipynb to begin!"
echo ""
echo "For command-line evaluation:"
echo "  python evaluate.py --task gsm8k --approach all --subset dev"
echo ""
echo "Enjoy experimenting with DSPy! 🚀"
