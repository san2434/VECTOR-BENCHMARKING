#!/bin/bash
# Setup script for RAG Vector Benchmarking System

set -e

echo "🚀 RAG Vector Benchmarking System - Setup Script"
echo "=================================================="
echo ""

# Check Python version
echo "✓ Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Found Python $python_version"

# Create virtual environment
echo ""
echo "✓ Creating virtual environment..."
python3 -m venv venv
echo "  Created venv/"

# Activate virtual environment
echo ""
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "✓ Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install dependencies
echo ""
echo "✓ Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "✓ Creating directories..."
mkdir -p data
mkdir -p results
mkdir -p logs

# Check if .env exists
echo ""
if [ -f .env ]; then
    echo "✓ .env file already exists"
else
    echo "✓ Creating .env from template..."
    cp .env.example .env
    echo "  ⚠️  Please edit .env with your API keys:"
    echo "     - OPENAI_API_KEY=your_key_here"
    echo "     - (Optional) PINECONE_API_KEY"
    echo "     - (Optional) PostgreSQL credentials"
fi

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Run: python main.py"
echo ""
