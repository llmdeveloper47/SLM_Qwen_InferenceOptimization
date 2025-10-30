#!/bin/bash

# Week 1 Environment Setup Script for RunPod
# This script sets up the complete environment for Week 1

echo "======================================================================"
echo "Week 1: Environment Setup for RunPod"
echo "======================================================================"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${BLUE}[1/6] Checking Python version...${NC}"
python_version=$(python --version 2>&1)
echo "Found: $python_version"

if ! python -c 'import sys; assert sys.version_info >= (3,8)' 2>/dev/null; then
    echo -e "${RED}Error: Python 3.8 or higher required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"

# Check CUDA availability
echo -e "\n${BLUE}[2/6] Checking CUDA/GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
else
    echo -e "${RED}Warning: nvidia-smi not found. GPU may not be available.${NC}"
fi

# Install Python dependencies
echo -e "\n${BLUE}[3/6] Installing Python dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r setup/requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
else
    echo -e "${RED}Error: Failed to install dependencies${NC}"
    exit 1
fi

# Verify PyTorch CUDA
echo -e "\n${BLUE}[4/6] Verifying PyTorch CUDA setup...${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CPU only')"

# Create necessary directories
echo -e "\n${BLUE}[5/6] Creating directory structure...${NC}"
mkdir -p data
mkdir -p results/plots
mkdir -p results/logs
mkdir -p results/profiler_traces
echo -e "${GREEN}✓ Directories created${NC}"

# Verify transformers and datasets
echo -e "\n${BLUE}[6/6] Verifying HuggingFace libraries...${NC}"
python -c "from transformers import __version__ as tv; from datasets import __version__ as dv; print(f'transformers: {tv}'); print(f'datasets: {dv}')"

echo -e "\n${GREEN}======================================================================"
echo "Environment setup complete!"
echo "======================================================================${NC}"

echo -e "\n${BLUE}Next steps:${NC}"
echo "  1. Run: python run_week1.py"
echo "  2. Or run individual scripts: python scripts/01_download_data.py"
echo "  3. Check results in: ./results/"

echo -e "\n${BLUE}To verify installation:${NC}"
echo "  python -c 'import torch, transformers, datasets; print(\"All imports successful!\")'"

echo ""

