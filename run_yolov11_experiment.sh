#!/bin/bash

# YOLOv11 Experiment Runner
# This script activates the anyglow conda environment and runs the YOLOv11 experiment

set -e  # Exit on any error

echo "🚀 Starting YOLOv11 Experiment with anyglow environment"
echo "========================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install conda first."
    exit 1
fi

# Initialize conda for bash if not already done
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "🔧 Initializing conda..."
    eval "$(conda shell.bash hook)"
fi

# Check if anyglow environment exists
if ! conda env list | grep -q "^anyglow "; then
    echo "❌ anyglow conda environment not found."
    echo "Please create it first with necessary dependencies."
    exit 1
fi

echo "🔧 Activating anyglow conda environment..."
conda activate anyglow

# Verify Python and key packages
echo "✅ Environment activated successfully!"
echo "📋 Environment details:"
echo "   Python: $(python --version)"
echo "   Working directory: $(pwd)"
echo "   CUDA available: $(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "PyTorch not found")"

# Check if the data directory exists
DATA_PATH="/hdd1t/mduc/ultralytics-gdrive-ops/logs/data_test_1"
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ Data directory not found: $DATA_PATH"
    exit 1
fi

echo "✅ Data directory found: $DATA_PATH"

# Run the experiment
echo ""
echo "🎯 Starting YOLOv11 training and testing experiment..."
echo "========================================================"

python yolov11_experiment.py

echo ""
echo "🎉 Experiment completed!"
echo "📁 Check the 'experiments' directory for results." 