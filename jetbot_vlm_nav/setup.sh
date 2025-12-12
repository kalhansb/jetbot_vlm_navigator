#!/bin/bash

# Quick setup script for JetBot VLM Navigator

echo "========================================="
echo "JetBot VLM Navigator - Setup Script"
echo "========================================="
echo ""

# Check if in workspace
if [ ! -f "src/jetbot_vlm_nav/package.xml" ]; then
    echo "Error: Please run this script from the workspace root (jetbot_ws)"
    exit 1
fi

echo "Step 1: Installing Python dependencies..."
echo "This may take 10-15 minutes for first-time setup."
echo ""

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8 support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and related packages
echo "Installing transformers and VLM dependencies..."
pip3 install transformers>=4.36 accelerate bitsandbytes sentencepiece protobuf

# Install additional dependencies
echo "Installing image processing dependencies..."
pip3 install Pillow opencv-python

echo ""
echo "Step 2: Building ROS 2 package..."
colcon build --packages-select jetbot_vlm_nav

echo ""
echo "Step 3: Sourcing workspace..."
source install/setup.bash

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To test the system:"
echo "  1. Verify CUDA: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "  2. Check GPU: nvidia-smi"
echo "  3. Launch node: ros2 launch jetbot_vlm_nav vlm_navigator.launch.py"
echo ""
echo "Note: First run will download ~13GB VLM model (may take 10-30 min)"
echo ""
