#!/bin/bash

# Quick Start Guide for JetBot VLM Explorer
# =========================================

echo "🤖 JetBot VLM Explorer - Quick Start"
echo ""

# Step 1: Source the workspace
echo "📦 Step 1: Sourcing workspace..."
source /home/kalhan/Projects/Real_Experiments/jetbot_ws/install/setup.bash
echo "✓ Workspace sourced"
echo ""

# Step 2: Start JetBot Driver (in separate terminal)
echo "🚗 Step 2: Start JetBot driver in a separate terminal:"
echo "   cd /home/kalhan/Projects/Real_Experiments/jetbot_ws/src"
echo "   python3 jetbot_driver_code.py"
echo ""

# Step 3: Launch VLM Explorer
echo "🔍 Step 3: Launch VLM Explorer"
echo "   Command: ros2 launch jetbot_vlm_nav vlm_navigator.launch.py"
echo ""
read -p "Press ENTER to launch VLM Explorer (or Ctrl+C to exit)..."

# Launch the explorer
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py

# Cleanup
echo ""
echo "🛑 Explorer stopped. Make sure to stop the JetBot driver as well."
