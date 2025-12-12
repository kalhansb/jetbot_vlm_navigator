#!/bin/bash

# Example launch commands for JetBot VLM Navigator

echo "========================================="
echo "JetBot VLM Navigator - Example Commands"
echo "========================================="
echo ""

echo "1. Basic usage - Find a bottle:"
echo "   ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:=bottle"
echo ""

echo "2. Find different objects:"
echo "   ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:=cup"
echo "   ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:=person"
echo "   ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:=chair"
echo ""

echo "3. Adjust speed parameters:"
echo "   ros2 launch jetbot_vlm_nav vlm_navigator.launch.py \\"
echo "     target_object:=bottle \\"
echo "     linear_speed:=0.15 \\"
echo "     angular_speed:=0.4"
echo ""

echo "4. Use different camera topic:"
echo "   ros2 launch jetbot_vlm_nav vlm_navigator.launch.py \\"
echo "     camera_topic:=/camera/image_raw \\"
echo "     target_object:=bottle"
echo ""

echo "5. Monitor status:"
echo "   ros2 topic echo /vlm_nav/status"
echo ""

echo "6. Check GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo ""

echo "========================================="
echo "Ready to start? Run one of the commands above!"
echo "========================================="
