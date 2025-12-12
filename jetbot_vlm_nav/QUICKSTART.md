# Quick Start Guide - JetBot VLM Navigator

## What is this?
A ROS 2 package that uses an on-device Vision Language Model (VLM) to detect objects and autonomously navigate your robot towards them. Runs entirely on your NVIDIA RTX 4060 - no cloud API needed!

## Prerequisites
- NVIDIA RTX 4060 (or better GPU with 8GB+ VRAM)
- ROS 2 installed
- Camera publishing to `/camera/color/image_raw`

## Quick Setup (3 steps)

### 1. Install Python Dependencies
```bash
cd ~/jetbot_ws/src/jetbot_vlm_nav
pip install -r requirements.txt
```

**Or use the setup script:**
```bash
cd ~/jetbot_ws/src/jetbot_vlm_nav
./setup.sh
```

### 2. Build Package
```bash
cd ~/jetbot_ws
colcon build --packages-select jetbot_vlm_nav
source install/setup.bash
```

### 3. Test Installation
```bash
cd ~/jetbot_ws/src/jetbot_vlm_nav
python3 test_setup.py
```

## Usage

### Basic - Find a bottle
```bash
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:=bottle
```

### Find other objects
```bash
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:=cup
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:=person
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:=phone
```

### Custom settings
```bash
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py \
  target_object:=bottle \
  camera_topic:=/camera/image_raw \
  linear_speed:=0.15 \
  angular_speed:=0.4
```

## Monitoring

### View detection status
```bash
ros2 topic echo /vlm_nav/status
```

### Monitor GPU usage
```bash
watch -n 1 nvidia-smi
```

### Check robot commands
```bash
ros2 topic echo /cmd_vel
```

## Important Notes

1. **First Run**: Downloads ~13GB model (10-30 minutes), subsequent runs are instant
2. **GPU Memory**: Uses ~5-6GB VRAM with 4-bit quantization
3. **Performance**: Processes images every 1 second (adjustable with `query_interval`)
4. **Camera**: Make sure camera topic is publishing before starting

## Troubleshooting

### CUDA not available
```bash
# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# If false, reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Out of GPU memory
```bash
# Make sure nothing else is using GPU
nvidia-smi

# Kill other processes or ensure use_4bit:=true
```

### Robot not moving
- Verify camera topic: `ros2 topic list | grep camera`
- Check cmd_vel topic matches your robot
- View status: `ros2 topic echo /vlm_nav/status`

## Files Overview

```
jetbot_vlm_nav/
├── README.md                    # Full documentation
├── QUICKSTART.md               # This file
├── requirements.txt            # Python dependencies
├── setup.sh                    # Auto-setup script
├── test_setup.py              # Dependency checker
├── examples.sh                # Usage examples
├── package.xml                # ROS package manifest
├── CMakeLists.txt            # Build configuration
├── setup.py                  # Python package setup
├── src/
│   └── vlm_navigator_node.py # Main node (VLM + navigation)
├── launch/
│   └── vlm_navigator.launch.py # Launch file
└── config/
    └── vlm_params.yaml        # Default parameters
```

## How It Works

1. **Captures** image from camera
2. **Processes** with on-device VLM (LLaVA running on RTX 4060)
3. **Detects** target object and estimates position
4. **Navigates** robot towards object
5. **Stops** when close enough

## Performance Tips

- **Faster response**: Lower `query_interval` (e.g., 0.5s)
- **Less GPU load**: Higher `query_interval` (e.g., 2.0s)  
- **Better accuracy**: Use slower speeds
- **Memory issues**: Ensure `use_4bit:=true`

## Support

For issues or questions, see full README.md or contact: kalhansandaru@gmail.com

---

**Ready to go?** Run: `ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:=bottle`
