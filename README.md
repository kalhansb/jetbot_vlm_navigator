# JetBot VLM Navigator

ROS 2 package for vision-language model (VLM) based object detection and autonomous navigation. Uses on-device VLMs (like LLaVA) running on your NVIDIA RTX 4060 to find and navigate towards objects.

## Features

- **On-Device VLM**: Runs locally on NVIDIA RTX 4060 (no API costs!)
- **Object Detection & Navigation**: Find any object using natural language and navigate towards it
- **Efficient**: Uses 4-bit quantization to fit 7B parameter models in 8GB VRAM
- **Flexible**: Can detect any object - just change the target parameter

## Hardware Requirements

- **GPU**: NVIDIA RTX 4060 (8GB VRAM) or better
- **RAM**: 16GB+ recommended
- **Camera**: RGB camera publishing to `/camera/color/image_raw`

## Dependencies

### System Requirements
```bash
# CUDA Toolkit (if not already installed)
# Check with: nvcc --version
sudo apt-get install nvidia-cuda-toolkit

# Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes sentencepiece protobuf
pip install Pillow opencv-python
```

### ROS 2 Dependencies
```bash
sudo apt-get install ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-image-transport
```

## Installation

1. Clone this package into your ROS 2 workspace:
```bash
cd ~/jetbot_ws/src
# Package should already be there
```

2. Install Python dependencies:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.36 accelerate bitsandbytes sentencepiece protobuf
```

3. Build the workspace:
```bash
cd ~/jetbot_ws
colcon build --packages-select jetbot_vlm_nav
source install/setup.bash
```

## Usage

### Basic Usage

Find and navigate towards a bottle:
```bash
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:=bottle
```

### Find Different Objects

```bash
# Find a chair
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:=chair

# Find a person
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:=person

# Find a cup
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:=cup
```

### Advanced Usage

Customize camera topic and speeds:
```bash
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py \
  camera_topic:=/camera/image_raw \
  target_object:=bottle \
  linear_speed:=0.15 \
  angular_speed:=0.4 \
  query_interval:=1.5
```

### Use Different VLM Models

```bash
# Use LLaVA 1.5 (alternative model)
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py \
  model_name:=llava-hf/llava-1.5-7b-hf \
  target_object:=bottle

# Use Qwen2-VL (another good option)
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py \
  model_name:=Qwen/Qwen2-VL-7B-Instruct \
  target_object:=bottle
```

## Configuration

Edit `config/vlm_params.yaml` to change default parameters:

```yaml
vlm_navigator:
  ros__parameters:
    model_name: "llava-hf/llava-v1.6-mistral-7b-hf"
    use_4bit: true              # Use 4-bit quantization
    target_object: "bottle"      # Object to find
    query_interval: 1.0          # How often to query VLM (seconds)
    linear_speed: 0.2            # Forward speed (m/s)
    angular_speed: 0.5           # Rotation speed (rad/s)
```

## How It Works

1. **Image Capture**: Subscribes to camera topic `/camera/color/image_raw`
2. **VLM Processing**: Every `query_interval` seconds, sends image to on-device VLM
3. **Object Detection**: VLM identifies if target object is present and its position
4. **Navigation**: Publishes velocity commands to `/cmd_vel` to approach the object
5. **Stopping**: Stops when object is close enough (based on relative size)

## Topics

### Subscribed Topics
- `/camera/color/image_raw` (sensor_msgs/Image): RGB camera feed

### Published Topics
- `/cmd_vel` (geometry_msgs/Twist): Velocity commands for robot
- `/vlm_nav/status` (std_msgs/String): Status messages about detection

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_topic` | string | `/camera/color/image_raw` | Camera image topic |
| `cmd_vel_topic` | string | `/cmd_vel` | Velocity command topic |
| `model_name` | string | `llava-hf/llava-v1.6-mistral-7b-hf` | HuggingFace VLM model |
| `use_4bit` | bool | `true` | Use 4-bit quantization |
| `device` | string | `cuda` | Device to run model on |
| `target_object` | string | `bottle` | Object to find |
| `query_interval` | float | `1.0` | Seconds between VLM queries |
| `linear_speed` | float | `0.2` | Maximum forward velocity (m/s) |
| `angular_speed` | float | `0.5` | Maximum rotation velocity (rad/s) |
| `stop_distance_threshold` | float | `0.35` | Stop when object size > threshold |

## Performance Tips

### GPU Memory
The default configuration uses 4-bit quantization which allows the 7B parameter model to run in ~5-6GB VRAM. If you have memory issues:

1. Ensure no other GPU processes are running
2. Close unnecessary applications
3. Try a smaller model or lower query interval

### Speed vs Accuracy
- **Faster response**: Decrease `query_interval` (e.g., 0.5 seconds)
- **Better efficiency**: Increase `query_interval` (e.g., 2.0 seconds)

### First Run
The first time you run the node, it will download the VLM model (~13GB for LLaVA-1.6-Mistral-7B). This may take 10-30 minutes depending on your internet speed. Subsequent runs will be much faster.

## Troubleshooting

### CUDA Out of Memory
```bash
# Solution 1: Ensure 4-bit quantization is enabled
use_4bit:=true

# Solution 2: Clear GPU memory
nvidia-smi
# Kill any processes using GPU
```

### Model Download Issues
```bash
# Set HuggingFace cache directory
export HF_HOME=~/hf_cache
mkdir -p $HF_HOME

# If download fails, try manually:
python -c "from transformers import LlavaNextProcessor; LlavaNextProcessor.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf')"
```

### Slow Performance
```bash
# Check GPU is being used:
nvidia-smi

# Check CUDA is available in Python:
python -c "import torch; print(torch.cuda.is_available())"
```

### Robot Not Moving
- Check `/cmd_vel` topic is correct for your robot
- Verify camera topic is publishing: `ros2 topic echo /camera/color/image_raw`
- Check status messages: `ros2 topic echo /vlm_nav/status`

## Supported VLM Models

| Model | Size | VRAM (4-bit) | Recommended For |
|-------|------|--------------|-----------------|
| `llava-hf/llava-v1.6-mistral-7b-hf` | 7B | ~6GB | **Best overall** |
| `llava-hf/llava-1.5-7b-hf` | 7B | ~5GB | Good alternative |
| `Qwen/Qwen2-VL-7B-Instruct` | 7B | ~6GB | Good for complex queries |

## Examples

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

### View Detection Status
```bash
ros2 topic echo /vlm_nav/status
```

### Test Different Objects
```bash
# Common household objects
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:="red cup"
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:="laptop"
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:="book"
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py target_object:="phone"
```

## Architecture

```
┌─────────────┐
│   Camera    │
└──────┬──────┘
       │ /camera/color/image_raw
       ▼
┌─────────────────────────┐
│  VLM Navigator Node     │
│  ┌──────────────────┐   │
│  │  Image Buffer    │   │
│  └────────┬─────────┘   │
│           │             │
│  ┌────────▼─────────┐   │
│  │  On-Device VLM   │   │ NVIDIA RTX 4060
│  │  (LLaVA/Qwen)    │   │ with 4-bit Quant
│  └────────┬─────────┘   │
│           │             │
│  ┌────────▼─────────┐   │
│  │ Object Detection │   │
│  │  & Localization  │   │
│  └────────┬─────────┘   │
│           │             │
│  ┌────────▼─────────┐   │
│  │  Navigation      │   │
│  │  Controller      │   │
│  └────────┬─────────┘   │
└───────────┼─────────────┘
            │ /cmd_vel
            ▼
      ┌──────────┐
      │  Robot   │
      └──────────┘
```

## License

Apache-2.0

## Author

Kalhan Sandaru (kalhansandaru@gmail.com)

## Acknowledgments

- Built on HuggingFace Transformers
- Uses LLaVA vision-language models
- Optimized for NVIDIA RTX GPUs
