# Exploration Mode - Changelog

## Overview
The VLM Navigator has been transformed into a **VLM Explorer** that randomly explores the environment while:
- Detecting obstacles ahead
- Narrating likely locations (kitchen, hallway, living room, etc.)
- Avoiding collisions through intelligent direction changes

---

## Key Changes

### 1. Class Rename
- **Old**: `VLMNavigator`
- **New**: `VLMExplorer`

### 2. Parameters Changed

#### Removed Parameters:
- `target_object` - No longer searching for specific objects
- `stop_distance_threshold` - Not approaching targets
- `center_tolerance` - Not centering on objects
- `safety_timeout` - Exploration doesn't need safety stops

#### Added Parameters:
- `explore_duration` (default: 2.0s) - How long to move in one direction
- `obstacle_keyword` (default: "obstacle") - Keyword to detect obstacles in VLM response

#### Modified Parameters:
- `query_interval`: Changed from 1.0s → 3.0s (less frequent queries for exploration)
- `max_new_tokens`: Changed from 150 → 200 (allow longer location descriptions)

### 3. State Variables Changed

#### Removed:
- `object_detected` - Not detecting specific objects
- `object_position` - Not tracking object positions
- `last_detection_time` - Not timing detections
- `last_search_action_time` - Not in search mode
- `is_rotating` - Not using rotate-pause pattern
- `search_rotate_time` / `search_pause_time` - Not searching

#### Added:
- `obstacle_detected` (bool) - Whether obstacle is ahead
- `current_location_description` (str) - VLM's description of current location
- `current_direction` (int) - Current movement direction command
- `last_direction_change` (Time) - When direction was last changed
- `available_directions` (list) - Possible movement commands

### 4. VLM Prompt Changed

#### Old Prompt (Object Detection):
```
List the unique objects you see in this image. If [target_object] is in the list, 
respond with JSON: {"target": "[target_object]", "detected": true, ...}
```

#### New Prompt (Exploration):
```
You are exploring. Describe where you likely are (kitchen, hallway, bedroom, etc.) 
and if there is an obstacle directly ahead. Respond with JSON:
{
  "location": "kitchen",
  "obstacle_ahead": true,
  "description": "I see a refrigerator and cabinets, likely in a kitchen"
}
```

### 5. Control Logic Completely Rewritten

#### Old Behavior (Navigation):
1. Search for target object by rotating
2. If object detected, center on it
3. Move forward if centered
4. Stop when close enough
5. Safety stop if object lost

#### New Behavior (Exploration):
1. Query VLM every 3 seconds for location + obstacles
2. If obstacle detected → turn left or right randomly
3. Every `explore_duration` seconds → pick new random direction:
   - 70% chance: Move forward
   - 15% chance: Turn left
   - 15% chance: Turn right
4. Continue in current direction between changes
5. Log location descriptions as narration

---

## Command Distribution

The explorer uses a weighted random approach:
- **70% FORWARD**: Prefers exploration over spinning
- **15% TURN_LEFT**: Occasional direction change
- **15% TURN_RIGHT**: Occasional direction change
- **Obstacle override**: If obstacle detected, immediately turn (50/50 left/right)

---

## Usage

### Launch File
```bash
ros2 launch jetbot_vlm_nav vlm_navigator.launch.py
```

### Parameters (via launch file or CLI)
```bash
ros2 run jetbot_vlm_nav vlm_navigator_node \
  --ros-args \
  -p explore_duration:=2.0 \
  -p query_interval:=3.0 \
  -p obstacle_keyword:=obstacle
```

### Expected Behavior
The robot will:
1. Start moving forward
2. Every 3 seconds, VLM analyzes the scene
3. Logs location narration: `🚶 Exploring forward through kitchen`
4. Changes direction randomly every 2 seconds
5. Immediately turns if obstacle detected: `🚧 Obstacle detected! Turning left`

---

## VLM Response Format

The VLM must return valid JSON:
```json
{
  "location": "kitchen",
  "obstacle_ahead": true,
  "description": "I see a refrigerator and cabinets ahead"
}
```

If parsing fails, exploration continues with last known state.

---

## Safety Features

1. **Auto-Stop**: JetBot driver stops motors after 0.2s (handled by driver node)
2. **Obstacle Avoidance**: VLM detects obstacles → immediate turn
3. **Sequential Processing**: VLM must complete before navigation acts
4. **Latest Image Only**: Only processes most recent camera frame
5. **Direction Limits**: Only uses safe commands (forward, left, right)

---

## Performance

- **VLM Query**: ~1.3s per inference (RTX 4060)
- **GPU Memory**: ~4 GB (4-bit quantization)
- **Query Rate**: Every 3 seconds (adjustable)
- **Direction Change**: Every 2 seconds (adjustable)
- **Image Size**: 336×336 (LLaVA optimal)

---

## Troubleshooting

### Robot spinning in place
- Increase `explore_duration` (more time moving forward)
- Decrease turn probabilities in code (currently 15% each)

### Missing obstacles
- Decrease `query_interval` (more frequent VLM checks)
- Adjust prompt to emphasize obstacle detection
- Check VLM's JSON responses in logs

### Boring narration
- Increase `max_new_tokens` (longer descriptions)
- Modify prompt to ask for more descriptive language
- Check that `description` field is being logged

### Too slow
- Increase `query_interval` (less frequent VLM calls)
- Reduce `max_new_tokens` (faster inference)
- Consider smaller VLM model (llava-1.5-7b)

---

## Future Enhancements

Possible improvements:
1. **Memory**: Remember visited locations, prefer unexplored areas
2. **Goals**: "Explore until you find the kitchen"
3. **Mapping**: Build semantic map of environment
4. **Object counting**: Track unique objects seen
5. **Path efficiency**: Avoid backtracking
6. **Multi-modal**: Add audio narration with TTS
7. **Social**: Detect and interact with people

---

## Technical Details

### Dependencies
- ROS 2 (Humble/Iron)
- PyTorch + CUDA 11.8
- Transformers ≥4.36
- LLaVA-1.6-Mistral-7B-HF
- cv_bridge, OpenCV

### Topics
- **Subscribe**: `/camera/color/image_raw` (sensor_msgs/Image)
- **Publish**: `/jetbot/cmd` (std_msgs/Int32)

### Command Mapping
```python
CMD_STOP = 0
CMD_FORWARD = 1
CMD_BACKWARD = 2
CMD_TURN_LEFT = 3
CMD_TURN_RIGHT = 4
CMD_SEARCH = 5  # Not used in exploration mode
```

---

Generated: 2025-12-11  
Last Updated: Exploration mode transformation complete
