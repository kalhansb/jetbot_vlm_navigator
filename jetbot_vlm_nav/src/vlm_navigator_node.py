#!/usr/bin/env python3

"""
VLM Explorer Node for JetBot
Uses on-device Vision Language Models to explore environment while narrating location
Optimized for NVIDIA RTX 4060
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import os
import random
from typing import Optional
import torch
from PIL import Image as PILImage
import io

# On-device VLM support
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("transformers not available. Install with: pip install transformers")

try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False
    print("LLaVA models require transformers>=4.36")


class VLMExplorer(Node):
    """
    ROS 2 Node that uses on-device VLM to explore and narrate environment
    """
    
    # Command constants (matching jetbot_driver_code.py)
    CMD_STOP = 0
    CMD_FORWARD = 1
    CMD_BACKWARD = 2
    CMD_TURN_LEFT = 3
    CMD_TURN_RIGHT = 4
    CMD_SEARCH = 5
    
    def __init__(self):
        super().__init__('vlm_explorer')
        
        # Declare parameters
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('cmd_topic', '/jetbot/cmd')
        self.declare_parameter('model_name', 'llava-hf/llava-v1.6-mistral-7b-hf')  # Options: llava-1.5-7b-hf, llava-v1.6-mistral-7b-hf, Qwen/Qwen2-VL-7B-Instruct
        self.declare_parameter('use_4bit', True)  # Use 4-bit quantization to fit in GPU memory
        self.declare_parameter('device', 'cuda')  # 'cuda' or 'cpu'
        self.declare_parameter('query_interval', 3.0)  # seconds between VLM queries for narration
        self.declare_parameter('image_width', 336)  # LLaVA optimal size
        self.declare_parameter('image_height', 336)
        self.declare_parameter('max_new_tokens', 200)  # For descriptive narration
        self.declare_parameter('explore_duration', 2.0)  # Time to move in one direction
        self.declare_parameter('obstacle_keyword', 'obstacle')  # Keyword to detect obstacles
        
        # Get parameters
        self.camera_topic = self.get_parameter('camera_topic').value
        self.cmd_topic = self.get_parameter('cmd_topic').value
        self.model_name = self.get_parameter('model_name').value
        self.use_4bit = self.get_parameter('use_4bit').value
        self.device = self.get_parameter('device').value
        self.query_interval = self.get_parameter('query_interval').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.max_new_tokens = self.get_parameter('max_new_tokens').value
        self.explore_duration = self.get_parameter('explore_duration').value
        self.obstacle_keyword = self.get_parameter('obstacle_keyword').value
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # State variables
        self.latest_image: Optional[np.ndarray] = None
        self.image_lock = False  # Lock to ensure we only use latest image
        self.is_processing = False
        self.vlm_completed = False  # Flag to indicate VLM has finished processing
        self.obstacle_detected = False
        self.current_location_description = "unknown location"
        
        # Exploration state
        self.current_direction = None  # Current movement direction
        self.last_direction_change = self.get_clock().now()
        self.available_directions = [self.CMD_FORWARD, self.CMD_TURN_LEFT, self.CMD_TURN_RIGHT]
        
        # Check GPU availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA not available, falling back to CPU')
            self.device = 'cpu'
        
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.get_logger().info(f'Using GPU: {gpu_name} with {gpu_memory:.2f} GB memory')
        
        # Initialize VLM model
        self.model = None
        self.processor = None
        self._init_vlm_model()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            1
        )
        
        # Publishers
        self.cmd_pub = self.create_publisher(Int32, self.cmd_topic, 10)
        self.status_pub = self.create_publisher(String, '/vlm_nav/status', 10)
        
        # Command mapping: 0=STOP, 1=FORWARD, 2=BACKWARD, 3=TURN_LEFT, 4=TURN_RIGHT, 5=SEARCH
        self.CMD_STOP = 0
        self.CMD_FORWARD = 1
        self.CMD_BACKWARD = 2
        self.CMD_TURN_LEFT = 3
        self.CMD_TURN_RIGHT = 4
        self.CMD_SEARCH = 5
        
        # Timer for control loop - this will trigger VLM queries when ready
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        # Track last query time for interval control
        self.last_query_time = self.get_clock().now()
        
        self.get_logger().info('VLM Explorer initialized. Ready to explore and narrate!')
        self.get_logger().info(f'Using on-device model: {self.model_name}')
    
    def _init_vlm_model(self):
        """Initialize the on-device VLM model"""
        if not TRANSFORMERS_AVAILABLE:
            self.get_logger().error('transformers library not available. Install with: pip install transformers accelerate bitsandbytes')
            return
        
        self.get_logger().info(f'Loading VLM model: {self.model_name}...')
        self.get_logger().info('This may take a few minutes on first run...')
        
        try:
            # Configure quantization for RTX 4060 (8GB VRAM)
            if self.use_4bit and self.device == 'cuda':
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.get_logger().info('Using 4-bit quantization for efficient GPU usage')
            else:
                quantization_config = None
            
            # Load processor and model
            if 'llava' in self.model_name.lower():
                self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == 'cuda' else None,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    low_cpu_mem_usage=True
                )
            else:
                # Generic vision model support
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == 'cuda' else None,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    low_cpu_mem_usage=True
                )
            
            if self.device == 'cpu':
                self.model = self.model.to('cpu')
            
            self.model.eval()
            
            # Print success message with model location
            import os
            cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
            self.get_logger().info('=' * 60)
            self.get_logger().info('✓ VLM model loaded successfully!')
            self.get_logger().info(f'  Model: {self.model_name}')
            self.get_logger().info(f'  Device: {self.device}')
            self.get_logger().info(f'  Cache location: {cache_dir}')
            if self.device == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                self.get_logger().info(f'  GPU Memory Used: {memory_allocated:.2f} GB')
            self.get_logger().info('=' * 60)
            
        except Exception as e:
            self.get_logger().error(f'Failed to load VLM model: {str(e)}')
            self.get_logger().error('Make sure you have installed: pip install transformers accelerate bitsandbytes')
            self.model = None
    
    def image_callback(self, msg: Image):
        """Callback for camera images - always keep only the latest"""
        try:
            # Only update if not currently processing
            # This ensures we always have the freshest image when VLM starts
            if not self.image_lock:
                # Convert ROS Image to OpenCV format (BGR)
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting image: {str(e)}')
    
    def prepare_image(self, image: np.ndarray) -> PILImage.Image:
        """Prepare OpenCV image for VLM processing"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to optimal size for model
        height, width = image_rgb.shape[:2]
        if width != self.image_width or height != self.image_height:
            image_rgb = cv2.resize(image_rgb, (self.image_width, self.image_height))
        
        # Convert to PIL Image
        pil_image = PILImage.fromarray(image_rgb)
        return pil_image
    
    def query_vlm(self):
        """Query the on-device VLM to detect the target object using latest image only"""
        if self.latest_image is None:
            self.get_logger().warn('No image available yet')
            return False
        
        if self.is_processing:
            self.get_logger().debug('Still processing previous query')
            return False
        
        if self.model is None:
            self.get_logger().error('VLM model not loaded')
            return False
        
        # Lock image updates and mark as processing
        self.image_lock = True
        self.is_processing = True
        self.vlm_completed = False
        
        try:
            # Capture the current latest image (no queuing, just current frame)
            current_image = self.latest_image.copy()
            
            # Prepare image
            pil_image = self.prepare_image(current_image)
            
            # Create prompt for on-device VLM
            prompt = """[INST] <image>
You are a robot exploring an environment. Describe what you see and answer:
1. What location or room type is this likely? (kitchen, hallway, office, etc.)
2. Is there an obstacle directly in front that would block forward movement?

Respond with ONLY a JSON object:

{"location": "kitchen", "obstacle_ahead": true, "description": "I see a kitchen with a counter very close in front of me"}

or

{"location": "hallway", "obstacle_ahead": false, "description": "I see a clear hallway ahead"}

IMPORTANT: Only set obstacle_ahead to true if an object is VERY CLOSE (within 1-2 feet) and blocking forward movement. Distant objects or objects far away should have obstacle_ahead set to false. [/INST]"""
            
            # Process with model
            response = self._query_on_device_vlm(prompt, pil_image)
            
            # Parse response
            self._process_vlm_response(response)
            
            # Mark VLM processing as completed
            self.vlm_completed = True
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error querying VLM: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.vlm_completed = True  # Mark completed even on error
            return False
        finally:
            self.is_processing = False
            self.image_lock = False  # Unlock to receive new images
    
    def _query_on_device_vlm(self, prompt: str, image: PILImage.Image) -> str:
        """Query on-device VLM model"""
        try:
            # Prepare inputs - note: LLaVA expects text first, then image
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            # Move to device
            if self.device == 'cuda':
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Generate response with suppressed warnings
            import warnings
            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*pad_token_id.*")
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )
            
            # Decode response
            generated_text = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract only the assistant's response (after the prompt)
            # LLaVA models often include the prompt in output
            if '[/INST]' in generated_text:
                response = generated_text.split('[/INST]')[-1].strip()
            else:
                response = generated_text.strip()
            
            # Log VLM response
            self.get_logger().info('─' * 60)
            self.get_logger().info('VLM Response:')
            self.get_logger().info(f'  {response}')
            self.get_logger().info('─' * 60)
            return response
            
        except Exception as e:
            self.get_logger().error(f'Error in VLM inference: {str(e)}')
            return '{"detected": false}'
    
    def _process_vlm_response(self, response: str):
        """Process the VLM response for exploration and obstacle detection"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = json.loads(response)
            
            # Extract location and obstacle info
            location = data.get('location', 'unknown')
            obstacle_ahead = data.get('obstacle_ahead', False)
            description = data.get('description', 'Unknown environment')
            
            self.current_location_description = location
            self.obstacle_detected = obstacle_ahead
            
            # Narrate what we see
            status_msg = String()
            if obstacle_ahead:
                status_msg.data = f'🚧 OBSTACLE AHEAD | Location: {location} | {description}'
                self.get_logger().warn(status_msg.data)
            else:
                status_msg.data = f'✓ PATH CLEAR | Location: {location} | {description}'
                self.get_logger().info(status_msg.data)
            
            self.status_pub.publish(status_msg)
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse VLM response as JSON: {str(e)}')
            self.get_logger().error(f'Response was: {response}')
        except Exception as e:
            self.get_logger().error(f'Error processing VLM response: {str(e)}')
    
    def control_loop(self):
        """Control loop for random exploration with obstacle avoidance"""
        # Check if it's time to query VLM
        current_time = self.get_clock().now()
        time_since_last_query = (current_time - self.last_query_time).nanoseconds / 1e9
        
        # Trigger VLM query if interval has passed and not currently processing
        if time_since_last_query >= self.query_interval and not self.is_processing:
            self.last_query_time = current_time
            self.vlm_completed = False
            self.query_vlm()
        
        # Only make navigation decisions after VLM has completed processing
        if not self.vlm_completed and self.is_processing:
            # Still processing VLM, maintain last command
            return
        
        cmd_msg = Int32()
        
        # Check if we need to change direction
        time_since_direction_change = (current_time - self.last_direction_change).nanoseconds / 1e9
        
        # If obstacle detected, change direction immediately
        if self.obstacle_detected:
            # Choose random turn direction (left or right)
            cmd_msg.data = random.choice([self.CMD_TURN_LEFT, self.CMD_TURN_RIGHT])
            self.get_logger().info(f'🚧 Obstacle detected! Turning {"left" if cmd_msg.data == self.CMD_TURN_LEFT else "right"}')
            self.last_direction_change = current_time
            self.obstacle_detected = False  # Reset until next VLM check
        
        # If enough time has passed, pick a new random direction
        elif time_since_direction_change >= self.explore_duration:
            # Randomly choose: forward (70%), turn left (15%), turn right (15%)
            rand = random.random()
            if rand < 0.7:
                cmd_msg.data = self.CMD_FORWARD
                self.get_logger().info(f'🚶 Exploring forward through {self.current_location_description}')
            elif rand < 0.85:
                cmd_msg.data = self.CMD_TURN_LEFT
                self.get_logger().info('↪ Turning left to explore new direction')
            else:
                cmd_msg.data = self.CMD_TURN_RIGHT
                self.get_logger().info('↩ Turning right to explore new direction')
            
            self.last_direction_change = current_time
            self.current_direction = cmd_msg.data
        
        # Continue in current direction
        else:
            if self.current_direction is not None:
                cmd_msg.data = self.current_direction
            else:
                # Default to forward if no direction set
                cmd_msg.data = self.CMD_FORWARD
                self.current_direction = self.CMD_FORWARD
        
        self.cmd_pub.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = VLMExplorer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot
        stop_cmd = Int32()
        stop_cmd.data = 0  # STOP command
        node.cmd_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
