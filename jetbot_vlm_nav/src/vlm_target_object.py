#!/usr/bin/env python3

"""
VLM Navigator Node for JetBot
Uses on-device Vision Language Models to detect objects and navigate towards them
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
from typing import Optional, Tuple
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


class VLMNavigator(Node):
    """
    ROS 2 Node that uses on-device VLM to detect objects and navigate towards them
    """
    
    def __init__(self):
        super().__init__('vlm_navigator')
        
        # Declare parameters
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('cmd_topic', '/jetbot/cmd')
        self.declare_parameter('model_name', 'llava-hf/llava-v1.6-mistral-7b-hf')  # Options: llava-1.5-7b-hf, llava-v1.6-mistral-7b-hf, Qwen/Qwen2-VL-7B-Instruct
        self.declare_parameter('use_4bit', True)  # Use 4-bit quantization to fit in GPU memory
        self.declare_parameter('device', 'cuda')  # 'cuda' or 'cpu'
        self.declare_parameter('target_object', 'bottle')
        self.declare_parameter('query_interval', 1.0)  # seconds between VLM queries
        self.declare_parameter('stop_distance_threshold', 0.35)  # relative size threshold
        self.declare_parameter('image_width', 336)  # LLaVA optimal size
        self.declare_parameter('image_height', 336)
        self.declare_parameter('max_new_tokens', 150)  # Reduced to prevent truncation
        self.declare_parameter('center_tolerance', 0.15)  # How centered object needs to be
        self.declare_parameter('safety_timeout', 5.0)  # Stop if no detection for this many seconds
        
        # Get parameters
        self.camera_topic = self.get_parameter('camera_topic').value
        self.cmd_topic = self.get_parameter('cmd_topic').value
        self.model_name = self.get_parameter('model_name').value
        self.use_4bit = self.get_parameter('use_4bit').value
        self.device = self.get_parameter('device').value
        self.target_object = self.get_parameter('target_object').value
        self.query_interval = self.get_parameter('query_interval').value
        self.stop_distance_threshold = self.get_parameter('stop_distance_threshold').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.max_new_tokens = self.get_parameter('max_new_tokens').value
        self.center_tolerance = self.get_parameter('center_tolerance').value
        self.safety_timeout = self.get_parameter('safety_timeout').value
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # State variables
        self.latest_image: Optional[np.ndarray] = None
        self.image_lock = False  # Lock to ensure we only use latest image
        self.is_processing = False
        self.vlm_completed = False  # Flag to indicate VLM has finished processing
        self.object_detected = False
        self.object_position: Optional[Tuple[float, float, float, float]] = None  # (x, y, width, height)
        self.last_detection_time = None  # Will be set on first detection
        self.current_command = "SEARCH"  # Track current command for safety
        
        # Search pattern state
        self.search_rotate_time = 0.1  # Rotate for 0.1 seconds (very brief)
        self.search_pause_time = 3.0   # Pause for 3.0 seconds to scan
        self.last_search_action_time = self.get_clock().now()
        self.is_rotating = True  # Start with rotation
        
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
        
        self.get_logger().info(f'VLM Navigator initialized. Looking for: {self.target_object}')
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
            prompt = f"""[INST] <image>
You are a robot vision system. Look at this image and list all the objects you can see.

Respond with ONLY a JSON array of object names:

["object1", "object2", "object3"]

List unique object types you see in the image. [/INST]"""
            
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
        """Process the VLM response and update object detection state"""
        try:
            # Extract JSON array from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                objects_list = json.loads(json_str)
            else:
                objects_list = json.loads(response)
            
            # Check if objects_list is actually a list
            if not isinstance(objects_list, list):
                self.get_logger().error(f'Expected list but got: {type(objects_list)}')
                objects_list = []
            
            # Convert all objects to lowercase for case-insensitive matching
            objects_lower = [str(obj).lower() for obj in objects_list]
            target_lower = self.target_object.lower()
            
            # Log detected objects
            if objects_list:
                objects_str = ', '.join(objects_list)
                self.get_logger().info(f'Objects detected: [{objects_str}]')
            else:
                self.get_logger().info('No objects detected in scene')
            
            # Check if target is in the list
            self.object_detected = target_lower in objects_lower
            
            if self.object_detected:
                # Target found in list - set default position (centered, moderate size)
                x = 0.5  # Center horizontally
                y = 0.5  # Center vertically
                size = 0.2  # Default size
                confidence = 1.0  # High confidence since it's in the list
                
                self.object_position = (x, y, size, confidence)
                self.last_detection_time = self.get_clock().now()
                
                status_msg = String()
                status_msg.data = f'✓ TARGET FOUND: {self.target_object} is in view | Objects: {objects_str}'
                self.status_pub.publish(status_msg)
                self.get_logger().info(status_msg.data)
            else:
                self.object_position = None
                objects_str = ', '.join(objects_list) if objects_list else 'none'
                status_msg = String()
                status_msg.data = f'✗ Target "{self.target_object}" NOT in list | Objects: {objects_str}'
                self.status_pub.publish(status_msg)
                self.get_logger().info(status_msg.data)
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse VLM response as JSON: {str(e)}')
            self.get_logger().error(f'Response was: {response}')
        except Exception as e:
            self.get_logger().error(f'Error processing VLM response: {str(e)}')
    
    def control_loop(self):
        """Control loop that triggers VLM queries and acts only after completion"""
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
            # Still processing VLM, maintain last command or stop
            return
        
        cmd_msg = Int32()
        
        # If no object detected - search mode (rotate to find it)
        if not self.object_detected or self.object_position is None:
            # Check if we've been searching too long without any detection
            if self.last_detection_time is not None:
                time_since_detection = (self.get_clock().now() - self.last_detection_time).nanoseconds / 1e9
                
                # Only safety stop if we had a detection before and lost it for too long
                if time_since_detection > self.safety_timeout:
                    cmd_msg.data = self.CMD_STOP
                    if self.current_command != "SAFETY_STOP":
                        self.get_logger().warn(f'SAFETY STOP: Lost object for {time_since_detection:.1f}s')
                        self.current_command = "SAFETY_STOP"
                    self.cmd_pub.publish(cmd_msg)
                    return
            
            # Search mode - rotate and pause pattern for better scanning
            time_elapsed = (self.get_clock().now() - self.last_search_action_time).nanoseconds / 1e9
            
            if self.is_rotating:
                # Currently rotating
                if time_elapsed < self.search_rotate_time:
                    cmd_msg.data = self.CMD_TURN_LEFT
                    if self.current_command != "SEARCH_ROTATE":
                        self.get_logger().info('Command: TURN_LEFT (searching)')
                        self.current_command = "SEARCH_ROTATE"
                else:
                    # Switch to pause
                    self.is_rotating = False
                    self.last_search_action_time = self.get_clock().now()
                    cmd_msg.data = self.CMD_STOP
                    self.get_logger().info('Command: STOP (pausing to scan)')
                    self.current_command = "SEARCH_PAUSE"
            else:
                # Currently paused
                if time_elapsed < self.search_pause_time:
                    cmd_msg.data = self.CMD_STOP
                else:
                    # Switch back to rotating
                    self.is_rotating = True
                    self.last_search_action_time = self.get_clock().now()
                    cmd_msg.data = self.CMD_TURN_LEFT
                    self.current_command = "SEARCH_ROTATE"
            
            self.cmd_pub.publish(cmd_msg)
            return
        
        x, y, size, confidence = self.object_position
        
        # Check if object is close enough (large size means close)
        if size > self.stop_distance_threshold:
            # STOP - we're close enough
            cmd_msg.data = self.CMD_STOP
            if self.current_command != "STOP":
                self.get_logger().info('Command: STOP (reached target!)')
                self.current_command = "STOP"
            self.cmd_pub.publish(cmd_msg)
            return
        
        # Determine command based on object position
        # x is horizontal position: 0.5 is center, <0.5 is left, >0.5 is right
        center_offset = x - 0.5
        
        if abs(center_offset) < self.center_tolerance:
            # Object is centered - FORWARD
            cmd_msg.data = self.CMD_FORWARD
            if self.current_command != "FORWARD":
                self.get_logger().info(f'Command: FORWARD (object centered, size={size:.2f})')
                self.current_command = "FORWARD"
        elif center_offset < 0:
            # Object on left - TURN LEFT
            cmd_msg.data = self.CMD_TURN_LEFT
            if self.current_command != "LEFT":
                self.get_logger().info(f'Command: LEFT (object at x={x:.2f})')
                self.current_command = "LEFT"
        else:
            # Object on right - TURN RIGHT
            cmd_msg.data = self.CMD_TURN_RIGHT
            if self.current_command != "RIGHT":
                self.get_logger().info(f'Command: RIGHT (object at x={x:.2f})')
                self.current_command = "RIGHT"
        
        self.cmd_pub.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = VLMNavigator()
    
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
