#!/usr/bin/env python3

"""
VLM Detector Node for JetBot - Debug/Testing Version
Only detects and reports objects without navigation
Optimized for NVIDIA RTX 4060
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
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


class VLMDetector(Node):
    """
    ROS 2 Node that uses on-device VLM to detect objects (detection only, no navigation)
    """
    
    def __init__(self):
        super().__init__('vlm_detector')
        
        # Declare parameters
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('model_name', 'llava-hf/llava-v1.6-mistral-7b-hf')
        self.declare_parameter('use_4bit', True)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('target_object', 'bottle')
        self.declare_parameter('query_interval', 2.0)  # Slower for debug
        self.declare_parameter('image_width', 336)
        self.declare_parameter('image_height', 336)
        self.declare_parameter('max_new_tokens', 200)
        self.declare_parameter('verbose', True)  # Print detailed info
        
        # Get parameters
        self.camera_topic = self.get_parameter('camera_topic').value
        self.model_name = self.get_parameter('model_name').value
        self.use_4bit = self.get_parameter('use_4bit').value
        self.device = self.get_parameter('device').value
        self.target_object = self.get_parameter('target_object').value
        self.query_interval = self.get_parameter('query_interval').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.max_new_tokens = self.get_parameter('max_new_tokens').value
        self.verbose = self.get_parameter('verbose').value
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # State variables
        self.latest_image: Optional[np.ndarray] = None
        self.is_processing = False
        self.detection_count = 0
        
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
            10
        )
        
        # Publishers
        self.detection_pub = self.create_publisher(String, '/vlm_detector/detection', 10)
        self.status_pub = self.create_publisher(String, '/vlm_detector/status', 10)
        
        # Timer for VLM queries
        self.query_timer = self.create_timer(self.query_interval, self.query_vlm)
        
        self.get_logger().info(f'VLM Detector initialized. Looking for: {self.target_object}')
        self.get_logger().info(f'Using on-device model: {self.model_name}')
        self.get_logger().info('Publishing detections to /vlm_detector/detection')
    
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
        """Callback for camera images"""
        try:
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
        """Query the on-device VLM to detect the target object"""
        if self.latest_image is None:
            self.get_logger().warn('No image available yet')
            return
        
        if self.is_processing:
            self.get_logger().debug('Still processing previous query')
            return
        
        if self.model is None:
            self.get_logger().error('VLM model not loaded')
            return
        
        self.is_processing = True
        self.detection_count += 1
        
        if self.verbose:
            self.get_logger().info(f'═══ Detection #{self.detection_count} ═══')
        
        try:
            # Prepare image
            pil_image = self.prepare_image(self.latest_image)
            
            # Create prompt for on-device VLM
            prompt = f"""[INST] <image>
You are a robot vision system. Analyze this image and detect if there is a {self.target_object} visible.

If you find a {self.target_object}, respond with ONLY a JSON object:
{{"detected": true, "position": {{"x": 0.5, "y": 0.5}}, "size": 0.3, "confidence": 0.9, "description": "brief description"}}

Where:
- x is horizontal position (0=left, 1=right)
- y is vertical position (0=top, 1=bottom)
- size is fraction of image (0.0 to 1.0)
- confidence is your certainty (0.0 to 1.0)
- description is what you see

If no {self.target_object} found, respond:
{{"detected": false, "scene": "brief description of what you see"}}

Respond ONLY with the JSON, nothing else. [/INST]"""
            
            # Process with model
            response = self._query_on_device_vlm(prompt, pil_image)
            
            # Parse response
            self._process_vlm_response(response)
            
        except Exception as e:
            self.get_logger().error(f'Error querying VLM: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            self.is_processing = False
    
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
            
            if self.verbose:
                self.get_logger().debug(f'VLM raw response: {response}')
            return response
            
        except Exception as e:
            self.get_logger().error(f'Error in VLM inference: {str(e)}')
            return '{"detected": false}'
    
    def _process_vlm_response(self, response: str):
        """Process the VLM response and publish detection results"""
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = json.loads(response)
            
            detected = data.get('detected', False)
            
            # Create detection message
            detection_msg = String()
            detection_msg.data = json.dumps(data, indent=2)
            self.detection_pub.publish(detection_msg)
            
            if detected:
                position = data.get('position', {})
                size = data.get('size', 0.0)
                confidence = data.get('confidence', 0.0)
                description = data.get('description', '')
                
                x = position.get('x', 0.5)
                y = position.get('y', 0.5)
                
                # Log detailed detection info
                self.get_logger().info('┌─────────────────────────────────────┐')
                self.get_logger().info('│ ✓ OBJECT DETECTED                  │')
                self.get_logger().info('└─────────────────────────────────────┘')
                self.get_logger().info(f'  Object: {self.target_object}')
                self.get_logger().info(f'  Position: x={x:.2f} ({"left" if x < 0.4 else "center" if x < 0.6 else "right"}), y={y:.2f}')
                self.get_logger().info(f'  Size: {size:.2f} ({"small" if size < 0.2 else "medium" if size < 0.5 else "large"})')
                self.get_logger().info(f'  Confidence: {confidence:.2%}')
                if description:
                    self.get_logger().info(f'  Description: {description}')
                self.get_logger().info('')
                
                status_msg = String()
                status_msg.data = f'DETECTED: {self.target_object} at x={x:.2f}, size={size:.2f}, conf={confidence:.2f}'
                self.status_pub.publish(status_msg)
            else:
                scene = data.get('scene', 'No description')
                self.get_logger().info('┌─────────────────────────────────────┐')
                self.get_logger().info('│ ✗ NO DETECTION                      │')
                self.get_logger().info('└─────────────────────────────────────┘')
                self.get_logger().info(f'  Looking for: {self.target_object}')
                self.get_logger().info(f'  Scene: {scene}')
                self.get_logger().info('')
                
                status_msg = String()
                status_msg.data = f'NOT DETECTED: {self.target_object}'
                self.status_pub.publish(status_msg)
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse VLM response as JSON: {str(e)}')
            self.get_logger().error(f'Response was: {response}')
        except Exception as e:
            self.get_logger().error(f'Error processing VLM response: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    node = VLMDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
