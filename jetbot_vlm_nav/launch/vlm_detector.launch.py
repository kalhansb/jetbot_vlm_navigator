from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Launch file for VLM Detector (debug/testing node)"""
    
    # Declare launch arguments
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/color/image_raw',
        description='Camera image topic to subscribe to'
    )
    
    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value='llava-hf/llava-v1.6-mistral-7b-hf',
        description='VLM model name from HuggingFace'
    )
    
    target_object_arg = DeclareLaunchArgument(
        'target_object',
        default_value='bottle',
        description='Object to search for'
    )
    
    use_4bit_arg = DeclareLaunchArgument(
        'use_4bit',
        default_value='true',
        description='Use 4-bit quantization for GPU efficiency'
    )
    
    query_interval_arg = DeclareLaunchArgument(
        'query_interval',
        default_value='2.0',
        description='Time interval between VLM queries (seconds)'
    )
    
    verbose_arg = DeclareLaunchArgument(
        'verbose',
        default_value='true',
        description='Print detailed detection information'
    )
    
    # VLM Detector node
    vlm_detector_node = Node(
        package='jetbot_vlm_nav',
        executable='vlm_detector_node.py',
        name='vlm_detector',
        output='screen',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'model_name': LaunchConfiguration('model_name'),
            'target_object': LaunchConfiguration('target_object'),
            'use_4bit': LaunchConfiguration('use_4bit'),
            'query_interval': LaunchConfiguration('query_interval'),
            'verbose': LaunchConfiguration('verbose'),
            'device': 'cuda',  # Use GPU
            'image_width': 336,
            'image_height': 336,
            'max_new_tokens': 200
        }]
    )
    
    return LaunchDescription([
        camera_topic_arg,
        model_name_arg,
        target_object_arg,
        use_4bit_arg,
        query_interval_arg,
        verbose_arg,
        vlm_detector_node
    ])
