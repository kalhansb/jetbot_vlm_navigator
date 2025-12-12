from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch file for VLM Navigator"""
    
    # Get package directory
    pkg_dir = get_package_share_directory('jetbot_vlm_nav')
    
    # Declare launch arguments
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/color/image_raw',
        description='Camera image topic to subscribe to'
    )
    
    cmd_vel_topic_arg = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value='/cmd_vel',
        description='Command velocity topic to publish to'
    )
    
    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value='llava-hf/llava-v1.6-mistral-7b-hf',
        description='VLM model name from HuggingFace'
    )
    
    target_object_arg = DeclareLaunchArgument(
        'target_object',
        default_value='bottle',
        description='Object to search for and navigate towards'
    )
    
    use_4bit_arg = DeclareLaunchArgument(
        'use_4bit',
        default_value='true',
        description='Use 4-bit quantization for GPU efficiency'
    )
    
    query_interval_arg = DeclareLaunchArgument(
        'query_interval',
        default_value='1.0',
        description='Time interval between VLM queries (seconds)'
    )
    
    linear_speed_arg = DeclareLaunchArgument(
        'linear_speed',
        default_value='0.2',
        description='Maximum linear velocity'
    )
    
    angular_speed_arg = DeclareLaunchArgument(
        'angular_speed',
        default_value='0.5',
        description='Maximum angular velocity'
    )
    
    center_tolerance_arg = DeclareLaunchArgument(
        'center_tolerance',
        default_value='0.15',
        description='How centered object needs to be for FORWARD command'
    )
    
    safety_timeout_arg = DeclareLaunchArgument(
        'safety_timeout',
        default_value='5.0',
        description='Stop if no detection for this many seconds'
    )
    
    # VLM Navigator node
    vlm_navigator_node = Node(
        package='jetbot_vlm_nav',
        executable='vlm_navigator_node.py',
        name='vlm_navigator',
        output='screen',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            'model_name': LaunchConfiguration('model_name'),
            'target_object': LaunchConfiguration('target_object'),
            'use_4bit': LaunchConfiguration('use_4bit'),
            'query_interval': LaunchConfiguration('query_interval'),
            'linear_speed': LaunchConfiguration('linear_speed'),
            'angular_speed': LaunchConfiguration('angular_speed'),
            'center_tolerance': LaunchConfiguration('center_tolerance'),
            'safety_timeout': LaunchConfiguration('safety_timeout'),
            'device': 'cuda',  # Use GPU
            'image_width': 336,
            'image_height': 336,
            'max_new_tokens': 200,
            'stop_distance_threshold': 0.35
        }]
    )
    
    return LaunchDescription([
        camera_topic_arg,
        cmd_vel_topic_arg,
        model_name_arg,
        target_object_arg,
        use_4bit_arg,
        query_interval_arg,
        linear_speed_arg,
        angular_speed_arg,
        center_tolerance_arg,
        safety_timeout_arg,
        vlm_navigator_node
    ])
