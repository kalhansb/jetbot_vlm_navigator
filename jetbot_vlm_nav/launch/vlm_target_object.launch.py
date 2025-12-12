#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('jetbot_vlm_nav')
    
    # Declare launch arguments
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/color/image_raw',
        description='Camera topic to subscribe to'
    )
    
    cmd_topic_arg = DeclareLaunchArgument(
        'cmd_topic',
        default_value='/jetbot/cmd',
        description='Command topic for robot control'
    )
    
    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value='llava-hf/llava-v1.6-mistral-7b-hf',
        description='VLM model to use (llava-hf/llava-v1.6-mistral-7b-hf, llava-hf/llava-1.5-7b-hf)'
    )
    
    target_object_arg = DeclareLaunchArgument(
        'target_object',
        default_value='bottle',
        description='Object to search for and navigate towards'
    )
    
    use_4bit_arg = DeclareLaunchArgument(
        'use_4bit',
        default_value='True',
        description='Use 4-bit quantization for GPU memory efficiency'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device to run inference on (cuda or cpu)'
    )
    
    query_interval_arg = DeclareLaunchArgument(
        'query_interval',
        default_value='1.0',
        description='Seconds between VLM queries'
    )
    
    stop_distance_arg = DeclareLaunchArgument(
        'stop_distance_threshold',
        default_value='0.35',
        description='Relative size threshold to stop (larger = closer)'
    )
    
    center_tolerance_arg = DeclareLaunchArgument(
        'center_tolerance',
        default_value='0.15',
        description='How centered object needs to be (0.0-0.5)'
    )
    
    safety_timeout_arg = DeclareLaunchArgument(
        'safety_timeout',
        default_value='5.0',
        description='Safety stop if no detection for this many seconds'
    )
    
    # VLM Target Object Navigator Node
    vlm_target_node = Node(
        package='jetbot_vlm_nav',
        executable='vlm_target_object.py',
        name='vlm_target_navigator',
        output='screen',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'cmd_topic': LaunchConfiguration('cmd_topic'),
            'model_name': LaunchConfiguration('model_name'),
            'target_object': LaunchConfiguration('target_object'),
            'use_4bit': LaunchConfiguration('use_4bit'),
            'device': LaunchConfiguration('device'),
            'query_interval': LaunchConfiguration('query_interval'),
            'stop_distance_threshold': LaunchConfiguration('stop_distance_threshold'),
            'center_tolerance': LaunchConfiguration('center_tolerance'),
            'safety_timeout': LaunchConfiguration('safety_timeout'),
            'image_width': 336,
            'image_height': 336,
            'max_new_tokens': 150,
        }],
        emulate_tty=True
    )
    
    return LaunchDescription([
        camera_topic_arg,
        cmd_topic_arg,
        model_name_arg,
        target_object_arg,
        use_4bit_arg,
        device_arg,
        query_interval_arg,
        stop_distance_arg,
        center_tolerance_arg,
        safety_timeout_arg,
        vlm_target_node,
    ])
