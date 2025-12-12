#!/usr/bin/env python3

"""
Test script to verify VLM Navigator setup
Checks dependencies and GPU availability
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name} installed")
        return True
    except ImportError:
        print(f"✗ {package_name} NOT installed")
        return False

def main():
    print("=" * 50)
    print("JetBot VLM Navigator - Dependency Check")
    print("=" * 50)
    print()
    
    all_ok = True
    
    # Check ROS 2
    print("Checking ROS 2 dependencies...")
    all_ok &= check_import('rclpy', 'rclpy')
    all_ok &= check_import('cv_bridge', 'cv_bridge')
    print()
    
    # Check Python packages
    print("Checking Python dependencies...")
    all_ok &= check_import('cv2', 'opencv-python')
    all_ok &= check_import('numpy', 'numpy')
    all_ok &= check_import('PIL', 'Pillow')
    all_ok &= check_import('torch', 'PyTorch')
    all_ok &= check_import('transformers', 'transformers')
    all_ok &= check_import('accelerate', 'accelerate')
    all_ok &= check_import('bitsandbytes', 'bitsandbytes')
    print()
    
    # Check CUDA
    print("Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU Memory: {memory_gb:.1f} GB")
        else:
            print("✗ CUDA NOT available (will use CPU - much slower)")
            all_ok = False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        all_ok = False
    print()
    
    # Check transformers models
    print("Checking transformers library...")
    try:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        print("✓ LLaVA models supported")
    except ImportError:
        print("✗ LLaVA models NOT supported (upgrade transformers: pip install transformers>=4.36)")
        all_ok = False
    print()
    
    # Summary
    print("=" * 50)
    if all_ok:
        print("✓ All checks passed! System ready.")
        print()
        print("Next steps:")
        print("1. Run: source install/setup.bash")
        print("2. Launch: ros2 launch jetbot_vlm_nav vlm_navigator.launch.py")
        print()
        print("Note: First run will download ~13GB model (10-30 min)")
    else:
        print("✗ Some checks failed. Please install missing dependencies.")
        print()
        print("To install dependencies:")
        print("  cd /home/kalhan/Projects/Real_Experiments/jetbot_ws/src/jetbot_vlm_nav")
        print("  pip install -r requirements.txt")
        return 1
    print("=" * 50)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
