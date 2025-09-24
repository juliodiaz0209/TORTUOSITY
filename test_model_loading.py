#!/usr/bin/env python3
"""
Test script to verify model loading works correctly
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
import sys

# Add current directory to path
sys.path.append('.')

from Tortuosity import load_maskrcnn_model, load_unet_model, device

def test_model_loading():
    """Test loading both models"""
    print(f"Testing model loading on device: {device}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Available files: {os.listdir('.')}")
    
    # Test Mask R-CNN loading
    try:
        print("\n--- Testing Mask R-CNN model loading ---")
        maskrcnn_model = load_maskrcnn_model("final_model (11).pth")
        print("✓ Mask R-CNN model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load Mask R-CNN model: {e}")
        # Try fallback models
        for fallback_path in ["final_model_tarsus.pth", "final_model.pth"]:
            try:
                print(f"Trying fallback: {fallback_path}")
                maskrcnn_model = load_maskrcnn_model(fallback_path)
                print(f"✓ Mask R-CNN model loaded from fallback: {fallback_path}")
                break
            except Exception as e2:
                print(f"✗ Failed to load from {fallback_path}: {e2}")
    
    # Test UNet loading (mejor modelo con attention gates)
    try:
        print("\n--- Testing UNet model loading ---")
        unet_model = load_unet_model("final_model_tarsus_improved (6).pth", device)
        print("✓ UNet model loaded successfully (with attention gates)")
    except Exception as e:
        print(f"✗ Failed to load UNet model: {e}")
        # Try fallback models
        for fallback_path in ["final_model_tarsus_improved.pth", "final_model_tarsus.pth"]:
            try:
                print(f"Trying fallback: {fallback_path}")
                unet_model = load_unet_model(fallback_path, device)
                print(f"✓ UNet model loaded from fallback: {fallback_path}")
                break
            except Exception as e2:
                print(f"✗ Failed to load from {fallback_path}: {e2}")

if __name__ == "__main__":
    test_model_loading() 