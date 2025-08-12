#!/usr/bin/env python3
"""
Script to check model file integrity and properties
"""

import os
import torch
import struct

def check_model_file(filepath):
    """Check if a model file exists and can be loaded"""
    print(f"\n--- Checking model file: {filepath} ---")
    
    if not os.path.exists(filepath):
        print(f"✗ File does not exist: {filepath}")
        return False
    
    # Check file size
    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
    if file_size == 0:
        print("✗ File is empty")
        return False
    
    # Try to read the file header
    try:
        with open(filepath, 'rb') as f:
            # Read first few bytes to check if it's a valid PyTorch file
            header = f.read(8)
            if len(header) >= 8:
                # Check for PyTorch magic number
                magic = struct.unpack('<Q', header)[0]
                print(f"File header (hex): {header.hex()}")
                print(f"Magic number: {magic}")
                
                # PyTorch files typically start with specific magic numbers
                if magic in [0x1950a86a, 0x1950a86b, 0x1950a86c]:
                    print("✓ Valid PyTorch file header detected")
                else:
                    print("⚠ Unknown file header - may not be a PyTorch file")
            else:
                print("⚠ File too small to read header")
    except Exception as e:
        print(f"✗ Error reading file header: {e}")
        return False
    
    # Try to load with torch.load
    try:
        print("Attempting to load with torch.load...")
        # Try with CPU first
        state_dict = torch.load(filepath, map_location='cpu')
        print("✓ Successfully loaded with torch.load (CPU)")
        
        # Check if it's a state dict
        if isinstance(state_dict, dict):
            print(f"✓ File contains state dict with {len(state_dict)} keys")
            # Show first few keys
            keys = list(state_dict.keys())[:5]
            print(f"Sample keys: {keys}")
        else:
            print(f"⚠ File contains: {type(state_dict)}")
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to load with torch.load: {e}")
        
        # Try alternative loading methods
        try:
            print("Trying with pickle...")
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Successfully loaded with pickle: {type(data)}")
            return True
        except Exception as e2:
            print(f"✗ Failed to load with pickle: {e2}")
            
        return False

def main():
    """Check all model files"""
    print("Model File Integrity Check")
    print("=" * 50)
    
    model_files = [
        "final_model (11).pth",
        "final_model_tarsus_improved.pth", 
        "final_model_tarsus.pth",
        "final_model.pth"
    ]
    
    results = {}
    for model_file in model_files:
        results[model_file] = check_model_file(model_file)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    for model_file, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {model_file}")
    
    # Check PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

if __name__ == "__main__":
    main() 