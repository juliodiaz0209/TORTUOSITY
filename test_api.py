#!/usr/bin/env python3
"""
Test script for the Tortuosity Analysis FastAPI
"""

import requests
import json
import time
from pathlib import Path

# Configure the API base URL
# For production, use: "https://tortuosity-backend-488176611125.us-central1.run.app"
# For local development, use: "http://localhost:8000"
API_BASE_URL = "https://tortuosity-backend-488176611125.us-central1.run.app"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Models loaded: {data['models_loaded']}")
            print(f"   Device: {data['device']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_api_info():
    """Test API info endpoint"""
    print("\n🔍 Testing API info endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api")
        if response.status_code == 200:
            data = response.json()
            print("✅ API info retrieved")
            print(f"   Message: {data['message']}")
            print(f"   Version: {data['version']}")
            return True
        else:
            print(f"❌ API info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API info error: {e}")
        return False

def test_analysis_info():
    """Test analysis info endpoint"""
    print("\n🔍 Testing analysis info endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/info")
        if response.status_code == 200:
            data = response.json()
            print("✅ Analysis info retrieved")
            print(f"   Description: {data['description']}")
            return True
        else:
            print(f"❌ Analysis info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Analysis info error: {e}")
        return False

def test_image_analysis():
    """Test image analysis endpoint"""
    print("\n🔍 Testing image analysis endpoint...")
    
    # Check if test image exists
    test_image_path = Path("meibomio.jpg")
    if not test_image_path.exists():
        print("⚠️  Test image 'meibomio.jpg' not found. Skipping analysis test.")
        return True
    
    try:
        with open(test_image_path, "rb") as f:
            files = {"file": f}
            print("📤 Uploading image for analysis...")
            start_time = time.time()
            
            response = requests.post(f"{API_BASE_URL}/analyze", files=files)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Image analysis completed successfully")
                print(f"   Processing time: {processing_time:.2f} seconds")
                print(f"   Average tortuosity: {data['data']['avg_tortuosity']}")
                print(f"   Number of glands: {data['data']['num_glands']}")
                print(f"   Individual tortuosities: {len(data['data']['individual_tortuosities'])} glands")
                return True
            else:
                print(f"❌ Image analysis failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   Response: {response.text}")
                return False
    except Exception as e:
        print(f"❌ Image analysis error: {e}")
        return False

def test_web_interface():
    """Test web interface endpoint"""
    print("\n🔍 Testing web interface endpoint...")
    try:
        response = requests.get(API_BASE_URL)
        if response.status_code == 200:
            print("✅ Web interface accessible")
            print(f"   Content-Type: {response.headers.get('content-type', 'Unknown')}")
            return True
        else:
            print(f"❌ Web interface failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Web interface error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Tortuosity Analysis FastAPI")
    print("=" * 50)
    
    tests = [
        test_health,
        test_api_info,
        test_analysis_info,
        test_web_interface,
        test_image_analysis
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the server logs for more details.")
    
    print("\n🌐 Access points:")
    print(f"   Web Interface: {API_BASE_URL}")
    print(f"   API Documentation: {API_BASE_URL}/docs")
    print(f"   Health Check: {API_BASE_URL}/health")

if __name__ == "__main__":
    main() 