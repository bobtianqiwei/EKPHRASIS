#!/usr/bin/env python3
"""
Test script for the EKPHRASIS ML system
"""

import requests
import json
import base64
from PIL import Image, ImageDraw
import io

def create_test_image():
    """Create a simple test image with rectangles"""
    # Create a 300x300 white image
    img = Image.new('RGB', (300, 300), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw some rectangles
    draw.rectangle([50, 50, 150, 150], fill='#9C9C9C')
    draw.rectangle([200, 100, 250, 200], fill='#535353')
    draw.rectangle([100, 200, 200, 250], fill='#323131')
    
    return img

def image_to_base64(img):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def test_server_health():
    """Test if the server is running"""
    try:
        response = requests.get('http://localhost:5000/health')
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server is running")
            print(f"  Status: {data.get('status')}")
            print(f"  Model loaded: {data.get('model_loaded')}")
            return data.get('model_loaded', False)
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to server. Make sure it's running on http://localhost:5000")
        return False

def test_single_prediction():
    """Test single image prediction"""
    try:
        # Create test image
        test_img = create_test_image()
        img_base64 = image_to_base64(test_img)
        
        # Send prediction request
        response = requests.post('http://localhost:5000/predict', 
                               json={'image': img_base64})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Single prediction successful")
            print(f"  Confidence: {result.get('confidence', 0):.4f}")
            print(f"  Class: {result.get('class', 'unknown')}")
            return True
        else:
            print(f"✗ Single prediction failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Single prediction error: {e}")
        return False

def test_multiple_predictions():
    """Test multiple image predictions"""
    try:
        # Create multiple test images
        test_images = []
        for i in range(5):
            test_img = create_test_image()
            img_base64 = image_to_base64(test_img)
            test_images.append(img_base64)
        
        # Send multiple prediction request
        response = requests.post('http://localhost:5000/predict_multiple', 
                               json={'images': test_images})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Multiple predictions successful")
            print(f"  Best score: {result['best']['confidence']:.4f} (index {result['best']['index']})")
            print(f"  Worst score: {result['worst']['confidence']:.4f} (index {result['worst']['index']})")
            return True
        else:
            print(f"✗ Multiple predictions failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Multiple predictions error: {e}")
        return False

def main():
    print("EKPHRASIS ML System Test")
    print("=" * 30)
    
    # Test 1: Server health
    print("\n1. Testing server health...")
    model_loaded = test_server_health()
    
    if not model_loaded:
        print("\n⚠️  Model not loaded. Please train the model first:")
        print("   python train_and_save_model.py")
        return
    
    # Test 2: Single prediction
    print("\n2. Testing single image prediction...")
    single_ok = test_single_prediction()
    
    # Test 3: Multiple predictions
    print("\n3. Testing multiple image predictions...")
    multiple_ok = test_multiple_predictions()
    
    # Summary
    print("\n" + "=" * 30)
    print("Test Summary:")
    print(f"  Server Health: {'✓' if model_loaded else '✗'}")
    print(f"  Single Prediction: {'✓' if single_ok else '✗'}")
    print(f"  Multiple Predictions: {'✓' if multiple_ok else '✗'}")
    
    if model_loaded and single_ok and multiple_ok:
        print("\n🎉 All tests passed! The ML system is working correctly.")
        print("You can now use the frontend interface.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 