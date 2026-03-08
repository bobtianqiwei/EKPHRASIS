#!/usr/bin/env python3
"""
EKPHRASIS System Launcher
Quick start script for the complete EKPHRASIS system
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True

def check_ml_dependencies():
    """Check if ML dependencies are installed"""
    try:
        import tensorflow
        import flask
        import flask_cors
        import numpy
        from PIL import Image
        print("✓ ML dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing ML dependency: {e}")
        return False

def ensure_flask_cors():
    """Install flask-cors if missing (common on fresh envs)."""
    try:
        import flask_cors
        return True
    except ImportError:
        pass
    print("Installing flask-cors...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "flask-cors"],
                      check=True, capture_output=True, text=True)
        print("✓ flask-cors installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Could not install flask-cors. Run: pip install flask-cors")
        return False

def check_model_file():
    """Check if the trained model exists"""
    model_path = Path("ml/composition_model.h5")
    if model_path.exists():
        print("✓ Trained model found")
        return True
    else:
        print("⚠️  Trained model not found")
        return False

def install_dependencies():
    """Install ML dependencies"""
    print("Installing ML dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "ml/requirements.txt"], 
                      check=True, capture_output=True)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def train_model():
    """Train the ML model"""
    print("Training the ML model...")
    print("This may take several minutes...")
    try:
        result = subprocess.run([sys.executable, "train_and_save_model.py"],
                              cwd="ml", check=True, capture_output=True, text=True)
        print("✓ Model training completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model training failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def start_server():
    """Start the ML server"""
    print("Starting ML server...")
    try:
        # Start server in background
        server_process = subprocess.Popen([sys.executable, "start_server.py"],
                                        cwd="ml", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if server is running
        if server_process.poll() is None:
            print("✓ ML server started successfully")
            return server_process
        else:
            stdout, stderr = server_process.communicate()
            print(f"❌ Server failed to start: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def open_interface():
    """Open the web interface (served by backend at localhost:5001)."""
    url = "http://localhost:5001/"
    print(f"Opening interface: {url}")
    webbrowser.open(url)
    return True

def main():
    print("EKPHRASIS System Launcher")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check ML dependencies (install flask-cors if only that is missing)
    if not check_ml_dependencies():
        if ensure_flask_cors() and check_ml_dependencies():
            pass
        else:
            print("\nInstalling dependencies...")
            if not install_dependencies():
                sys.exit(1)
    
    # Check model file
    if not check_model_file():
        print("\nTraining model...")
        if not train_model():
            print("❌ Failed to train model. Please check your dataset and try again.")
            sys.exit(1)
    
    # Start server
    server_process = start_server()
    if not server_process:
        sys.exit(1)
    
    # Open interface
    if not open_interface():
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("🎉 EKPHRASIS is now running!")
    print("\nSystem Status:")
    print("  ✓ ML Server: http://localhost:5001")
    print("  ✓ Interface: interface/interface.html")
    print("\nUsage:")
    print("  1. Create a composition on the canvas")
    print("  2. Click 'Evaluate with AI' to analyze")
    print("  3. View results and feedback")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        # Keep the script running
        server_process.wait()
    except KeyboardInterrupt:
        print("\n\nStopping EKPHRASIS...")
        server_process.terminate()
        server_process.wait()
        print("✓ Server stopped")

if __name__ == "__main__":
    main() 