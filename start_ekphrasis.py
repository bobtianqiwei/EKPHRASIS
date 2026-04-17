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
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def get_installed_version(package_name):
    """Return installed package version, or None when package is missing."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def parse_major_version(version_text):
    """Return the integer major version when possible."""
    if not version_text:
        return None
    try:
        return int(version_text.split(".", 1)[0])
    except ValueError:
        return None


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    if sys.version_info >= (3, 13):
        print("❌ Python 3.13 is not supported yet for this project")
        print("Please use Python 3.9 to 3.12")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True

def check_ml_dependencies():
    """Check if ML dependencies are installed"""
    tensorflow_version = get_installed_version("tensorflow")
    numpy_version = get_installed_version("numpy")
    flask_version = get_installed_version("flask")
    flask_cors_version = get_installed_version("flask-cors")
    pillow_version = get_installed_version("Pillow")

    if not all([tensorflow_version, numpy_version, flask_version, flask_cors_version, pillow_version]):
        print("❌ Some ML dependencies are missing")
        return False

    if parse_major_version(numpy_version) and parse_major_version(numpy_version) >= 2:
        print(
            "❌ Incompatible dependency combination detected: "
            f"tensorflow {tensorflow_version} with numpy {numpy_version}"
        )
        print("   EKPHRASIS currently needs NumPy 1.x in this setup.")
        return False

    try:
        import tensorflow
        import flask
        import flask_cors
        import numpy
        from PIL import Image
        print("✓ ML dependencies are installed")
        return True
    except Exception as e:
        print(f"❌ ML dependency check failed: {e}")
        return False

def check_model_file():
    """Check if at least one vocabulary model exists in ml/models/"""
    models_dir = PROJECT_ROOT / "ml" / "models"
    if not models_dir.exists():
        print("⚠️  ml/models/ not found")
        return False
    if list(models_dir.glob("*.h5")):
        print("✓ Trained model(s) found")
        return True
    print("⚠️  No .h5 model in ml/models/")
    return False

def install_dependencies():
    """Install ML dependencies"""
    print("Installing ML dependencies...")
    try:
        flask_version = get_installed_version("flask")
        flask_cors_version = get_installed_version("flask-cors")
        pillow_version = get_installed_version("Pillow")
        tensorflow_version = get_installed_version("tensorflow")
        numpy_version = get_installed_version("numpy")

        # Keep the fix as small as possible when only NumPy is incompatible.
        if (
            flask_version
            and flask_cors_version
            and pillow_version
            and tensorflow_version
            and parse_major_version(numpy_version) is not None
            and parse_major_version(numpy_version) >= 2
        ):
            target_numpy = "1.26.4" if sys.version_info >= (3, 12) else "1.24.3"
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f"numpy=={target_numpy}"],
                check=True,
                cwd=str(PROJECT_ROOT),
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "ml" / "requirements.txt")],
                check=True,
                cwd=str(PROJECT_ROOT),
            )
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def train_model():
    """Train the ML model for default vocabulary (visual_balance)"""
    print("Training the ML model (visual_balance)...")
    print("This may take several minutes...")
    try:
        subprocess.run(
            [sys.executable, "train_and_save_model.py", "visual_balance"],
            cwd=str(PROJECT_ROOT / "ml"),
            check=True,
            text=True,
        )
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
        server_process = subprocess.Popen(
            [sys.executable, "start_server.py"],
            cwd=str(PROJECT_ROOT / "ml"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
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
    
    # Check ML dependencies
    if not check_ml_dependencies():
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
