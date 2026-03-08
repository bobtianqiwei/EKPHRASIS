#!/usr/bin/env python3
"""
Start script for the EKPHRASIS Composition Analysis Server
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['flask', 'tensorflow', 'numpy', 'PIL']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    return True

def check_model_file():
    """Check if at least one criterion model exists in ml/models/"""
    models_dir = Path(__file__).resolve().parent / "models"
    if not models_dir.exists():
        print("Models directory 'ml/models/' not found!")
        return False
    if not list(models_dir.glob("*.h5")):
        print("No .h5 model file found in ml/models/. Train a vocabulary first:")
        print("  python train_and_save_model.py visual_balance")
        return False
    return True

def main():
    print("EKPHRASIS Composition Analysis Server")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model file
    if not check_model_file():
        sys.exit(1)
    
    print("Starting server...")
    print("Server will be available at: http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    print("-" * 40)
    
    # Start the server (5001: macOS often uses 5000 for AirPlay)
    try:
        from model_server import app
        app.run(debug=True, port=5001, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 