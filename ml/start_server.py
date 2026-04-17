#!/usr/bin/env python3
"""
Start script for the EKPHRASIS Composition Analysis Server
"""

import os
import sys
import subprocess
import time
import json
import urllib.request
from pathlib import Path

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5001
HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
DEEP_HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/health/predict"
STARTUP_TIMEOUT_SECONDS = 60
HEALTH_CHECK_INTERVAL_SECONDS = 5
DEEP_HEALTH_TIMEOUT_SECONDS = 20
CHILD_FLAG = "EKPHRASIS_SERVER_CHILD"

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['flask', 'tensorflow', 'numpy', 'PIL']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except Exception as e:
            missing_packages.append(f"{package} ({e})")
    
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

def fetch_json(url, timeout_seconds):
    """Fetch one JSON payload from the local server."""
    request_obj = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request_obj, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body)

def wait_for_server_ready(timeout_seconds):
    """Wait until the worker reports that the model is loaded."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            status_code, payload = fetch_json(HEALTH_URL, 3)
            if status_code == 200 and payload.get("model_loaded"):
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def start_child_process():
    """Start the worker process that serves Flask and TensorFlow."""
    script_path = Path(__file__).resolve()
    env = os.environ.copy()
    env[CHILD_FLAG] = "1"
    return subprocess.Popen([sys.executable, str(script_path)], cwd=str(script_path.parent), env=env)

def stop_child_process(child_process):
    """Stop the worker process cleanly, then force kill if needed."""
    if child_process.poll() is not None:
        return
    child_process.terminate()
    try:
        child_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child_process.kill()
        child_process.wait(timeout=5)

def restart_child_process(child_process, reason):
    """Restart the worker after a crash or deep-health failure."""
    print(f"[supervisor] Restarting ML worker: {reason}")
    stop_child_process(child_process)
    new_child = start_child_process()
    if wait_for_server_ready(STARTUP_TIMEOUT_SECONDS):
        print("[supervisor] Worker is healthy again.")
    else:
        print("[supervisor] Worker did not become ready in time; supervisor will keep watching.")
    return new_child

def run_child_server():
    """Run the actual Flask server without the auto-reloader."""
    from model_server import app
    app.run(debug=False, use_reloader=False, threaded=True, port=SERVER_PORT, host='0.0.0.0')

def run_supervisor():
    """Run a parent supervisor that restarts the worker when inference hangs."""
    print("Starting server supervisor...")
    print(f"Server will be available at: http://localhost:{SERVER_PORT}")
    print("Press Ctrl+C to stop the server")
    print("-" * 40)

    child_process = start_child_process()
    if wait_for_server_ready(STARTUP_TIMEOUT_SECONDS):
        print("[supervisor] Worker started successfully.")
    else:
        print("[supervisor] Worker did not report ready during startup.")

    try:
        while True:
            time.sleep(HEALTH_CHECK_INTERVAL_SECONDS)
            if child_process.poll() is not None:
                child_process = restart_child_process(child_process, f"worker exited with code {child_process.returncode}")
                continue
            try:
                status_code, payload = fetch_json(DEEP_HEALTH_URL, DEEP_HEALTH_TIMEOUT_SECONDS)
                if status_code != 200 or payload.get("status") != "ready":
                    child_process = restart_child_process(
                        child_process,
                        payload.get("error") or f"deep health status {status_code}"
                    )
            except Exception as e:
                child_process = restart_child_process(child_process, f"deep health timeout/failure: {e}")
    except KeyboardInterrupt:
        print("\nStopping server supervisor...")
        stop_child_process(child_process)
        print("Server stopped.")

def main():
    print("EKPHRASIS Composition Analysis Server")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model file
    if not check_model_file():
        sys.exit(1)
    
    try:
        if os.environ.get(CHILD_FLAG) == "1":
            run_child_server()
        else:
            run_supervisor()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
