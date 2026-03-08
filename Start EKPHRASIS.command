#!/bin/bash
# Start EKPHRASIS.command - double-click to run (macOS)
# Developed by Bob Tianqi Wei

# Go to the folder where this script lives (project root)
cd "$(dirname "$0")"

# Run the Python launcher (starts ML server + opens browser to http://localhost:5001/)
python3 start_ekphrasis.py

# Keep window open if script exits (e.g. error before server starts)
if [ $? -ne 0 ]; then
    echo ""
    read -n 1 -s -r -p "Press any key to close this window..."
fi
