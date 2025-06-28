# EKPHRASIS Quick Start Guide

## What is EKPHRASIS?

EKPHRASIS is an AI-powered visual composition analysis system. Create compositions with geometric shapes and get instant feedback on visual harmony using machine learning.

## Quick Setup (3 Steps)

### Step 1: Install Dependencies
```bash
cd ml
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train_and_save_model.py
```
*This may take a few minutes the first time*

### Step 3: Start the System
```bash
python start_server.py
```

Then open `interface/interface.html` in your browser.

## How to Use

1. **Create**: Draw rectangles on the canvas using different shades of grey
2. **Evaluate**: Click "Evaluate with AI" to analyze your composition
3. **Learn**: View scores and feedback for 10 automatically generated variations

## What You'll See

- **Canvas**: Interactive drawing area (375x375 pixels)
- **Color Palette**: 5 shades of grey to choose from
- **Controls**: Draw, clear, export, and evaluate buttons
- **Results**: Best/worst scores and AI feedback
- **Downloads**: Best and worst variations saved automatically

## System Requirements

- Python 3.7+
- Web browser (Chrome, Firefox, Safari, Edge)
- 4GB+ RAM (for ML model)
- Internet connection (for initial model download)

## Troubleshooting

**"Model not found" error:**
- Run the training script first: `python train_and_save_model.py`

**"Server connection failed" error:**
- Make sure the server is running: `python start_server.py`
- Check that port 5000 is available

**Import errors:**
- Install dependencies: `pip install -r requirements.txt`

## Files Overview

- `interface/interface.html` - Main user interface
- `ml/model_server.py` - AI prediction server
- `ml/train_and_save_model.py` - Model training script
- `dataset/balance/Bob's classes/` - Training data

## Support

For detailed documentation, see `README.md` and `ml/README.md`.

---

**EKPHRASIS by Bob Tianqi Wei, Shayne Shen, UC Berkeley, 2024** 