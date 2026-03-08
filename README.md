# EKPHRASIS - Generating Visual Aids for Graphic Design Education

EKPHRASIS is an interactive educational system that helps students understand graphic design language—such as "visual harmony"—by generating concrete visual aids from their own compositions. Users create a composition on a digital canvas; the system then provides "less" and "more" effective examples (relative to the chosen criterion) so learners can compare and refine their intuition.

This repository accompanies the paper **"Generating Visual Aids to Help Students Understand Graphic Design with EKPHRASIS"** (CHI EA ’25).

- **Paper (open access):** [https://doi.org/10.1145/3706599.3719807](https://doi.org/10.1145/3706599.3719807)

## Citation

If you use or build on EKPHRASIS, please cite our paper:

```bibtex
@inproceedings{wei2025generating,
  title={Generating Visual Aids to Help Students Understand Graphic Design with EKPHRASIS},
  author={Wei, Bob Tianqi and Shen, Shayne and Almeda, Shm Garanganao and Hartmann, Bjoern},
  booktitle={Proceedings of the Extended Abstracts of the CHI Conference on Human Factors in Computing Systems},
  pages={1--7},
  year={2025}
}
```

## Features

- **Interactive Canvas**: Draw and arrange rectangles in different shades of grey
- **Visual Aids**: ML-backed system generates "less" and "more" effective examples (e.g. less/more visual harmony) from your composition
- **Variation Generation**: Automatically creates multiple variations; best and worst (relative to the criterion) are shown left/right of the canvas
- **Educational Focus**: Supports associating design vocabulary with visual examples for learning

## System Architecture

```
EKPHRASIS/
├── interface/
│   ├── interface.html          # Main user interface
│   └── logo.png
├── ml/
│   ├── model_server.py         # Flask API server
│   ├── train_and_save_model.py # Train one model per vocabulary
│   ├── start_server.py         # Server startup script
│   ├── test_system.py          # System test script
│   ├── models/                 # One .h5 per vocabulary (e.g. visual_balance.h5)
│   ├── requirements.txt
│   └── README.md
├── dataset/                    # Data per vocabulary
│   ├── visual_balance/         # For vocabulary "visual_balance"
│   │   ├── class_0/            # Less [criterion]
│   │   └── class_1/            # More [criterion]
│   ├── balance/                # Legacy path (Bob's classes)
│   │   └── Bob's classes/
│   │       ├── class_0/
│   │       └── class_1/
│   └── test_images/            # Test images
├── other_files/
│   └── archive/                # Old demos, notebooks (e.g. demo_backup)
├── start_ekphrasis.py          # Launcher: train (if needed), start server, open browser
├── Start EKPHRASIS.command     # macOS double-click launcher
└── README.md
```

## Quick Start

### 1. Setup ML Backend

```bash
# Navigate to ML directory
cd ml

# Install dependencies
pip install -r requirements.txt

# Train a vocabulary model (first time or when adding a new one)
python train_and_save_model.py visual_balance

# Start the ML server
python start_server.py
```

The server will start on `http://localhost:5001` (port 5001 is used to avoid conflict with macOS AirPlay on 5000).

### 2. Open the Interface

Open **http://localhost:5001/** in your browser (the backend serves the interface), or run from the project root:

```bash
python start_ekphrasis.py
```

to start the server and open the interface automatically.

### 3. Create and Get Visual Aids

1. Use the canvas to create a composition with rectangles
2. Choose colors from the grey palette
3. Click **"Generate visual aids"** to get less/more effective examples (e.g. less/more visual balance)
4. Compare the two images shown left and right of the canvas

**System requirements**: Python 3.7+, modern web browser, 4GB+ RAM for the ML model. The server runs on port **5001** (avoid conflict with macOS AirPlay on 5000).

## How It Works

### Frontend (interface.html)
- Interactive canvas for composition creation
- Color palette with 5 shades of grey
- Real-time drawing and dragging of rectangles
- Integration with ML backend via REST API

### Backend (ML Server)
- Flask server providing prediction endpoints
- VGG16-based neural network for composition analysis
- Binary classification: harmonious vs less harmonious
- Processes multiple image variations simultaneously

### Workflow
1. User creates composition on canvas
2. System generates multiple variations with slight modifications
3. All variations are sent to the ML model for prediction
4. "Less" and "more" effective examples (e.g. less/more visual harmony) are shown left and right of the canvas
5. Optional feedback text is shown when applicable

## API Endpoints

- `GET /health` - Server status check
- `POST /predict` - Single image prediction
- `POST /predict_multiple` - Multiple image predictions with ranking

## Model Details

- **Architecture**: VGG16 + custom classification layers
- **Input**: 224x224 RGB images
- **Output**: Confidence score (0-1) for harmony
- **Training**: Binary classification on composition dataset

## Testing

Run the test script to verify system functionality:

```bash
cd ml
python test_system.py
```

## Troubleshooting

### Common Issues

1. **Server not starting**: Check if port 5001 is available (macOS may use 5000 for AirPlay)
2. **Model not found**: Run `cd ml && python train_and_save_model.py visual_balance` (requires `dataset/visual_balance/class_0` and `class_1`)
3. **Import errors**: Install dependencies with `pip install -r requirements.txt` and `pip install flask-cors`
4. **Frontend can't connect**: Ensure server is running and open **http://localhost:5001/**

### Performance Notes

- Model predictions take ~100-200ms per image
- Server can handle multiple concurrent requests
- For production, consider model optimization

## Development

### Adding a new design vocabulary (new model)

Each design term (e.g. Visual Harmony) has its own trained model. To add another:

1. **Add dataset**: Create dataset/<vocabulary_id>/ with class_0/ (less) and class_1/ (more). Example: dataset/visual_harmony/class_0/, dataset/visual_harmony/class_1/.
2. **Train**: `cd ml && python train_and_save_model.py <vocabulary_id>` (e.g. `python train_and_save_model.py visual_harmony`). This saves `ml/models/<vocabulary_id>.h5`.
3. **Register** in `ml/model_server.py`: add to the `CRITERIA` list, e.g. `{'id': 'visual_harmony', 'name': 'Visual Harmony', 'model_file': 'models/visual_harmony.h5'}`.
4. Restart the server. The frontend dropdown is filled from `GET /criteria`.

### Adding New Features

1. **Frontend**: Modify `interface/interface.html`
2. **Backend**: Extend `ml/model_server.py`
3. **Model**: Update training script and retrain

### Dataset Structure

One folder per vocabulary under `dataset/<vocabulary_id>/`, each with `class_0` (less) and `class_1` (more). The training script reads from `dataset/<vocabulary_id>/` (e.g. `dataset/visual_balance/`).

```
dataset/
├── visual_balance/             # Used by: python train_and_save_model.py visual_balance
│   ├── class_0/                 # Less [criterion] (PNG files)
│   └── class_1/                 # More [criterion]
├── balance/Bob's classes/       # Legacy; can mirror to visual_balance or other id
│   ├── class_0/
│   └── class_1/
└── test_images/                 # Optional test images
```

## Credits

EKPHRASIS by Bob Tianqi Wei, Shayne Shen, Shm Garanganao Almeda, and Bjoern Hartmann, UC Berkeley. CHI EA ’25.

## License

This project is for research and educational purposes.