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
│   └── interface.html          # Main user interface
├── ml/
│   ├── model_server.py         # Flask API server
│   ├── train_and_save_model.py # Model training script
│   ├── start_server.py         # Server startup script
│   ├── test_system.py          # System testing script
│   ├── requirements.txt        # Python dependencies
│   └── README.md              # ML module documentation
└── dataset/
    └── balance/
        └── Bob's classes/
            ├── class_0/        # Less harmonious compositions
            └── class_1/        # More harmonious compositions
```

## Quick Start

### 1. Setup ML Backend

```bash
# Navigate to ML directory
cd ml

# Install dependencies
pip install -r requirements.txt

# Train the model (first time only)
python train_and_save_model.py

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
3. Click **"Generate visual aids"** to get less/more effective examples (e.g. less/more visual harmony)
4. Compare the two images shown left and right of the canvas

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
2. **Model not found**: Run training script first
3. **Import errors**: Install dependencies with `pip install -r requirements.txt` and `pip install flask-cors`
4. **Frontend can't connect**: Ensure server is running and open http://localhost:5001/

### Performance Notes

- Model predictions take ~100-200ms per image
- Server can handle multiple concurrent requests
- For production, consider model optimization

## Development

### Adding New Features

1. **Frontend**: Modify `interface/interface.html`
2. **Backend**: Extend `ml/model_server.py`
3. **Model**: Update training script and retrain

### Dataset Structure

```
dataset/balance/Bob's classes/
├── class_0/  # Less harmonious compositions (PNG files)
└── class_1/  # More harmonious compositions (PNG files)
```

## Credits

EKPHRASIS by Bob Tianqi Wei, Shayne Shen, Shm Garanganao Almeda, and Bjoern Hartmann, UC Berkeley. CHI EA ’25.

## License

This project is for research and educational purposes.