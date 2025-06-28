# EKPHRASIS - Visual Composition Analysis System

EKPHRASIS is an interactive system for creating and analyzing visual compositions using machine learning. Users can create compositions with geometric shapes and receive AI-powered feedback on visual harmony.

## Features

- **Interactive Canvas**: Draw and arrange rectangles in different shades of grey
- **AI Analysis**: Machine learning model evaluates composition harmony
- **Variation Generation**: Automatically creates 10 variations of your composition
- **Score Comparison**: Shows best and worst performing variations
- **Real-time Feedback**: Provides constructive feedback based on AI analysis

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

The server will start on `http://localhost:5000`

### 2. Open the Interface

Open `interface/interface.html` in your web browser.

### 3. Create and Evaluate

1. Use the canvas to create a composition with rectangles
2. Choose colors from the grey palette
3. Click "Evaluate with AI" to analyze your composition
4. View the results and feedback

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
2. System generates 10 variations with slight modifications
3. All variations are sent to ML model for prediction
4. Best and worst scores are displayed
5. Constructive feedback is provided based on results

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

1. **Server not starting**: Check if port 5000 is available
2. **Model not found**: Run training script first
3. **Import errors**: Install dependencies with `pip install -r requirements.txt`
4. **Frontend can't connect**: Ensure server is running on localhost:5000

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

EKPHRASIS by Bob Tianqi Wei, Shayne Shen, UC Berkeley, 2024

## License

This project is for research and educational purposes.