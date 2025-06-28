# EKPHRASIS ML Model Integration

This directory contains the machine learning components for the EKPHRASIS composition analysis system.

## Files Overview

- `train_and_save_model.py` - Script to train the VGG16-based composition classification model
- `model_server.py` - Flask server that provides ML prediction API endpoints
- `start_server.py` - Convenience script to start the ML server
- `requirements.txt` - Python dependencies
- `composition_model.h5` - Trained model file (will be generated after training)

## Setup Instructions

### 1. Install Dependencies

```bash
cd ml
pip install -r requirements.txt
```

### 2. Train the Model

First, make sure your dataset is properly organized:
```
dataset/balance/Bob's classes/
├── class_0/  # Less harmonious compositions
└── class_1/  # More harmonious compositions
```

Then train the model:
```bash
python train_and_save_model.py
```

This will:
- Load the VGG16 pre-trained model
- Add custom classification layers
- Train on your composition dataset
- Save the trained model as `composition_model.h5`

### 3. Start the ML Server

```bash
python start_server.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
- **GET** `/health`
- Returns server status and model loading status

### Single Image Prediction
- **POST** `/predict`
- Body: `{"image": "base64_encoded_image_string"}`
- Returns prediction confidence and class

### Multiple Images Prediction
- **POST** `/predict_multiple`
- Body: `{"images": ["base64_img1", "base64_img2", ...]}`
- Returns best and worst scores with indices

## Integration with Frontend

The frontend interface (`interface/interface.html`) is now integrated with the ML server:

1. User creates a composition on the canvas
2. Clicks "Evaluate with AI" button
3. System generates 10 variations of the composition
4. Sends all variations to the ML server for prediction
5. Displays the best and worst scores
6. Provides feedback based on the results
7. Automatically saves the best and worst variations as images

## Model Architecture

- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Input Size**: 224x224x3 RGB images
- **Classification**: Binary (harmonious vs less harmonious)
- **Output**: Sigmoid activation (0-1 confidence score)

## Troubleshooting

### Model Not Found
If you get "Model file not found" error:
1. Make sure you've run the training script first
2. Check that `composition_model.h5` exists in the ml directory

### Import Errors
If you get import errors:
1. Install all dependencies: `pip install -r requirements.txt`
2. Make sure you're using Python 3.7+

### Server Connection Issues
If the frontend can't connect to the server:
1. Make sure the server is running on port 5000
2. Check that CORS is enabled (should be automatic)
3. Verify the frontend is making requests to `http://localhost:5000`

## Performance Notes

- The model uses VGG16 which is relatively large but provides good feature extraction
- Predictions typically take 100-200ms per image
- The server can handle multiple concurrent requests
- For production use, consider using a lighter model or optimizing the inference pipeline 