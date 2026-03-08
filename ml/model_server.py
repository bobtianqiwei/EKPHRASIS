from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import base64
from PIL import Image
import io
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Serve frontend from project interface/ folder (so user opens http://localhost:5001/)
INTERFACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'interface')

# Each design vocabulary has its own model. Add new entries when you train new models.
# model_file is relative to the ml/ directory.
CRITERIA = [
    {'id': 'visual_harmony', 'name': 'Visual Harmony', 'model_file': 'composition_model.h5'},
    {'id': 'visual_balance', 'name': 'Visual Balance', 'model_file': 'visual_balance_model.h5'},
]

# Load all criterion models at startup (criterion_id -> Keras model)
models = {}
for c in CRITERIA:
    try:
        models[c['id']] = load_model(c['model_file'])
        print(f"Model loaded for criterion: {c['name']} ({c['model_file']})")
    except Exception as e:
        print(f"Warning: Could not load {c['model_file']} for '{c['name']}': {e}")

# Legacy single-model reference for /health (any loaded model counts as "ready")
model = models.get('visual_harmony') or (list(models.values())[0] if models else None)

def predict_image_from_base64(base64_string, model_obj):
    """
    Predict the composition score from a base64 encoded image
    """
    if model_obj is None:
        return {"error": "Model not loaded"}
    
    try:
        # Decode base64 string to image
        image_data = base64.b64decode(base64_string.split(',')[1])
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 224x224 (VGG16 input size)
        img = img.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Make prediction
        prediction = model_obj.predict(img_array)
        confidence = float(prediction[0][0])
        
        return {
            "confidence": confidence,
            "class": "class_1" if confidence >= 0.5 else "class_0",
            "score": confidence if confidence >= 0.5 else 1 - confidence
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict composition score
    Expects JSON with 'image' field containing base64 encoded image
    """
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        criterion = data.get('criterion', 'visual_harmony')
        model_obj = models.get(criterion)
        if not model_obj:
            return jsonify({"error": f"Unknown criterion or model not loaded: {criterion}"}), 400
        result = predict_image_from_base64(data['image'], model_obj)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/criteria', methods=['GET'])
def list_criteria():
    """Return list of design vocabularies (id, name) for the dropdown."""
    return jsonify([{'id': c['id'], 'name': c['name']} for c in CRITERIA if c['id'] in models])

@app.route('/predict_multiple', methods=['POST'])
def predict_multiple():
    """
    API endpoint to predict multiple images and return best/worst scores.
    Expects JSON with 'images' (base64 array) and optional 'criterion' (e.g. 'visual_harmony').
    """
    try:
        data = request.get_json()
        if 'images' not in data:
            return jsonify({"error": "No images data provided"}), 400
        
        criterion = data.get('criterion', 'visual_harmony')
        model_obj = models.get(criterion)
        if not model_obj:
            return jsonify({"error": f"Unknown criterion or model not loaded: {criterion}"}), 400
        
        images = data['images']
        results = []
        
        for i, img_base64 in enumerate(images):
            result = predict_image_from_base64(img_base64, model_obj)
            if 'error' not in result:
                result['index'] = i
                results.append(result)
        
        if not results:
            return jsonify({"error": "No valid predictions"}), 400
        
        # Sort by confidence score
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        best_result = results[0]
        worst_result = results[-1]
        
        return jsonify({
            "best": {
                "index": best_result['index'],
                "confidence": best_result['confidence'],
                "class": best_result['class']
            },
            "worst": {
                "index": worst_result['index'],
                "confidence": worst_result['confidence'],
                "class": worst_result['class']
            },
            "all_results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/')
def index():
    """Serve the frontend so user opens http://localhost:5001/ (no file://)."""
    return send_from_directory(INTERFACE_DIR, 'interface.html')

@app.route('/logo.png')
def logo():
    """Serve topbar logo."""
    return send_from_directory(INTERFACE_DIR, 'logo.png')

if __name__ == '__main__':
    app.run(debug=True, port=5001) 