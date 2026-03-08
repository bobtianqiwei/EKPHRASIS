from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import re
import base64
import tempfile
import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Serve frontend from project interface/ folder (so user opens http://localhost:5001/)
INTERFACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'interface')

# Each design vocabulary has its own model. Add new entries when you train new models.
# model_file is relative to the ml/ directory (e.g. models/visual_balance.h5).
CRITERIA = [
    {'id': 'visual_balance', 'name': 'Visual Balance', 'model_file': 'models/visual_balance.h5'},
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
model = list(models.values())[0] if models else None

def predict_image_from_base64(base64_string, model_obj):
    """
    Predict the composition score from a base64 encoded image.
    Preprocessing matches the notebook: Keras load_img(target_size=(224,224)) + img_to_array + /255.
    """
    if model_obj is None:
        return {"error": "Model not loaded"}
    
    try:
        # Parse data URL (e.g. data:image/png;base64,...)
        header, b64 = base64_string.split(',', 1)
        image_data = base64.b64decode(b64)
        suffix = '.png'
        if 'jpeg' in header or 'jpg' in header:
            suffix = '.jpg'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name
        try:
            img = image.load_img(tmp_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            prediction = model_obj.predict(img_array)
            confidence = float(prediction[0][0])
            return {
                "confidence": confidence,
                "class": "class_1" if confidence >= 0.5 else "class_0",
                "score": confidence if confidence >= 0.5 else 1 - confidence
            }
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
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
        
        criterion = data.get('criterion', 'visual_balance')
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

def _sanitize_for_folder(s):
    """Allow only alphanumeric, underscore, hyphen for folder names."""
    s = (s or '').strip() or 'unknown'
    s = re.sub(r'[^\w\-]', '_', s)
    return s[:80] or 'unknown'

def _label_folder_path(project_root, vocabulary, labeler):
    """Path: dataset/new/{vocabulary}_{date}_{labeler}"""
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    vocab = _sanitize_for_folder(vocabulary)
    lbl = _sanitize_for_folder(labeler)
    folder_name = f"{vocab}_{date_str}_{lbl}"
    return os.path.join(project_root, 'dataset', 'new', folder_name)

@app.route('/label_counts', methods=['GET'])
def label_counts():
    """Return counts for dataset/new/{vocabulary}_{date}_{labeler}. Query: vocabulary, labeler."""
    try:
        vocabulary = request.args.get('vocabulary', 'visual_balance')
        labeler = request.args.get('labeler', '')
        project_root = os.path.dirname(INTERFACE_DIR)
        dataset_dir = _label_folder_path(project_root, vocabulary, labeler)
        c0_dir = os.path.join(dataset_dir, 'class_0')
        c1_dir = os.path.join(dataset_dir, 'class_1')
        c0_count = len([f for f in os.listdir(c0_dir) if f.endswith(('.png', '.jpg'))]) if os.path.isdir(c0_dir) else 0
        c1_count = len([f for f in os.listdir(c1_dir) if f.endswith(('.png', '.jpg'))]) if os.path.isdir(c1_dir) else 0
        return jsonify({"class_0": c0_count, "class_1": c1_count, "total": c0_count + c1_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save_label', methods=['POST'])
def save_label():
    """
    Save a labeled image to dataset/new/{vocabulary}_{date}_{labeler}.
    Expects JSON: { vocabulary: str, labeler: str, label: 'class_0'|'class_1', image: base64 }
    """
    try:
        data = request.get_json()
        vocabulary = data.get('vocabulary') or data.get('criterion') or 'visual_balance'
        labeler = data.get('labeler') or 'unknown'
        label = data.get('label')
        image_b64 = data.get('image')
        
        if not label or not image_b64:
            return jsonify({"error": "Missing label or image"}), 400
            
        project_root = os.path.dirname(INTERFACE_DIR)
        dataset_dir = _label_folder_path(project_root, vocabulary, labeler)
        
        class_dir = os.path.join(dataset_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        other_class = 'class_1' if label == 'class_0' else 'class_0'
        os.makedirs(os.path.join(dataset_dir, other_class), exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%H%M%S%f')
        filepath = os.path.join(class_dir, f"image_{timestamp}.png")
        
        header, b64 = image_b64.split(',', 1)
        image_data = base64.b64decode(b64)
        with open(filepath, 'wb') as f:
            f.write(image_data)
            
        c0_dir = os.path.join(dataset_dir, 'class_0')
        c1_dir = os.path.join(dataset_dir, 'class_1')
        c0_count = len([f for f in os.listdir(c0_dir) if f.endswith(('.png', '.jpg'))])
        c1_count = len([f for f in os.listdir(c1_dir) if f.endswith(('.png', '.jpg'))])
        
        return jsonify({
            "success": True,
            "counts": {"class_0": c0_count, "class_1": c1_count, "total": c0_count + c1_count}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_multiple', methods=['POST'])
def predict_multiple():
    """
    API endpoint to predict multiple images and return best/worst scores.
    Expects JSON with 'images' (base64 array) and optional 'criterion' (e.g. 'visual_balance').
    """
    try:
        data = request.get_json()
        if 'images' not in data:
            return jsonify({"error": "No images data provided"}), 400
        
        criterion = data.get('criterion', 'visual_balance')
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

@app.route('/test_images', methods=['GET'])
def get_test_images():
    """Return all images in dataset/test_images as base64 strings."""
    try:
        project_root = os.path.dirname(INTERFACE_DIR)
        test_images_dir = os.path.join(project_root, 'dataset', 'test_images')
        if not os.path.isdir(test_images_dir):
            return jsonify({"error": "dataset/test_images directory not found"}), 404
        
        images = []
        # Sort files to ensure stable order
        for filename in sorted(os.listdir(test_images_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(test_images_dir, filename)
                with open(filepath, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                    ext = filename.split('.')[-1].lower()
                    mime_type = "image/jpeg" if ext in ['jpg', 'jpeg'] else "image/png"
                    images.append(f"data:{mime_type};base64,{encoded}")
        return jsonify({"images": images})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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