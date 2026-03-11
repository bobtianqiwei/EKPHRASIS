from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import re
import random
import base64
import tempfile
import datetime
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Serve frontend from project interface/ folder (so user opens http://localhost:5001/)
INTERFACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'interface')

# Each design vocabulary has its own model. Add new entries when you train new models.
# model_file is relative to the ml/ directory (e.g. models/visual_balance.h5).
CRITERIA = [
    {'id': 'visual_balance', 'name': 'Visual Balance', 'model_file': 'models/visual_balance.h5'},
    {'id': 'visual_harmony', 'name': 'Visual Harmony', 'model_file': 'models/visual_harmony.h5'},
    {'id': 'visual_hierarchy', 'name': 'Visual Hierarchy', 'model_file': 'models/visual_hierarchy.h5'},
    {'id': 'contrast', 'name': 'Contrast', 'model_file': 'models/contrast.h5'},
    {'id': 'rhythm', 'name': 'Rhythm', 'model_file': 'models/rhythm.h5'},
    {'id': 'emphasis', 'name': 'Emphasis', 'model_file': 'models/emphasis.h5'},
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
    """Return list of design vocabularies (id, name) for the dropdown. Includes all CRITERIA so users can label data for any vocabulary; only those with loaded models support prediction."""
    return jsonify([{'id': c['id'], 'name': c['name']} for c in CRITERIA])

def _sanitize_for_folder(s):
    """Allow only alphanumeric, underscore, hyphen for folder names."""
    s = (s or '').strip() or 'unknown'
    s = re.sub(r'[^\w\-]', '_', s)
    return s[:80] or 'unknown'

def _sanitize_study_filename(name):
    """Allow only alphanumeric, hyphen, underscore, dot for study filenames (no path traversal)."""
    if not name or '..' in name or '/' in name or '\\' in name:
        return None
    s = re.sub(r'[^\w\-.]', '_', name)
    return s[:120] if s else None

def _study_user_dir(project_root, username):
    """Path: dataset/study/{username}/"""
    user = _sanitize_for_folder(username)
    return os.path.join(project_root, 'dataset', 'study', user)

def _study_events_path(project_root, username):
    """Path: dataset/study/{username}/events.jsonl"""
    return os.path.join(_study_user_dir(project_root, username), 'events.jsonl')

def _read_study_events(project_root, username):
    """Read JSONL study events for one user; skip malformed lines."""
    events_path = _study_events_path(project_root, username)
    if not os.path.isfile(events_path):
        return []
    events = []
    with open(events_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    events.sort(key=lambda e: (e.get('timestamp') or '', e.get('event_id') or ''))
    return events

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

@app.route('/balance_labels', methods=['POST'])
def balance_labels():
    """
    For dataset/new/{vocabulary}_{date}_{labeler}, randomly delete files from the
    larger class until class_0 and class_1 counts are equal.
    Expects JSON: { vocabulary: str, labeler: str }
    """
    try:
        data = request.get_json() or {}
        vocabulary = data.get('vocabulary') or data.get('criterion') or 'visual_balance'
        labeler = data.get('labeler') or 'unknown'
        project_root = os.path.dirname(INTERFACE_DIR)
        dataset_dir = _label_folder_path(project_root, vocabulary, labeler)
        c0_dir = os.path.join(dataset_dir, 'class_0')
        c1_dir = os.path.join(dataset_dir, 'class_1')
        if not os.path.isdir(c0_dir):
            return jsonify({"error": "class_0 folder not found", "counts": {"class_0": 0, "class_1": 0, "total": 0}}), 400
        if not os.path.isdir(c1_dir):
            return jsonify({"error": "class_1 folder not found", "counts": {"class_0": 0, "class_1": 0, "total": 0}}), 400
        c0_files = [f for f in os.listdir(c0_dir) if f.endswith(('.png', '.jpg'))]
        c1_files = [f for f in os.listdir(c1_dir) if f.endswith(('.png', '.jpg'))]
        c0_count, c1_count = len(c0_files), len(c1_files)
        if c0_count > c1_count:
            to_remove = random.sample(c0_files, c0_count - c1_count)
            for f in to_remove:
                try:
                    os.unlink(os.path.join(c0_dir, f))
                except Exception:
                    pass
        elif c1_count > c0_count:
            to_remove = random.sample(c1_files, c1_count - c0_count)
            for f in to_remove:
                try:
                    os.unlink(os.path.join(c1_dir, f))
                except Exception:
                    pass
        c0_count = len([f for f in os.listdir(c0_dir) if f.endswith(('.png', '.jpg'))])
        c1_count = len([f for f in os.listdir(c1_dir) if f.endswith(('.png', '.jpg'))])
        return jsonify({
            "success": True,
            "counts": {"class_0": c0_count, "class_1": c1_count, "total": c0_count + c1_count}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save_study', methods=['POST'])
def save_study():
    """
    Save a study recording image to dataset/study/{username}/{filename}.
    Expects JSON: { username: str, filename: str, image: base64 or data URL }
    Creates dataset/study/{username}/ if needed. Filename must be safe (no path traversal).
    """
    try:
        data = request.get_json()
        username = (data.get('username') or '').strip()
        filename = data.get('filename')
        image_b64 = data.get('image')
        if not username or not filename or not image_b64:
            return jsonify({"error": "Missing username, filename, or image"}), 400
        safe_filename = _sanitize_study_filename(filename)
        if not safe_filename or not safe_filename.endswith(('.png', '.jpg')):
            return jsonify({"error": "Invalid filename"}), 400
        project_root = os.path.dirname(INTERFACE_DIR)
        user_dir = _study_user_dir(project_root, username)
        os.makedirs(user_dir, exist_ok=True)
        filepath = os.path.join(user_dir, safe_filename)
        if ',' in image_b64:
            image_b64 = image_b64.split(',', 1)[1]
        image_data = base64.b64decode(image_b64)
        with open(filepath, 'wb') as f:
            f.write(image_data)
        return jsonify({"success": True, "path": filepath})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/log_study_event', methods=['POST'])
def log_study_event():
    """
    Append one structured study event to dataset/study/{username}/events.jsonl.
    Expects JSON: { username: str, event: { ... } }
    """
    try:
        data = request.get_json() or {}
        username = (data.get('username') or '').strip()
        event = data.get('event')
        if not username or not isinstance(event, dict):
            return jsonify({"error": "Missing username or event"}), 400
        project_root = os.path.dirname(INTERFACE_DIR)
        user_dir = _study_user_dir(project_root, username)
        os.makedirs(user_dir, exist_ok=True)
        events_path = _study_events_path(project_root, username)
        event.setdefault('participant_id', username)
        event.setdefault('server_received_at', datetime.datetime.utcnow().isoformat() + 'Z')
        with open(events_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=True) + '\n')
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/study_users', methods=['GET'])
def study_users():
    """Return all study usernames that have a folder under dataset/study/."""
    try:
        project_root = os.path.dirname(INTERFACE_DIR)
        study_dir = os.path.join(project_root, 'dataset', 'study')
        if not os.path.isdir(study_dir):
            return jsonify([])
        users = sorted([name for name in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, name))])
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/study_events', methods=['GET'])
def study_events():
    """Return structured study events for one user, with optional filtering."""
    try:
        username = (request.args.get('username') or '').strip()
        if not username:
            return jsonify({"error": "Missing username"}), 400
        session_id = (request.args.get('session_id') or '').strip()
        app_mode = (request.args.get('app_mode') or '').strip()
        criterion = (request.args.get('criterion') or '').strip()
        event_type = (request.args.get('event_type') or '').strip()
        project_root = os.path.dirname(INTERFACE_DIR)
        all_events = _read_study_events(project_root, username)
        filtered = all_events
        if session_id:
            filtered = [e for e in filtered if (e.get('session_id') or '') == session_id]
        if app_mode:
            filtered = [e for e in filtered if (e.get('app_mode') or '') == app_mode]
        if criterion:
            filtered = [e for e in filtered if (e.get('criterion') or '') == criterion]
        if event_type:
            filtered = [e for e in filtered if (e.get('event_type') or '') == event_type]
        return jsonify({
            "events": filtered,
            "available_sessions": sorted({e.get('session_id') for e in all_events if e.get('session_id')}),
            "available_modes": sorted({e.get('app_mode') for e in all_events if e.get('app_mode')}),
            "available_criteria": sorted({e.get('criterion') for e in all_events if e.get('criterion')}),
            "available_event_types": sorted({e.get('event_type') for e in all_events if e.get('event_type')})
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_multiple', methods=['POST'])
def predict_multiple():
    """
    Predict multiple images. Returns best_class_0 (highest confidence among class_0),
    best_class_1 (highest confidence among class_1), and all_results.
    Less = best_class_0, More = best_class_1. If no result in a class, that key is null.
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
        
        class_0_results = [r for r in results if r.get('class') == 'class_0']
        class_1_results = [r for r in results if r.get('class') == 'class_1']
        best_class_0 = max(class_0_results, key=lambda x: x['confidence']) if class_0_results else None
        best_class_1 = max(class_1_results, key=lambda x: x['confidence']) if class_1_results else None
        
        def to_item(r):
            return {"index": r['index'], "confidence": r['confidence'], "class": r['class']}
        
        return jsonify({
            "best_class_0": to_item(best_class_0) if best_class_0 else None,
            "best_class_1": to_item(best_class_1) if best_class_1 else None,
            "best": to_item(best_class_1) if best_class_1 else None,
            "worst": to_item(best_class_0) if best_class_0 else None,
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