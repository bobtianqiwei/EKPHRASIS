# EKPHRASIS ML Module

Machine learning components for composition analysis. One trained model per design vocabulary.

## Layout

- `model_server.py` – Flask API (predict, predict_multiple, criteria, health). Loads all `.h5` models listed in `CRITERIA`.
- `train_and_save_model.py` – Train one vocabulary: `python train_and_save_model.py <vocabulary_id>` (e.g. `visual_balance`). Reads `../dataset/<vocabulary_id>/class_0` and `class_1`, saves `models/<vocabulary_id>.h5`.
- `start_server.py` – Starts the server (checks that `models/` has at least one `.h5`).
- `models/` – One `.h5` file per vocabulary (e.g. `visual_balance.h5`).
- `requirements.txt` – Python dependencies.

## Dataset (per vocabulary)

```
dataset/<vocabulary_id>/
├── class_0/   # Less [criterion] images
└── class_1/   # More [criterion] images
```

Example: `dataset/visual_balance/class_0/`, `dataset/visual_balance/class_1/`. Label mapping: class_0 → 0, class_1 → 1. App shows high confidence = More (right), low = Less (left).

## Train a vocabulary

```bash
cd ml
pip install -r requirements.txt
python train_and_save_model.py visual_balance
```

Then add the criterion in `model_server.py` if not already in `CRITERIA`, and restart the server.

## API

- **GET** `/health` – Server and model status.
- **GET** `/criteria` – List of vocabularies (id, name) whose model loaded.
- **POST** `/predict` – Single image, returns confidence and class.
- **POST** `/predict_multiple` – List of images, returns best/worst indices and scores.

## Troubleshooting

- **No model found**: Run `python train_and_save_model.py visual_balance` (or your vocabulary id). Ensure `dataset/visual_balance/class_0` and `class_1` exist.
- **Import errors**: `pip install -r requirements.txt`
- **Port**: Server uses 5001 (macOS AirPlay often uses 5000).
