# EKPHRASIS – Generating Visual Aids for Graphic Design Education

EKPHRASIS is an interactive system that helps learners build intuition for graphic design vocabulary (e.g. “visual balance”) by creating compositions and receiving “less” and “more” effective examples from an ML model. Users draw on a canvas; the system shows reference images and supports multiple modes: **Visual Aids**, **Chess** (turn-based with a computer), **Test Model**, and **Label Data** for collecting training data.

This repository accompanies the paper **“Generating Visual Aids to Help Students Understand Graphic Design with EKPHRASIS”** (CHI EA ’25).

- **Paper (open access):** [https://doi.org/10.1145/3706599.3719807](https://doi.org/10.1145/3706599.3719807)

## Citation

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

- **Canvas** – Draw and arrange grey rectangles; undo/redo, export.
- **Visual Aids** – Generate “less” and “more” examples for the selected criterion. Left = highest-confidence **class_0** (Less), right = highest-confidence **class_1** (More). If only one class appears after several tries, the other slot shows the lowest-confidence variant. Feedback text and reference images are shown only in this mode.
- **Chess** – Turn-based: you add a block, then the computer adds one. Computer strategies: **Help** (picks highest-confidence class_1), **Oppose** (highest-confidence class_0), **Random**. “Reached” threshold is configurable in the ML status modal (default 80%).
- **Test Model** – Upload images or use `dataset/test_images`; run the current criterion and see image + score (e.g. “45.2% More”) per result.
- **Label Data** – Label random compositions as “does not fit” (0) or “fits” (1). Saves to `dataset/new/{vocabulary}_{date}_{labeler}/class_0|class_1`. Optional **Crop Dataset** to balance classes; keyboard ← / → for 0 / 1; labeler name cached in browser.
- **Demo vs Study** – Bottom bar: **Demo** (no logging) or **Study**. In Study mode, a **User** field appears; the app records (1) saved images under `dataset/study/{username}/` and (2) structured interaction events to `dataset/study/{username}/events.jsonl` (JSONL). Useful for HCI experiments.
- **Analytics** – A dedicated mode to inspect Study logs: filter by user/session/mode/vocabulary/event type, view summaries and distributions, click an event to see a right-side detail panel (composition redrawn from `state_snapshot` plus saved feedback images when available), and export the filtered events as JSON/CSV.
- **Persistent state** – Mode, Demo/Study choice, and username are stored in the browser (localStorage). Refresh keeps the same state. **Reset mode & username** in the ML status modal clears this cache.

## System architecture

```
EKPHRASIS/
├── interface/
│   ├── interface.html    # Single-page UI (canvas, modes, modals)
│   └── logo.png
├── ml/
│   ├── model_server.py   # Flask API (predict, predict_multiple, labels, etc.)
│   ├── train_and_save_model.py  # Train one model per vocabulary
│   ├── start_server.py   # Start server on port 5001
│   ├── models/           # One .h5 per vocabulary (e.g. visual_balance.h5)
│   └── requirements.txt
├── dataset/
│   ├── <vocabulary_id>/  # Training data: class_0/, class_1/ (e.g. visual_balance/)
│   ├── new/              # New labels: <vocabulary>_<date>_<labeler>/class_0|class_1
│   ├── study/            # Study recordings: <username>/*.png + events.jsonl (when using Study mode)
│   └── test_images/      # Images for Test Model “Run dataset/test_images”
├── start_ekphrasis.py    # Optional: start server and open browser
└── README.md
```

## Quick start

### 1. Backend

```bash
cd ml
pip install -r requirements.txt
python train_and_save_model.py visual_balance   # needs dataset/visual_balance/class_0, class_1
python start_server.py
```

Server runs at **http://localhost:5001** (port 5001 to avoid conflict with macOS AirPlay on 5000).

### 2. Interface

Open **http://localhost:5001/** in a browser (the backend serves the interface). Or from the project root:

```bash
python start_ekphrasis.py
```

### 3. Using the app

- **Practice vocabulary** (bottom bar): choose the criterion (e.g. Visual Balance).
- **Mode**: Canvas, Visual Aids, Chess, Test Model, Label Data, Analytics.
- **Demo / Study** (bottom bar, left of ML): Demo = no logging; Study = record to `dataset/study/{username}/`. In Study, enter **User** (username); each Generate (Visual Aids) or move (Chess) saves composition and system feedback with timestamped filenames.
- **ML status** (bottom bar): Button always shows “ML”; dot color indicates state (green = connected, yellow = disconnected, gray = checking). Click to open modal (server message, model info, variation range, Chess reached threshold, Show model scores, **Reset mode & username**).
- In **Visual Aids**: draw, click “Generate visual aids”; left/right show Less and More examples; feedback in the message area when applicable.
- In **Label Data**: set Vocabulary and Labeler (labeler is cached); use buttons or ← / → to label; optionally “Crop Dataset” to balance classes.

**Requirements:** Python 3.7+, browser, ~4GB+ RAM for the model.

## How it works

### Frontend (`interface/interface.html`)

- Single HTML file: canvas, mode-specific UI, modals (settings, about, backend status).
- **Visual Aids:** Builds user canvas + 20 variations, calls `POST /predict_multiple`; retries up to 10 times until both class_0 and class_1 appear (or uses lowest-confidence variant for the missing slot). Displays Less = best_class_0, More = best_class_1. In Study mode with username, saves composition + visualaids-less/more images via `POST /save_study`.
- **Chess:** After each user block, calls `/predict` for current composition; if below “reached” threshold (default 80%, set in ML modal), computer adds a block via `/predict_multiple` on 20 candidates; Help uses best_class_1, Oppose uses best_class_0. In Study mode, saves composition and result images to `dataset/study/{username}/`.
- **Study event logging (JSONL):** In Study mode, user and system actions are appended to `dataset/study/{username}/events.jsonl` via `POST /log_study_event`. Events include timestamps, `session_id`, `app_mode`, `criterion`, `event_type`, `state_id`, and a `payload` (e.g., `state_snapshot`, `block_id`, and saved image filenames when applicable).
- **Analytics mode:** Fetches available study users and events from the backend and provides filtering, event timeline + detail view, and export (JSON/CSV) of the currently filtered events.
- **Test Model:** Choose file or “Run dataset/test_images”; sends images to `/predict_multiple` and shows each image with score and class (More/Less).
- **Label Data:** Sends vocabulary, labeler, label (class_0/class_1), image to `POST /save_label`; counts from `GET /label_counts`; balance via `POST /balance_labels`.

### Backend (`ml/model_server.py`)

- **Prediction:** VGG16 backbone + small head; input 224×224 RGB, output confidence in [0,1]; class_0 if &lt; 0.5, class_1 otherwise.
- **`/predict_multiple`:** Returns **best_class_0** (among class_0 predictions, highest confidence), **best_class_1** (among class_1, highest confidence), plus `all_results`. Used for Less/More and for Chess Help/Oppose.

### Data

- **Training:** `dataset/<vocabulary_id>/class_0/` and `class_1/` (images). Train with `python train_and_save_model.py <vocabulary_id>`.
- **New labels:** `dataset/new/<vocabulary>_<date>_<labeler>/class_0/` and `class_1/`. Copy or move into `dataset/<vocabulary_id>/` when ready to retrain.
- **Study recordings:** `dataset/study/<username>/`. In Study mode, images are saved under this folder and structured events are appended to `events.jsonl` (one JSON object per line). Filenames follow `{username}-v-composition-{n}-{timestamp}-{class}-{score}.png`, `{username}-v-visualaids-less|more-{n}-{timestamp}-{score}.png` (Visual Aids), and `{username}-chess-composition|result-{n}-{timestamp}-{class}-{score}.png` (Chess). Ignored by git via `.gitignore`.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve `interface/interface.html` |
| GET | `/health` | Server and model status |
| GET | `/criteria` | List vocabularies (id, name) for dropdown |
| POST | `/predict` | Single image → confidence, class |
| POST | `/predict_multiple` | Multiple images → best_class_0, best_class_1, all_results |
| GET | `/label_counts` | Query `vocabulary`, `labeler` → counts for today’s run folder |
| POST | `/save_label` | Body: vocabulary, labeler, label (class_0\|class_1), image (base64) |
| POST | `/balance_labels` | Body: vocabulary, labeler; crop larger class to match smaller |
| POST | `/save_study` | Body: username, filename, image (base64); save to `dataset/study/{username}/{filename}` |
| POST | `/log_study_event` | Body: username, event (object); append one JSONL event to `dataset/study/{username}/events.jsonl` |
| GET | `/study_users` | List usernames that have a `dataset/study/{username}/` folder |
| GET | `/study_events` | Query: username + optional filters (`session_id`, `app_mode`, `criterion`, `event_type`); returns matching events + available filter values |
| GET | `/study_image` | Query: username, filename; return one saved study image (read-only) |
| GET | `/test_images` | List images in `dataset/test_images` as base64 |

## Model

- **Architecture:** VGG16 (ImageNet, frozen) + Flatten + Dense(128) + Dense(1, sigmoid).
- **Training:** `train_and_save_model.py`; reads `dataset/<vocabulary_id>/class_0` and `class_1`; 80/20 train/val; class weights only when imbalanced; saves `ml/models/<vocabulary_id>.h5`.
- **Adding a vocabulary:** Add folder `dataset/<id>/class_0`, `class_1`; run `python train_and_save_model.py <id>`; add `{'id':'<id>','name':'...','model_file':'models/<id>.h5'}` to `CRITERIA` in `model_server.py`; restart server. The system ships with six first-tier vocabularies (Visual Balance, Visual Harmony, Visual Hierarchy, Contrast, Rhythm, Emphasis); only those with trained models under `ml/models/<id>.h5` support prediction; all appear in the dropdown so you can label data for any of them.

## Testing

```bash
cd ml
python test_system.py
```

## Troubleshooting

- **Server won’t start:** Port 5001 in use or blocked.
- **Model not found:** Run `python train_and_save_model.py visual_balance` (requires `dataset/visual_balance/class_0` and `class_1`).
- **Frontend can’t connect:** Use http://localhost:5001/ (not file://).
- **“Could not get both Less and More”:** Now handled by showing one class and using the lowest-confidence variant for the other slot; if you still see errors, check backend logs.

## Credits

EKPHRASIS by Bob Tianqi Wei, Shayne Shen, Shm Garanganao Almeda, and Bjoern Hartmann, UC Berkeley. CHI EA ’25.

## License

This project is for research and education.
