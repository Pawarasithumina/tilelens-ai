# TileLens AI

**Deep Learning-Based Ceramic Tile Inspection System**

TileLens AI is a local computer vision application for identifying ceramic tiles and detecting visible surface defects in images. It combines a MobileNetV3-Small binary image classifier with a YOLO11n object detector in a Streamlit dashboard designed for individual and batch inspection.

> **Project status:** Functional local prototype. Predictions are experimental and should be independently verified before any production or quality-control use.

## Problem and objective

Visual ceramic tile inspection can be time-consuming and inconsistent. TileLens AI explores a two-stage deep learning workflow that first checks whether an uploaded image contains a tile and then examines tile images for visible defects. The goal is to provide an accessible tool for reviewing inspection results, not to replace professional quality assurance.

## Two-stage architecture

```text
Uploaded image(s)
       |
       v
Stage 1: MobileNetV3-Small
Tile vs. Not Tile classification
       |
       +--- Not Tile ---> Flag / do not perform tile defect detection
       |
       v
Stage 2: YOLO11n
Tile defect detection
       |
       v
Inspection results in Streamlit dashboard
```

The YOLO11n detector covers three annotated defect categories: **edge-chipping**, **hole**, and **line**. Inferences indicate possible visible defects; they are not guarantees that a tile is defect-free.

## Key features

- Tile / non-tile image classification using MobileNetV3-Small.
- Object detection for three visible ceramic tile defect categories using YOLO11n.
- Single-image and multi-image inspection workflows.
- Local, interactive Streamlit dashboard.
- Dashboard components for analytics, inspection history, and reports.
- Included small trained model checkpoints for local inference.
- Reproducible local setup and launch scripts for Ubuntu/Linux.

## Model evaluation

The following numbers come from the project's saved test evaluation JSON files. They refer to the specified test sets, **not** general performance guarantees.

### Stage 1: tile classifier

| Metric | Recorded result |
|---|---:|
| Architecture | MobileNetV3-Small |
| Classification classes | `not_tile`, `tile` |
| Test images | 440 |
| Tile images | 321 |
| Non-tile images | 119 |
| Test accuracy | 100.00% |
| Test macro F1-score | 1.0000 |
| Best training epoch | 3 |

A perfect score on this test set does **not** prove perfect generalization. Further testing on independent images, different tile types, lighting conditions, and camera setups is needed.

### Stage 2: defect detector

| Metric | Recorded result |
|---|---:|
| Architecture | YOLO11n |
| Test images | 224 |
| Evaluation image size | 640 |
| Mean precision | 60.72% |
| Mean recall | 58.49% |
| mAP@50 | 57.78% |
| mAP@50–95 | 29.40% |

**Per-class mAP@50–95:**

| Defect category | mAP@50–95 |
|---|---:|
| Edge-chipping | 32.48% |
| Hole | 43.20% |
| Line | 12.53% |

The detector has meaningful limitations, especially for the **line** category. Missed defects and false positive detections are possible.

Evaluation artifacts can be found under `models/tile_classifier/` and `models/tile_damage_detector/`.

## Project structure

```text
tilelens-ai/
├── dashboard/
│   ├── app.py
│   ├── inference.py
│   ├── components/
│   └── utils/
├── models/
│   ├── tile_classifier/
│   └── tile_damage_detector/
├── src/
│   └── preprocessing/
├── .streamlit/
│   └── config.toml
├── requirements-dashboard.txt
├── setup.sh
├── run.sh
└── README.md
```

The original training data and local inspection history are excluded from the public repository.

## Installation and local usage

This project is designed to run locally on Ubuntu/Linux with **Python 3** and support for Python virtual environments (`python3-venv`). The setup script installs CPU-compatible PyTorch and the required dashboard packages. An internet connection is required for installation.

```bash
git clone https://github.com/Pawarasithumina/tilelens-ai.git
cd tilelens-ai
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

After startup, open **http://localhost:8501** in a browser.

`setup.sh` creates the `.venv` environment, installs CPU PyTorch and dashboard dependencies, and checks package compatibility. `run.sh` verifies that the model checkpoint files exist before starting Streamlit.

**Bundled model checkpoints:**

- `models/tile_classifier/mobilenetv3_small_tile_v2.pt`
- `models/tile_damage_detector/yolo11n_tile_damage_v1_best.pt`

## Data, reproducibility, and responsible use

- Raw datasets, training archives, and private inspection history are not distributed in this repository.
- Preprocessing scripts and saved evaluation artifacts are available, but retraining may require acquiring the original datasets and checking their licenses.
- Model scores come from internal held-out test sets; independent external validation is still necessary.
- The application is a research/development prototype, not a certified industrial inspection system.
- Users should manually verify critical inspection decisions and respect the licenses of datasets, models, and dependencies.

## Future improvements

- Test on external ceramic tile images captured in different real-world environments.
- Improve difficult defect categories, especially fine line defects.
- Evaluate confidence thresholds, error cases, and robustness under changes in lighting and background.
- Enhance deployment packaging and documentation for broader use.

## Author

**Pawara Sithumina** — [GitHub](https://github.com/Pawarasithumina)
