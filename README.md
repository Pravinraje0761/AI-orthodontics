# Sex Classification from Panoramic Dental X-rays (OPG)

[![CI](https://github.com/Pravinraje0761/AI-orthodontics/actions/workflows/ci.yml/badge.svg)](https://github.com/Pravinraje0761/AI-orthodontics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A ResNet18 classifier (PyTorch, ImageNet-pretrained) that predicts sex from orthopantomograms,
with Grad-CAM heatmaps to show which mandibular regions drive each prediction. The landmarks
of interest (gonion, menton, condylion) are defined in `Landmarks.docx`.

## Evaluation status

**There is no valid accuracy figure yet.** An earlier version of this README reported 94.2%
test accuracy. That figure is withdrawn, for two reasons found in the original training code:

1. **Train/test leakage.** The dataset holds about 9 pre-augmented copies of each of 152
   source X-rays (`04.png`, `04_aug_0005.png`, ...), and the split was random over images.
   Reproducing that split shows **every one of the 276 test images had a copy of the same
   source X-ray in the training set**, so the test measured recognition of seen X-rays.
2. **Selection on the test set.** The checkpoint saved was the one with the best test accuracy.

`train_gender.py` now splits by source X-ray into train / validation / test (`splits.py`, no
X-ray shared between splits), chooses the checkpoint on validation, and scores the test set
once. Its headline metric uses the original test X-rays only, since augmented copies are not
independent samples. Results from a re-run will be added here.

With about 31 test X-rays, any accuracy estimate will have a wide confidence interval
(about ±14 points at 80% accuracy, 95% CI); report it with that interval.

## Usage

```bash
pip install -r requirements.txt

# Training data: one folder per class, see data/README.md
python train_gender.py --data-dir "data/OPG With landmarks" --epochs 3
#   -> outputs/best_gender_model.pth, outputs/metrics.json

# Grad-CAM heatmaps (paths configurable with OPG_DATA_DIR, OPG_MODEL_PATH, OPG_OUTPUT_DIR)
python generate_heatmaps.py
python generate_heatmaps_classified.py
python generate_landmark_heatmaps.py
```

## Tests

```bash
pytest
```

The tests check that no source X-ray crosses splits, and run the full training script for one
epoch on a tiny synthetic dataset (CPU, no downloads). CI runs them on every push.

## Data

The X-ray images are patient data. The training script reads them from `data/` (see
[`data/README.md`](data/README.md) for the layout). Before keeping any images public, record
their source, the consent or ethics approval that covers release, and a licence, as listed in
that file.

## Project structure

| File | Purpose |
|---|---|
| `train_gender.py` | Training with leak-free grouped split; writes the model and `metrics.json` |
| `splits.py` | Groups augmented copies with their source X-ray; stratified grouped splitting |
| `generate_heatmaps*.py`, `generate_landmark_heatmaps.py` | Grad-CAM visualisation |
| `Landmarks.docx` | Anatomical landmark definitions |
| `tests/` | Split-integrity and end-to-end smoke tests |

## License

Code: [MIT](LICENSE). The images are not covered by this licence.
