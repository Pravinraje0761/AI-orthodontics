# AI-Based Gender Identification from OPG Radiographs

Machine learning experiments on orthopantomogram (OPG) X-rays: augmentation, unsupervised and supervised models, leakage-aware evaluation, and explainable localization (saliency / white-region–focused maps).

**Author:** [@Pravinraje0761](https://github.com/Pravinraje0761)

> **Note:** This repository is for research and education. Outputs are not validated for clinical diagnosis or patient care.

---

## What’s in this repo

| Area | Description |
|------|-------------|
| **Tabular pipeline** | Landmark-based features → statistics, feature work, classical ML (`complete_pipeline.py`, `feature_selection.py`, `statistical_analysis.py`, `model_training_evaluation.py`, `model_saving_deployment.py`). |
| **Images** | OPGs under `opg with landmarks/` (`MALE` / `FEMALE`), plus augmented sets (e.g. `augmented_800`, `augmented_classwise`). |
| **Deep learning** | Autoencoder pretraining (`train_autoencoder_opg.py`), supervised CNN training (`train_supervised_opg.py`), grouped split for augmentation leakage control (`train_supervised_grouped_split.py`). |
| **Explainability** | SmoothGrad-style saliency and Grad-CAM–related utilities (`generate_focused_explanations.py`, `generate_gradcam_outputs.py`). |
| **Reporting** | Written summary in [`PROJECT_REPORT.md`](PROJECT_REPORT.md); figures and metrics under `results/`. |
| **Saved models** | Keras / NumPy artifacts in `models/`. |

---

## Requirements

- Python **3.10+** (recommended)
- Core libraries used across scripts:

  `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `Pillow`, `tensorflow`

Install dependencies (example):

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn Pillow tensorflow
```

Adjust TensorFlow install if you use GPU or a specific platform build ([tensorflow.org/install](https://www.tensorflow.org/install)).

---

## Repository layout (high level)

```
AI_ORTHODONTICS/
├── README.md
├── PROJECT_REPORT.md          # Detailed methodology and results
├── complete_pipeline.py       # End-to-end tabular ML pipeline
├── image_augmentation.py      # OPG augmentation CLI
├── train_autoencoder_opg.py
├── train_supervised_opg.py
├── train_supervised_grouped_split.py
├── generate_focused_explanations.py
├── generate_gradcam_outputs.py
├── feature_selection.py
├── statistical_analysis.py
├── model_training_evaluation.py
├── model_saving_deployment.py
├── models/                    # Trained checkpoints / loss curves
├── opg with landmarks/       # Image data and augmented folders
└── results/                   # Metrics, plots, explanation outputs
```

---

## Quick start

From the project root:

1. **Augment images** (see script help for paths and counts):

   ```bash
   python image_augmentation.py --help
   ```

2. **Train autoencoder** (expects augmented images under your configured folder):

   ```bash
   python train_autoencoder_opg.py
   ```

3. **Supervised training with grouped split** (reduces train/val leakage from augmented copies of the same source):

   ```bash
   python train_supervised_grouped_split.py
   ```

4. **Focused explanation maps** (after a supervised model is available):

   ```bash
   python generate_focused_explanations.py --help
   ```

5. **Tabular landmark pipeline** (requires your CSV path inside `complete_pipeline.py` or as wired in your copy):

   ```bash
   python complete_pipeline.py
   ```

Exact defaults (paths, image sizes, epochs) are defined in each script; open the file or run `--help` where supported.

---

## Methods highlights

- **Augmentation:** geometric and photometric transforms via Pillow (`image_augmentation.py`).
- **Leakage-aware CV:** splits group IDs so all `_aug_*` variants from one original stay in one fold (`train_supervised_grouped_split.py`).
- **Explainability:** SmoothGrad-based saliency combined with intensity-guided highlighting (`generate_focused_explanations.py`).

For numeric results, confusion matrices, and interpretation, see **`PROJECT_REPORT.md`**.

---

## License

If you add a license (e.g. MIT), place it in `LICENSE` and describe it here.
