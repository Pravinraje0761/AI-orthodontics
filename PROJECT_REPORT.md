# AI-Based Gender Identification from OPG Radiographs with Explainable Localization

## 1) Project Objective

Build a machine learning pipeline to identify gender from OPG (orthopantomogram) X-ray images, improve dataset size through augmentation, train supervised models, and generate visual explainability outputs that highlight clinically relevant bright/radiopaque regions used by the model.

## 2) Dataset Summary

- Initial source images: `17` total
  - `MALE`: `7`
  - `FEMALE`: `10`
- Image formats detected: `.jpg`, `.png`
- Main dataset root used: `opg with landmarks`

## 3) Work Performed (End-to-End)

### 3.1 Code Repair and Pipeline Stabilization

The existing Python pipeline files contained major syntax/runtime errors and undefined references. These were repaired by restructuring files into valid, reusable modules and scripts:

- `complete_pipeline.py` rebuilt into a clean, executable end-to-end pipeline.
- `feature_selection.py` refactored into a callable feature-selection function.
- `statistical_analysis.py` refactored into a callable statistical-analysis function.
- `model_training_evaluation.py` refactored into callable training/evaluation logic.
- `model_saving_deployment.py` fixed for safe model save/load usage.
- Lint/diagnostic check completed: errors reduced to zero for edited files.

### 3.2 Image Augmentation

Created augmentation utility:

- Script: `image_augmentation.py`

Generated augmentation sets:

1. Single-folder augmented set:
   - Output: `opg with landmarks/augmented_800`
   - Generated: `800` images

2. Class-wise balanced augmented set:
   - Output: `opg with landmarks/augmented_classwise`
   - Generated:
     - `FEMALE`: `400`
     - `MALE`: `400`
   - Total: `800`

### 3.3 Unsupervised Training (Autoencoder)

Because labels were initially unavailable in structured form, an unsupervised autoencoder was trained first:

- Script: `train_autoencoder_opg.py`
- Input: `opg with landmarks/augmented_800`
- Split: `80/20`
  - Train: `640`
  - Validation: `160`

Final autoencoder metrics:

- Train loss (MSE): `0.002007`
- Validation loss (MSE): `0.001900`

Saved artifacts:

- `models/opg_autoencoder.keras`
- `models/autoencoder_train_loss.npy`
- `models/autoencoder_val_loss.npy`

### 3.4 Supervised Training (70/30)

After class folders were prepared (`MALE`, `FEMALE`), supervised training was performed:

- Script: `train_supervised_opg.py`
- Split: `70/30` (standard directory split)
- Observed high metrics (~98%+) due to augmentation leakage risk.

### 3.5 Leakage-Safe Re-Training (Grouped Split by Original ID)

To produce trustworthy metrics, grouped splitting was introduced so all augmentations from a single original image remain in only one split:

- Script: `train_supervised_grouped_split.py`
- Strategy: grouped split on base ID before `_aug_`
- Split target: `70/30` (group-safe)

Model comparison on same grouped split:

- RandomForest validation accuracy: `0.6392`
- SVM (RBF) validation accuracy: `0.6564` (best)

Best grouped-split metrics (SVM):

- Train samples: `509`
- Validation samples: `291`
- Train accuracy: `1.0000`
- Validation accuracy: `0.6564`
- Precision: `0.5455`
- Recall: `1.0000`
- F1-score: `0.7059`

Interpretation:

- This lower accuracy is expected and more realistic than leakage-inflated results.
- Perfect train accuracy indicates overfitting due to very limited independent originals.

### 3.6 Visual Reporting with Matplotlib

Generated report files using matplotlib:

- `results/grouped_split_metrics.txt`
- `results/model_comparison_accuracy.png`
- `results/confusion_matrix_grouped_split.png`

### 3.7 Explainability / Localization Outputs

Multiple explainability attempts were made. Final accepted output is white-region-focused small-batch localization:

- Script: `generate_focused_explanations.py`
- Method: SmoothGrad-based saliency + white-intensity-guided heatmap + highlighted focus region
- Final output folder:
  - `results/focused_explanations_whitecam_20_each`
  - `FEMALE`: `20` images
  - `MALE`: `20` images
  - Total: `40` images

Cleanup performed:

- Removed older/unwanted explanation folders:
  - `results/focused_explanations`
  - `results/focused_explanations_whitecam`
  - `results/gradcam_outputs`
  - `results/gradcam_outputs_original`

## 4) Final Deliverables

### Code Files

- `complete_pipeline.py`
- `image_augmentation.py`
- `train_autoencoder_opg.py`
- `train_supervised_opg.py`
- `train_supervised_grouped_split.py`
- `generate_focused_explanations.py`

### Model / Numeric Artifacts

- `models/opg_autoencoder.keras`
- `models/autoencoder_train_loss.npy`
- `models/autoencoder_val_loss.npy`

### Reports / Visual Outputs

- `results/grouped_split_metrics.txt`
- `results/model_comparison_accuracy.png`
- `results/confusion_matrix_grouped_split.png`
- `results/focused_explanations_whitecam_20_each/*`

## 5) Recommended Project Header (for presentation/report)

Use this as your official project topic header:

**AI-Based Gender Identification from OPG Radiographs with Explainable Localization of Diagnostic Regions**

## 6) Key Conclusion

The project successfully implemented data augmentation, supervised and unsupervised modeling, leakage-safe evaluation, and explainable localization outputs. While performance on leakage-safe evaluation is moderate (~65.64% validation accuracy), the workflow is now technically sound and ready for improvement with more independent labeled OPG images.
