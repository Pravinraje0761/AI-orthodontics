# AI Orthodontics: Gender Classification from OPG X-rays

This project implements a deep learning model to classify the gender of subjects based on Panoramic Dental X-rays (OPGs) and anatomical landmarks.

## Overview

The model uses a ResNet18 architecture (PyTorch) to analyze anatomical features such as the mandible, jawline, and teeth structure. It achieves high accuracy by focusing on key landmarks that differentiate male and female skeletal structures.

## Performance

- **Accuracy**: **94.20%**
- **Precision (Female)**: 0.96
- **Precision (Male)**: 0.93
- **Recall (Female)**: 0.93
- **Recall (Male)**: 0.95

## Project Structure

- `train_gender.py`: Main script for training the model with data augmentation and 80-20 split.
- `generate_heatmaps.py`: Generates Grad-CAM heatmaps to visualize anatomical focus areas.
- `best_gender_model.pth`: The trained model weights (Best performing).
- `gender_heatmaps/`: Visual explanations showing how the model identifies gender features.
- `classified_xrays/`: Sample results of re-classifying original X-rays.

## Anatomical Landmarks

The model focuses on several key landmarks defined in `Landmarks.docx`, including:
- **Gonion (Go)**: The lowest, posterior, and most outward point of the mandible.
- **Menton (Me)**: The lowest point on the symphysis of the mandible.
- **Condylion (Co)**: The most superior and posterior point on the head of the condyle.

## Usage

### Training
```bash
python train_gender.py
```

### Visualizing Explanations (Heatmaps)
```bash
python generate_heatmaps.py
```

## Requirements
- PyTorch
- Torchvision
- Scikit-learn
- OpenCV
- Matplotlib
- Python-docx (for landmark extraction)
