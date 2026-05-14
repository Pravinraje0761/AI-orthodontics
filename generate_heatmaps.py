import os
import sys
print("STARTING HEATMAP GENERATION", flush=True)
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2
from PIL import Image

# Parameters
IMG_SIZE = 224
DATA_DIR = r'c:\Users\pravi\OneDrive\Desktop\AI_ORTHODONTICS\OPG With landmarks-20260514T070507Z-3-001\OPG With landmarks'
MODEL_PATH = 'best_gender_model.pth' # Or final_gender_model.pth

# Grad-CAM Class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output)
            
        output[0, class_idx].backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        return heatmap

def generate():
    print("INSIDE GENERATE", flush=True)
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found. Wait for training to finish.")
        return

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    classes = full_dataset.classes

    # Model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # Heatmaps output dir
    output_dir = 'gender_heatmaps'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use the last convolutional layer
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    print("Generating heatmaps for random samples...")
    for i in range(10):
        idx = np.random.randint(len(full_dataset))
        img_path, label = full_dataset.samples[idx]
        
        # Load raw image
        raw_img = cv2.imread(img_path)
        raw_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))
        
        # Prepare input
        input_img, _ = full_dataset[idx]
        input_tensor = input_img.unsqueeze(0)
        
        # Grad-CAM
        heatmap = grad_cam(input_tensor)
        
        # Overlay
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(raw_img, 0.6, heatmap_color, 0.4, 0)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = output.max(1)
            pred_class = classes[pred.item()]
            actual_class = classes[label]

        # Save
        save_path = os.path.join(output_dir, f'heatmap_{i}_{pred_class}_actual_{actual_class}.png')
        cv2.imwrite(save_path, superimposed_img)
        print(f"Saved: {save_path}")

    print("Done.")

if __name__ == "__main__":
    generate()
