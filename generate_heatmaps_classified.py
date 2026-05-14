import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Ensure output is flushed immediately
def log(message):
    print(message, flush=True)

# Parameters
IMG_SIZE = 224
MODEL_PATH = 'best_gender_model.pth'
CLASSIFIED_DIR = r'C:\Users\pravi\OneDrive\Desktop\AI_ORTHODONTICS\classified_xrays'
OUTPUT_DIR = r'C:\Users\pravi\OneDrive\Desktop\AI_ORTHODONTICS\new_heatmaps_classified'

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

def generate_new_heatmaps():
    if not os.path.exists(MODEL_PATH):
        log(f"Error: Model file {MODEL_PATH} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Model
    log("Initializing model...")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Use the last convolutional layer
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    log(f"Generating new heatmaps from {CLASSIFIED_DIR}...")
    
    # Pick 5 females and 5 males from classified_xrays
    count = 0
    for gender in ['female', 'male']:
        gender_dir = os.path.join(CLASSIFIED_DIR, gender)
        if not os.path.exists(gender_dir):
            continue
            
        img_names = [f for f in os.listdir(gender_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Pick up to 5 random images
        selected = np.random.choice(img_names, min(5, len(img_names)), replace=False)
        
        for img_name in selected:
            img_path = os.path.join(gender_dir, img_name)
            
            try:
                # Load raw image
                raw_img = cv2.imread(img_path)
                raw_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))
                
                # Prepare input
                img_pil = Image.open(img_path).convert('RGB')
                input_tensor = transform(img_pil).unsqueeze(0).to(device)
                
                # Grad-CAM
                heatmap = grad_cam(input_tensor)
                
                # Overlay
                heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(raw_img, 0.6, heatmap_color, 0.4, 0)
                
                # Save
                save_name = f"heatmap_classified_{gender}_{img_name}"
                save_path = os.path.join(OUTPUT_DIR, save_name)
                cv2.imwrite(save_path, superimposed_img)
                log(f"  Saved: {save_name}")
                count += 1
                
            except Exception as e:
                log(f"  Error processing {img_name}: {e}")

    log(f"--- Task Completed ---")
    log(f"Total new heatmaps generated: {count}")
    log(f"Results are available in: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_new_heatmaps()
