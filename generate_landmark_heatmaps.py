import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import random

# Parameters
IMG_SIZE = 224
MODEL_PATH = 'best_gender_model.pth'
CLASSIFIED_DIR = r'C:\Users\pravi\OneDrive\Desktop\AI_ORTHODONTICS\classified_xrays'
OUTPUT_DIR = r'C:\Users\pravi\OneDrive\Desktop\AI_ORTHODONTICS\classified_landmark_analysis'

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

def get_landmark_regions(img_size):
    """
    Refined heuristic regions based on user's reference image for high anatomical accuracy.
    Values are percentages of img_size (y_min, y_max, x_min, x_max).
    """
    return {
        "Gonial Angle": [(0.70, 0.88, 0.05, 0.22), (0.70, 0.88, 0.78, 0.95)],
        "Coronoid Height": [(0.30, 0.75, 0.05, 0.22), (0.30, 0.75, 0.78, 0.95)],
        "Condylar Height": [(0.30, 0.75, 0.05, 0.22), (0.30, 0.75, 0.78, 0.95)],
        "Mental Foramen": [(0.75, 0.90, 0.25, 0.40), (0.75, 0.90, 0.60, 0.75)],
        "Intercondylar Dist": [(0.10, 0.25, 0.15, 0.85)],
        "Sigmoid Notch": [(0.20, 0.38, 0.12, 0.28), (0.20, 0.38, 0.72, 0.88)]
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 2))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    grad_cam = GradCAM(model, model.layer4[-1])
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    landmark_defs = get_landmark_regions(IMG_SIZE)
    
    # Pick 10 images
    samples = []
    for g in ['female', 'male']:
        g_dir = os.path.join(CLASSIFIED_DIR, g)
        files = [f for f in os.listdir(g_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        chosen = random.sample(files, 5)
        for f in chosen:
            samples.append((os.path.join(g_dir, f), g))

    for i, (path, actual_gender) in enumerate(samples):
        print(f"Processing image {i+1}: {path}")
        img_pil = Image.open(path).convert('RGB')
        raw_img = cv2.imread(path)
        raw_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))
        
        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = output.max(1)
            pred_gender = 'female' if pred.item() == 0 else 'male'

        # Full Heatmap
        full_heatmap = grad_cam(input_tensor)
        full_heatmap_resized = cv2.resize(full_heatmap, (IMG_SIZE, IMG_SIZE))
        
        # Create a large composite image
        # Top row: Original, Full Heatmap
        # Next rows: 6 landmarks
        canvas_w = IMG_SIZE * 3
        canvas_h = IMG_SIZE * 3
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Helper to put image on canvas
        def put_img(img, r, c, label):
            y, x = r * IMG_SIZE, c * IMG_SIZE
            canvas[y:y+IMG_SIZE, x:x+IMG_SIZE] = img
            cv2.putText(canvas, label, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Row 0
        put_img(raw_img, 0, 0, f"Original ({actual_gender})")
        
        heatmap_color = cv2.applyColorMap(np.uint8(255 * full_heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(raw_img, 0.6, heatmap_color, 0.4, 0)
        put_img(overlay, 0, 1, f"Full Model Attention")
        
        # Prediction Info
        info_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        cv2.putText(info_img, f"Pred: {pred_gender}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info_img, f"Actual: {actual_gender}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        put_img(info_img, 0, 2, "Info")

        # Row 1 & 2: Individual Landmarks
        row, col = 1, 0
        for name, regions in landmark_defs.items():
            # Create a mask for this landmark
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            for (y1, y2, x1, x2) in regions:
                mask[int(y1*IMG_SIZE):int(y2*IMG_SIZE), int(x1*IMG_SIZE):int(x2*IMG_SIZE)] = 1.0
            
            # Combine mask with model attention to show "Model focus on this landmark"
            # Apply a Gaussian blur to the mask to create a "dispersed" effect
            mask_blurred = cv2.GaussianBlur(mask, (51, 51), 0)
            
            lm_attention = full_heatmap_resized * mask_blurred
            if np.max(lm_attention) > 0:
                lm_attention /= np.max(lm_attention)
            
            lm_heatmap = cv2.applyColorMap(np.uint8(255 * lm_attention), cv2.COLORMAP_JET)
            lm_overlay = cv2.addWeighted(raw_img, 0.7, lm_heatmap, 0.3, 0)
            
            # Removed green circles for a cleaner, dispersed look
            
            put_img(lm_overlay, row, col, name)
            col += 1
            if col == 3:
                col = 0
                row += 1

        save_path = os.path.join(OUTPUT_DIR, f"analysis_{i}_{actual_gender}.png")
        cv2.imwrite(save_path, canvas)
        print(f"Saved analysis to {save_path}")

if __name__ == "__main__":
    main()
