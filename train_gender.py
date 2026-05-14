import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import sys

# Ensure output is flushed immediately
def log(message):
    print(message, flush=True)

log("--- Starting Training Process ---")

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 3 # Reduced for safety and speed
DATA_DIR = r'c:\Users\pravi\OneDrive\Desktop\AI_ORTHODONTICS\OPG With landmarks-20260514T070507Z-3-001\OPG With landmarks'

# 1. Data Augmentation to prevent "memorizing" (overfitting)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_model():
    if not os.path.exists(DATA_DIR):
        log(f"Error: Data directory not found at {DATA_DIR}")
        return

    # Load full dataset first without transform to split indices
    full_dataset = datasets.ImageFolder(DATA_DIR)
    targets = full_dataset.targets
    classes = full_dataset.classes
    log(f"Total images found: {len(full_dataset)}")
    log(f"Classes: {classes}")

    # 2. 80-20 Split
    train_idx, test_idx = train_test_split(
        np.arange(len(targets)), 
        test_size=0.20, 
        random_state=42, 
        stratify=targets
    )
    
    # Create subsets with appropriate transforms
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
            
        def __len__(self):
            return len(self.subset)

    train_subset = Subset(full_dataset, train_idx)
    test_subset = Subset(full_dataset, test_idx)
    
    train_data = TransformedSubset(train_subset, train_transform)
    test_data = TransformedSubset(test_subset, test_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    log(f"Training samples: {len(train_data)}")
    log(f"Testing samples: {len(test_data)}")

    # 3. Model Implementation (ResNet18 with Dropout to prevent overfitting)
    log("Initializing ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify final layer and add Dropout
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5), # High dropout to prevent memorizing
        nn.Linear(num_ftrs, 2)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Lower learning rate for fine-tuning

    # 4. Training Loop
    log("Starting training...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            log(f"  Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        epoch_acc = 100. * correct / total
        log(f"--- Epoch {epoch+1} Summary: Loss {running_loss/len(train_loader):.4f}, Acc {epoch_acc:.2f}% ---")

        # Validation at end of each epoch
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        log(f"--- Validation Accuracy: {test_acc:.2f}% ---")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_gender_model.pth')
            log("  Model saved (Best so far)")

    log(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 'final_gender_model.pth')
    log("Final model saved as final_gender_model.pth")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
