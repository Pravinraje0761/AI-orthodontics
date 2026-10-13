"""Train the ResNet18 sex classifier on panoramic dental X-rays (OPGs).

Evaluation is leak-free: train, validation and test are split by *source X-ray*
(see splits.py), the checkpoint is chosen on validation, and the test set is scored
once at the end. The headline test metrics use original X-rays only; augmented copies
of a test X-ray are not independent samples and are reported separately.

    python train_gender.py --data-dir "data/OPG With landmarks" --epochs 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from splits import grouped_split, original_id

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class Transformed(Dataset):
    def __init__(self, subset: Subset, transform):
        self.subset, self.transform = subset, transform

    def __getitem__(self, i):
        x, y = self.subset[i]
        return self.transform(x), y

    def __len__(self):
        return len(self.subset)


def build_model(pretrained: bool = True) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 2))
    return model


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        out.append(model(x.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data/OPG With landmarks",
                    help="ImageFolder root with one sub-folder per class (female/, male/)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet",
                    help="'none' skips the ImageNet download (smoke tests only)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    full = datasets.ImageFolder(args.data_dir)
    paths = [p for p, _ in full.samples]
    labels = np.array(full.targets)
    train_idx, val_idx, test_idx, groups = grouped_split(paths, labels, seed=args.seed)
    for name, ix in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(f"{name:5s}: {len(ix):4d} images from {len(set(groups[ix])):3d} source X-rays")

    loaders = {
        "train": DataLoader(Transformed(Subset(full, train_idx), train_transform),
                            batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(Transformed(Subset(full, val_idx), eval_transform), batch_size=args.batch_size),
        "test": DataLoader(Transformed(Subset(full, test_idx), eval_transform), batch_size=args.batch_size),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=args.weights == "imagenet").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val, ckpt = -1.0, out / "best_gender_model.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_acc = accuracy_score(labels[val_idx], predict(model, loaders["val"], device))
        print(f"epoch {epoch}: train loss {total_loss / len(loaders['train']):.4f}, val acc {val_acc:.4f}")
        if val_acc > best_val:  # checkpoint chosen on validation, never on test
            best_val = val_acc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    y_pred = predict(model, loaders["test"], device)
    y_true = labels[test_idx]
    is_original = np.array([original_id(paths[i]) == Path(paths[i]).stem for i in test_idx])

    metrics = {
        "split": "grouped by source X-ray; checkpoint selected on validation",
        "n_test_source_xrays": int(len(set(groups[test_idx]))),
        "test_originals_only": {
            "n": int(is_original.sum()),
            "accuracy": float(accuracy_score(y_true[is_original], y_pred[is_original])),
            "report": classification_report(y_true[is_original], y_pred[is_original],
                                            target_names=full.classes, output_dict=True, zero_division=0),
        },
        "test_all_images_incl_augmented_copies": {
            "n": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        },
        "best_val_accuracy": float(best_val),
        "args": vars(args),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "args"}, indent=2, default=str))


if __name__ == "__main__":
    main()
