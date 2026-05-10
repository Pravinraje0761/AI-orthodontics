import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_group_id(file_path: Path) -> str:
    name = file_path.stem
    return name.split("_aug_")[0] if "_aug_" in name else name


def build_grouped_split(
    data_root: Path, train_ratio: float = 0.7, seed: int = 42
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[str]]:
    class_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()], key=lambda p: p.name)
    class_names = [d.name for d in class_dirs]
    class_to_label = {name: i for i, name in enumerate(class_names)}

    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    rng = random.Random(seed)

    for class_dir in class_dirs:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for image_path in class_dir.glob(ext):
                grouped[extract_group_id(image_path)].append(image_path)

        group_ids = list(grouped.keys())
        rng.shuffle(group_ids)
        split_idx = max(1, int(len(group_ids) * train_ratio))
        train_groups = set(group_ids[:split_idx])

        label = class_to_label[class_dir.name]
        for gid, paths in grouped.items():
            target = train_samples if gid in train_groups else val_samples
            target.extend((p, label) for p in paths)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples, class_names


def load_samples(
    samples: list[tuple[Path, int]], image_size: tuple[int, int] = (64, 64)
) -> tuple[np.ndarray, np.ndarray]:
    x_data = []
    y_data = []
    for image_path, label in samples:
        image = Image.open(image_path).convert("L").resize(image_size)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        x_data.append(arr)
        y_data.append(label)

    x = np.array(x_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)
    return x.reshape(len(x), -1), y


def main() -> None:
    data_root = Path("opg with landmarks") / "augmented_classwise"
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_samples, val_samples, class_names = build_grouped_split(
        data_root=data_root, train_ratio=0.7, seed=42
    )
    x_train, y_train = load_samples(train_samples)
    x_val, y_val = load_samples(val_samples)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "SVM_RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=10, gamma="scale")),
            ]
        ),
    }

    best_name = ""
    best_scores = None
    best_val_pred = None
    model_scores: dict[str, float] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        scores = {
            "train_acc": accuracy_score(y_train, train_pred),
            "eval_acc": accuracy_score(y_val, val_pred),
            "eval_precision": precision_score(y_val, val_pred),
            "eval_recall": recall_score(y_val, val_pred),
            "eval_f1": f1_score(y_val, val_pred),
        }
        model_scores[name] = scores["eval_acc"]
        print(f"{name} validation accuracy: {scores['eval_acc']:.4f}")
        if best_scores is None or scores["eval_acc"] > best_scores["eval_acc"]:
            best_name = name
            best_scores = scores
            best_val_pred = val_pred

    print(f"Classes: {class_names}")
    print(f"Train samples (group-safe): {len(train_samples)}")
    print(f"Validation samples (group-safe): {len(val_samples)}")
    print(f"Selected model: {best_name}")
    print(f"Final train accuracy: {best_scores['train_acc']:.4f}")
    print(f"Final val accuracy: {best_scores['eval_acc']:.4f}")
    print(f"Evaluation accuracy: {best_scores['eval_acc']:.4f}")
    print(f"Evaluation precision: {best_scores['eval_precision']:.4f}")
    print(f"Evaluation recall: {best_scores['eval_recall']:.4f}")
    print(f"Evaluation F1: {best_scores['eval_f1']:.4f}")

    # Save metrics to a text report.
    report_path = output_dir / "grouped_split_metrics.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Leakage-safe supervised output (70/30 grouped split)\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Train samples: {len(train_samples)}\n")
        f.write(f"Validation samples: {len(val_samples)}\n")
        f.write(f"Selected model: {best_name}\n")
        f.write(f"Train accuracy: {best_scores['train_acc']:.4f}\n")
        f.write(f"Validation accuracy: {best_scores['eval_acc']:.4f}\n")
        f.write(f"Precision: {best_scores['eval_precision']:.4f}\n")
        f.write(f"Recall: {best_scores['eval_recall']:.4f}\n")
        f.write(f"F1-score: {best_scores['eval_f1']:.4f}\n")

    # Figure 1: Model validation accuracy comparison.
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(model_scores.keys())
    values = [model_scores[n] for n in names]
    bars = ax.bar(names, values, color=["#4e79a7", "#f28e2b"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Model Comparison (Grouped Split)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison_accuracy.png", dpi=200)
    plt.close(fig)

    # Figure 2: Confusion matrix of selected model.
    cm = confusion_matrix(y_val, best_val_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix ({best_name})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_grouped_split.png", dpi=200)
    plt.close(fig)

    print(f"Saved report: {report_path}")
    print(f"Saved figure: {output_dir / 'model_comparison_accuracy.png'}")
    print(f"Saved figure: {output_dir / 'confusion_matrix_grouped_split.png'}")


if __name__ == "__main__":
    main()
