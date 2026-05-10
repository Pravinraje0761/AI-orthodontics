from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


def make_occlusion_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    patch_size: int = 32,
    stride: int = 32,
) -> tuple[np.ndarray, float]:
    baseline_score = float(model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0][0])
    h, w, _ = img_array.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            occluded = img_array.copy()
            occluded[y : y + patch_size, x : x + patch_size, :] = 0
            score = float(model.predict(np.expand_dims(occluded, axis=0), verbose=0)[0][0])
            drop = max(0.0, baseline_score - score)
            heatmap[y : y + patch_size, x : x + patch_size] += drop
            counts[y : y + patch_size, x : x + patch_size] += 1.0

    counts[counts == 0] = 1.0
    heatmap = heatmap / counts
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    return heatmap, baseline_score


def overlay_heatmap_on_image(
    image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    heatmap_resized = Image.fromarray(np.uint8(heatmap * 255)).resize(
        (image_rgb.shape[1], image_rgb.shape[0]), resample=Image.Resampling.BILINEAR
    )
    heatmap_resized = np.array(heatmap_resized) / 255.0
    colormap = plt.get_cmap("jet")
    heatmap_color = colormap(heatmap_resized)[:, :, :3]

    base = image_rgb.astype(np.float32) / 255.0
    overlay = (1 - alpha) * base + alpha * heatmap_color
    return np.uint8(np.clip(overlay, 0, 1) * 255)


def process_folder(model_path: Path, input_root: Path, output_root: Path) -> None:
    model = tf.keras.models.load_model(model_path, compile=False)
    class_dirs = [
        d for d in input_root.iterdir() if d.is_dir() and d.name.lower() in {"male", "female"}
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    processed = 0

    for class_dir in sorted(class_dirs, key=lambda p: p.name):
        target_dir = output_root / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        image_paths = sorted(
            [*class_dir.glob("*.jpg"), *class_dir.glob("*.jpeg"), *class_dir.glob("*.png")]
        )

        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB").resize((128, 128))
            image_arr = np.array(image, dtype=np.float32)
            heatmap, score = make_occlusion_heatmap(image_arr, model, patch_size=32, stride=32)
            overlay = overlay_heatmap_on_image(image_arr.astype(np.uint8), heatmap, alpha=0.45)
            out_name = f"{image_path.stem}_explain_{score:.3f}.jpg"
            out_path = target_dir / out_name
            if out_path.exists():
                continue
            Image.fromarray(overlay).save(out_path, format="JPEG", quality=95)

            processed += 1
            if processed % 20 == 0:
                print(f"Generated {processed} explanation images...", flush=True)

    print(f"Done. Generated {processed} explanation outputs in: {output_root}")


def main() -> None:
    model_path = Path("models") / "opg_supervised_cnn_70_30.keras"
    input_root = Path("opg with landmarks")
    output_root = Path("results") / "gradcam_outputs_original"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    process_folder(model_path=model_path, input_root=input_root, output_root=output_root)


if __name__ == "__main__":
    main()
