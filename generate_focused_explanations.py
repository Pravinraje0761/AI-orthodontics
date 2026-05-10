from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw


def load_image(path: Path, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size)
    return np.asarray(img, dtype=np.float32)


def smoothgrad_saliency(
    model: tf.keras.Model,
    image_arr: np.ndarray,
    samples: int = 24,
    noise_std: float = 8.0,
) -> tuple[np.ndarray, float]:
    base = image_arr.copy()
    grads_accum = np.zeros_like(base, dtype=np.float32)

    pred_score = float(model(np.expand_dims(base, 0), training=False).numpy()[0][0])
    target_positive = pred_score >= 0.5

    for _ in range(samples):
        noise = np.random.normal(0.0, noise_std, size=base.shape).astype(np.float32)
        noisy = np.clip(base + noise, 0, 255)
        x = tf.convert_to_tensor(np.expand_dims(noisy, 0))

        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = model(x, training=False)[:, 0]
            target = pred if target_positive else (1.0 - pred)
        grads = tape.gradient(target, x).numpy()[0]
        grads_accum += np.abs(grads)

    saliency = np.mean(grads_accum / samples, axis=-1)
    saliency -= saliency.min()
    saliency /= (saliency.max() + 1e-8)
    return saliency, pred_score


def white_guided_heatmap(image_arr: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    gray = np.mean(image_arr / 255.0, axis=-1)
    p75 = float(np.percentile(gray, 75))
    white_strength = np.clip((gray - p75) / (1.0 - p75 + 1e-8), 0.0, 1.0)

    combined = saliency * white_strength
    if np.max(combined) < 1e-6:
        combined = saliency

    # Smooth heatmap so highlighted zones are stable and readable.
    hm = tf.convert_to_tensor(combined[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
    hm = tf.nn.avg_pool2d(hm, ksize=5, strides=1, padding="SAME")
    combined = hm.numpy()[0, :, :, 0]

    combined -= combined.min()
    combined /= (combined.max() + 1e-8)
    return combined


def overlay_heatmap(image_arr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[:, :, :3]
    base = image_arr / 255.0
    mixed = (1 - alpha) * base + alpha * colored
    return np.uint8(np.clip(mixed, 0, 1) * 255)


def draw_top_region(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    h, w = heatmap.shape
    flat_idx = np.argsort(heatmap.ravel())[::-1]
    top_k = flat_idx[: max(64, int(0.02 * h * w))]
    ys, xs = np.unravel_index(top_k, (h, w))
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    out = Image.fromarray(image)
    draw = ImageDraw.Draw(out)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2)
    return np.array(out)


def pick_few_images(root: Path, per_class: int = 3) -> list[tuple[Path, str]]:
    chosen: list[tuple[Path, str]] = []
    for class_name in ["FEMALE", "MALE"]:
        class_dir = root / class_name
        paths = sorted(
            [*class_dir.glob("*.jpg"), *class_dir.glob("*.jpeg"), *class_dir.glob("*.png")]
        )[:per_class]
        chosen.extend((p, class_name) for p in paths)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate white-focused explanation overlays.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("opg with landmarks") / "augmented_classwise",
        help="Root folder with FEMALE/MALE subfolders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results") / "focused_explanations_whitecam",
        help="Output folder for explanation images.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=20,
        help="Number of images to process per class.",
    )
    args = parser.parse_args()

    model_path = Path("models") / "opg_supervised_cnn_70_30.keras"
    input_root = args.input_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path, compile=False)
    selected = pick_few_images(input_root, per_class=args.per_class)
    if not selected:
        raise ValueError("No images found in MALE/FEMALE folders.")

    for image_path, class_name in selected:
        img = load_image(image_path, size=(128, 128))
        saliency, score = smoothgrad_saliency(model, img, samples=24, noise_std=8.0)
        heatmap = white_guided_heatmap(img, saliency)
        overlay = overlay_heatmap(img, heatmap, alpha=0.5)
        boxed = draw_top_region(overlay, heatmap)

        class_out = output_root / class_name
        class_out.mkdir(parents=True, exist_ok=True)
        out_name = f"{image_path.stem}_whitecam_{score:.3f}.jpg"
        Image.fromarray(boxed).save(class_out / out_name, format="JPEG", quality=95)
        print(f"Saved: {class_name}/{out_name}")

    print(f"Done. Focused explanation outputs saved in: {output_root}")


if __name__ == "__main__":
    main()
