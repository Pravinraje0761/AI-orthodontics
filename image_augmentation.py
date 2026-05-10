import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def collect_images(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def random_transform(image: Image.Image) -> Image.Image:
    img = image.convert("RGB")

    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.2:
        img = ImageOps.flip(img)

    angle = random.uniform(-12, 12)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)

    w, h = img.size
    scale = random.uniform(0.88, 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    left = random.randint(0, w - nw) if w > nw else 0
    top = random.randint(0, h - nh) if h > nh else 0
    img = img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.Resampling.BICUBIC)

    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(random.uniform(0.8, 1.2))

    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(random.uniform(0.8, 1.25))

    sharpness = ImageEnhance.Sharpness(img)
    img = sharpness.enhance(random.uniform(0.7, 1.5))

    if random.random() < 0.35:
        sigma = random.uniform(0.2, 1.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    return img


def augment_to_target(input_dir: Path, output_dir: Path, target_count: int, seed: int) -> int:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_images = collect_images(input_dir)
    if not source_images:
        raise ValueError(f"No images found in: {input_dir}")

    generated = 0
    source_idx = 0

    while generated < target_count:
        src = source_images[source_idx % len(source_images)]
        source_idx += 1

        with Image.open(src) as im:
            aug = random_transform(im)

        out_name = f"{src.stem}_aug_{generated + 1:04d}.jpg"
        out_path = output_dir / out_name
        aug.save(out_path, format="JPEG", quality=95)
        generated += 1

        if generated % 100 == 0 or generated == target_count:
            print(f"Generated {generated}/{target_count}")

    return generated


def augment_class_subfolders(
    input_root: Path, output_root: Path, target_per_class: int, seed: int
) -> dict[str, int]:
    class_dirs = [d for d in input_root.iterdir() if d.is_dir() and d.name.lower() in {"male", "female"}]
    if not class_dirs:
        raise ValueError(f"No Male/Female subfolders found in: {input_root}")

    results: dict[str, int] = {}
    for idx, class_dir in enumerate(sorted(class_dirs, key=lambda p: p.name.lower())):
        class_output = output_root / class_dir.name.upper()
        generated = augment_to_target(
            input_dir=class_dir,
            output_dir=class_output,
            target_count=target_per_class,
            seed=seed + idx,
        )
        results[class_dir.name.upper()] = generated
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate augmented OPG images.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("opg with landmarks"),
        help="Path to source image folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("opg with landmarks") / "augmented_800",
        help="Where augmented images will be saved.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=800,
        help="Total number of augmented images to generate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--class-wise",
        action="store_true",
        help="Augment Male/Female class subfolders separately.",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=400,
        help="Number of augmented images to generate in each class subfolder.",
    )
    args = parser.parse_args()

    if args.class_wise:
        class_results = augment_class_subfolders(
            input_root=args.input_dir,
            output_root=args.output_dir,
            target_per_class=args.target_per_class,
            seed=args.seed,
        )
        total = sum(class_results.values())
        print(f"Done. Generated class-wise data in: {args.output_dir}")
        print(f"Breakdown: {class_results}")
        print(f"Total generated: {total}")
    else:
        total = augment_to_target(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_count=args.target_count,
            seed=args.seed,
        )
        print(f"Done. Generated {total} augmented images in: {args.output_dir}")


if __name__ == "__main__":
    main()
