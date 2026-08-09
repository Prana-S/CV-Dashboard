#!/usr/bin/env python3
"""Prepare a balanced, clean marine-life vs. plastic-debris image dataset.

The generated images are RGB JPEGs at the requested size.  Pixel normalization
is deliberately saved as part of preprocessing_config.json instead of baking
0-1 floating point values into JPEG files; the Week 2 TensorFlow pipeline can
apply ``Rescaling(1./255)`` without losing image fidelity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ("marine_life", "plastic")


@dataclass(frozen=True)
class SourceImage:
    path: Path
    digest: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, balance, resize, split, and augment a binary image dataset."
    )
    parser.add_argument("--marine-source", type=Path, required=True,
                        help="Folder containing marine-life images.")
    parser.add_argument("--plastic-source", type=Path, required=True,
                        help="Folder containing plastic-debris images.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"),
                        help="Destination directory (default: data/processed).")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Square output resolution (default: 224).")
    parser.add_argument("--validation-split", type=float, default=0.2,
                        help="Fraction reserved for validation (default: 0.2).")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Optional cap on original images per class.")
    parser.add_argument("--augmentations-per-image", type=int, default=2,
                        help="Training augmentations made for each original (default: 2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing output directory.")
    return parser.parse_args()


def image_digest(path: Path) -> str:
    # Hash the original file, not the resized output, so duplicates are easy to trace.
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_unique_images(source_dir: Path) -> tuple[list[SourceImage], list[dict[str, str]]]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    valid: list[SourceImage] = []
    rejected: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        try:
            # verify() catches truncated/corrupt files without loading every pixel into memory.
            with Image.open(path) as image:
                image.verify()
            digest = image_digest(path)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            rejected.append({"file": str(path), "reason": f"unreadable: {error}"})
            continue
        if digest in seen_hashes:
            rejected.append({"file": str(path), "reason": "duplicate within class"})
            continue
        seen_hashes.add(digest)
        valid.append(SourceImage(path=path, digest=digest))
    return valid, rejected


def resize_to_square(image: Image.Image, image_size: int) -> Image.Image:
    """Convert to RGB and center-crop after fitting, preserving aspect ratio."""
    # Phone photos often store their orientation in EXIF instead of rotating pixels.
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.LANCZOS)


def save_jpeg(image: Image.Image, destination: Path) -> None:
    # The split/class folders may not exist until the first image is written.
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=95, optimize=True)


def augmented_variants(image: Image.Image, count: int, rng: random.Random) -> Iterable[tuple[str, Image.Image]]:
    """Create reproducible rotation/flip variants for *training data only*."""
    for index in range(count):
        # Keep changes modest so the image still represents a realistic underwater view.
        angle = rng.choice((-20, -15, -10, 10, 15, 20))
        variant = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
        operations = [f"rot{angle:+d}"]
        if index % 2 == 0:
            variant = ImageOps.mirror(variant)
            operations.append("flip-h")
        yield "_".join(operations), variant


def prepare_class(
    class_name: str,
    sources: list[SourceImage],
    output_dir: Path,
    image_size: int,
    validation_split: float,
    augmentations_per_image: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    shuffled = sources[:]
    rng.shuffle(shuffled)

    # Split originals first. Augmenting before this point would leak near-identical
    # images into validation and make the model look better than it really is.
    validation_count = max(1, round(len(shuffled) * validation_split)) if len(shuffled) > 1 else 0
    validation_sources = shuffled[:validation_count]
    training_sources = shuffled[validation_count:]
    records: list[dict[str, object]] = []

    for split, items in (("validation", validation_sources), ("train", training_sources)):
        for source_index, source in enumerate(items, start=1):
            with Image.open(source.path) as raw_image:
                prepared = resize_to_square(raw_image, image_size)
            basename = f"{class_name}_{source_index:05d}"
            destination = output_dir / split / class_name / f"{basename}.jpg"
            save_jpeg(prepared, destination)
            records.append({
                "output_file": str(destination.relative_to(output_dir)),
                "class_name": class_name,
                "class_index": CLASS_NAMES.index(class_name),
                "split": split,
                "source_file": str(source.path),
                "source_sha256": source.digest,
                "augmentation": "original",
            })
            if split == "train":
                # Validation needs to stay untouched so it remains an honest check.
                for aug_index, (operations, variant) in enumerate(
                    augmented_variants(prepared, augmentations_per_image, rng), start=1
                ):
                    augmented_destination = output_dir / split / class_name / f"{basename}_aug{aug_index}.jpg"
                    save_jpeg(variant, augmented_destination)
                    records.append({
                        "output_file": str(augmented_destination.relative_to(output_dir)),
                        "class_name": class_name,
                        "class_index": CLASS_NAMES.index(class_name),
                        "split": split,
                        "source_file": str(source.path),
                        "source_sha256": source.digest,
                        "augmentation": operations,
                    })
    return records


def main() -> None:
    args = parse_args()
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if not 0 < args.validation_split < 1:
        raise ValueError("--validation-split must be between 0 and 1")
    if args.augmentations_per_image < 0:
        raise ValueError("--augmentations-per-image cannot be negative")

    class_sources: dict[str, list[SourceImage]] = {}
    rejected: list[dict[str, str]] = []
    for class_name, directory in (("marine_life", args.marine_source), ("plastic", args.plastic_source)):
        images, class_rejected = load_unique_images(directory)
        class_sources[class_name] = images
        rejected.extend([{"class_name": class_name, **entry} for entry in class_rejected])

    available = {name: len(items) for name, items in class_sources.items()}
    # Matching the smaller class prevents the model from learning to favor the majority label.
    balanced_count = min(available.values()) if available else 0
    if args.max_per_class is not None:
        if args.max_per_class <= 1:
            raise ValueError("--max-per-class must be at least 2")
        balanced_count = min(balanced_count, args.max_per_class)
    if balanced_count < 2:
        raise ValueError(
            "Need at least two valid, unique images in each class to create train and validation sets. "
            f"Found: {available}"
        )

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        # Requiring an explicit flag protects a finished dataset from accidental replacement.
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Use --overwrite.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # A fixed seed makes the same source folders produce the same split every time.
    rng = random.Random(args.seed)
    records: list[dict[str, object]] = []
    for class_name in CLASS_NAMES:
        selected = class_sources[class_name][:balanced_count]
        records.extend(prepare_class(
            class_name, selected, args.output_dir, args.image_size, args.validation_split,
            args.augmentations_per_image, rng,
        ))

    counts = Counter((str(item["split"]), str(item["class_name"])) for item in records)
    # Keep the preprocessing choices next to the dataset so Week 2 uses the same contract.
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "classes": {"marine_life": 0, "plastic": 1},
        "image_size": [args.image_size, args.image_size],
        "normalization": {"formula": "pixel_value / 255.0", "range": [0.0, 1.0]},
        "validation_split": args.validation_split,
        "augmentation": {
            "applies_to": "train only",
            "operations": ["rotation: +/-10, +/-15, or +/-20 degrees", "horizontal flip"],
            "variants_per_original": args.augmentations_per_image,
        },
        "source_images_selected_per_class": balanced_count,
        "output_counts": {f"{split}/{class_name}": count for (split, class_name), count in sorted(counts.items())},
        "rejected_files": rejected,
        "images": records,
    }
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "preprocessing_config.json").write_text(json.dumps({
        "image_size": [args.image_size, args.image_size],
        "color_mode": "rgb",
        "normalization": "pixel_value / 255.0",
        "class_indices": {"marine_life": 0, "plastic": 1},
    }, indent=2) + "\n")

    print("Week 1 dataset prepared successfully.")
    print(f"Balanced originals selected per class: {balanced_count}")
    for (split, class_name), count in sorted(counts.items()):
        print(f"  {split}/{class_name}: {count} images")
    print(f"Manifest: {args.output_dir / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
