#!/usr/bin/env python3
"""Create balanced marine-life and plastic image crops from the SeaClear annotations.

SeaClear is an object-detection dataset. This script turns its COCO annotations
into the two folders expected by the Week 1 classifier preparation script.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

# These are the SeaClear labels that directly answer the project's binary question.
PLASTIC_CATEGORIES = {
    "tarp_plastic", "container_plastic", "bottle_plastic", "pipe_plastic",
    "net_plastic", "cup_plastic", "bag_plastic", "sanitaries_plastic",
    "snack_wrapper_plastic", "lid_plastic", "rope_plastic",
}
MARINE_LIFE_CATEGORIES = {
    "animal_etc", "animal_sponge", "animal_shells", "animal_urchin",
    "animal_fish", "animal_starfish",
}


@dataclass(frozen=True)
class CropCandidate:
    """One labeled object crop that can become a classifier training image."""

    image_id: int
    image_path: Path
    source_name: str
    category: str
    bbox: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build balanced marine-life and plastic crops from the SeaClear dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/raw/seaclear_original/extracted/Seaclear_Marine_Debris_Dataset"),
        help="Extracted SeaClear directory containing dataset.json (default: local dataset folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Folder where marine_life and plastic source crops are saved.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=500,
        help="Number of original crops to save for each class (default: 500).",
    )
    parser.add_argument(
        "--min-box-side",
        type=int,
        default=64,
        help="Skip tiny annotations smaller than this many pixels on either side (default: 64).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Extra image around each object as a fraction of its box size (default: 0.15).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated crop images in the output class folders.",
    )
    return parser.parse_args()


def read_annotations(dataset_dir: Path) -> dict[str, object]:
    annotation_file = dataset_dir / "dataset.json"
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Could not find the SeaClear annotation file: {annotation_file}")
    return json.loads(annotation_file.read_text())


def index_image_files(dataset_dir: Path) -> dict[str, Path]:
    """SeaClear stores unique file names in nested site/camera folders."""
    files = {path.name: path for path in dataset_dir.rglob("*.jpg")}
    if not files:
        raise FileNotFoundError(f"No JPG images were found in: {dataset_dir}")
    return files


def class_for_category(category_name: str) -> str | None:
    if category_name in PLASTIC_CATEGORIES:
        return "plastic"
    if category_name in MARINE_LIFE_CATEGORIES:
        return "marine_life"
    return None


def build_candidates(
    data: dict[str, object], image_files: dict[str, Path], min_box_side: int,
) -> dict[str, list[CropCandidate]]:
    categories = {item["id"]: item["name"] for item in data["categories"]}  # type: ignore[index]
    images = {item["id"]: item for item in data["images"]}  # type: ignore[index]

    # If the same original photo has both target labels, keep it out of both classes.
    # This prevents the same camera frame from appearing on opposite sides of the task.
    target_labels_by_image: dict[int, set[str]] = defaultdict(set)
    for annotation in data["annotations"]:  # type: ignore[index]
        label = class_for_category(categories[annotation["category_id"]])
        if label:
            target_labels_by_image[annotation["image_id"]].add(label)

    best_crop_by_image: dict[str, dict[int, CropCandidate]] = {
        "marine_life": {},
        "plastic": {},
    }
    for annotation in data["annotations"]:  # type: ignore[index]
        category = categories[annotation["category_id"]]
        label = class_for_category(category)
        image_id = annotation["image_id"]
        if not label or len(target_labels_by_image[image_id]) != 1:
            continue

        x, y, width, height = annotation["bbox"]
        if min(width, height) < min_box_side:
            continue
        image_info = images[image_id]
        image_path = image_files.get(image_info["file_name"])
        if image_path is None:
            continue

        candidate = CropCandidate(
            image_id=image_id,
            image_path=image_path,
            source_name=image_info["file_name"],
            category=category,
            bbox=(x, y, width, height),
        )
        current = best_crop_by_image[label].get(image_id)
        # One crop per source photo keeps sequential video frames from dominating the data.
        if current is None or width * height > current.bbox[2] * current.bbox[3]:
            best_crop_by_image[label][image_id] = candidate

    return {label: list(per_image.values()) for label, per_image in best_crop_by_image.items()}


def padded_crop(image: Image.Image, bbox: tuple[float, float, float, float], padding: float) -> Image.Image:
    x, y, width, height = bbox
    extra_x = width * padding
    extra_y = height * padding
    left = max(0, round(x - extra_x))
    top = max(0, round(y - extra_y))
    right = min(image.width, round(x + width + extra_x))
    bottom = min(image.height, round(y + height + extra_y))
    return image.crop((left, top, right, bottom)).convert("RGB")


def clear_generated_images(folder: Path) -> None:
    if not folder.exists():
        return
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() == ".jpg":
            path.unlink()


def save_crops(
    label: str,
    candidates: list[CropCandidate],
    output_dir: Path,
    per_class: int,
    padding: float,
    rng: random.Random,
) -> list[dict[str, object]]:
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    chosen = shuffled[:per_class]
    class_dir = output_dir / label
    class_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for number, candidate in enumerate(chosen, start=1):
        with Image.open(candidate.image_path) as image:
            crop = padded_crop(image, candidate.bbox, padding)
        output_file = class_dir / f"{label}_{number:04d}_{candidate.category}.jpg"
        crop.save(output_file, format="JPEG", quality=95, optimize=True)
        records.append({
            "output_file": str(output_file),
            "source_file": str(candidate.image_path),
            "source_image": candidate.source_name,
            "source_image_id": candidate.image_id,
            "category": candidate.category,
            "bbox": candidate.bbox,
        })
    return records


def main() -> None:
    args = parse_args()
    if args.per_class < 1:
        raise ValueError("--per-class must be at least 1")
    if args.min_box_side < 1:
        raise ValueError("--min-box-side must be at least 1")
    if not 0 <= args.padding <= 1:
        raise ValueError("--padding must be between 0 and 1")

    data = read_annotations(args.dataset_dir)
    candidates = build_candidates(data, index_image_files(args.dataset_dir), args.min_box_side)
    available = {label: len(items) for label, items in candidates.items()}
    if min(available.values()) < args.per_class:
        raise ValueError(
            f"Not enough clean candidates for {args.per_class} images per class. Found: {available}"
        )

    if not args.overwrite:
        existing = [
            path for label in ("marine_life", "plastic")
            for path in (args.output_dir / label).glob("*.jpg")
        ]
        if existing:
            raise FileExistsError(
                f"Generated crops already exist in {args.output_dir}. Use --overwrite to replace them."
            )

    for label in ("marine_life", "plastic"):
        clear_generated_images(args.output_dir / label)

    rng = random.Random(args.seed)
    records = {
        label: save_crops(label, candidates[label], args.output_dir, args.per_class, args.padding, rng)
        for label in ("marine_life", "plastic")
    }
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dataset": "SeaClear Marine Debris Detection & Segmentation Dataset",
        "source_license": "CC BY 4.0",
        "selection_rules": {
            "plastic_categories": sorted(PLASTIC_CATEGORIES),
            "marine_life_categories": sorted(MARINE_LIFE_CATEGORIES),
            "minimum_box_side_pixels": args.min_box_side,
            "padding_fraction": args.padding,
            "mixed_label_source_images_excluded": True,
            "one_crop_per_source_image": True,
        },
        "available_clean_candidates": available,
        "saved_per_class": args.per_class,
        "records": records,
    }
    (args.output_dir / "seaclear_crop_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("SeaClear crops created successfully.")
    print(f"Available clean candidates: {available}")
    print(f"Saved {args.per_class} crops to each class folder in: {args.output_dir}")


if __name__ == "__main__":
    main()
