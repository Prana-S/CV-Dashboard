#!/usr/bin/env python3
"""Create an untouched SeaClear test set that has no source-image overlap with Week 1."""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

from build_seaclear_binary_dataset import (
    build_candidates,
    clear_generated_images,
    index_image_files,
    read_annotations,
    save_crops,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an unseen test set from the SeaClear dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/raw/seaclear_original/extracted/Seaclear_Marine_Debris_Dataset"),
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("data/raw/seaclear_crop_manifest.json"),
        help="Manifest that records the source images already used for Week 1.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--min-box-side", type=int, default=64)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_image_ids(manifest_path: Path) -> set[int]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Week 1 crop manifest was not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    return {
        int(record["source_image_id"])
        for label in ("marine_life", "plastic")
        for record in manifest["records"][label]
    }


def main() -> None:
    args = parse_args()
    if args.per_class < 1:
        raise ValueError("--per-class must be at least 1")
    if args.min_box_side < 1:
        raise ValueError("--min-box-side must be at least 1")
    if not 0 <= args.padding <= 1:
        raise ValueError("--padding must be between 0 and 1")

    used_image_ids = source_image_ids(args.training_manifest)
    annotations = read_annotations(args.dataset_dir)
    candidates = build_candidates(
        annotations,
        index_image_files(args.dataset_dir),
        args.min_box_side,
    )
    unseen_candidates = {
        label: [candidate for candidate in items if candidate.image_id not in used_image_ids]
        for label, items in candidates.items()
    }
    available = {label: len(items) for label, items in unseen_candidates.items()}
    if min(available.values()) < args.per_class:
        raise ValueError(f"Not enough unseen candidates for {args.per_class} images per class. Found: {available}")

    existing_files = [
        path
        for label in ("marine_life", "plastic")
        for path in (args.output_dir / label).glob("*.jpg")
    ]
    if existing_files and not args.overwrite:
        raise FileExistsError(f"Test images already exist in {args.output_dir}. Use --overwrite to replace them.")

    for label in ("marine_life", "plastic"):
        clear_generated_images(args.output_dir / label)

    rng = random.Random(args.seed)
    records = {
        label: save_crops(label, unseen_candidates[label], args.output_dir, args.per_class, args.padding, rng)
        for label in ("marine_life", "plastic")
    }
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dataset": "SeaClear Marine Debris Detection & Segmentation Dataset",
        "source_images_excluded_from_week_1": len(used_image_ids),
        "unseen_candidates_available": available,
        "saved_per_class": args.per_class,
        "records": records,
    }
    (args.output_dir / "test_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Unseen SeaClear test set created successfully.")
    print(f"Source image IDs excluded from Week 1: {len(used_image_ids)}")
    print(f"Saved {args.per_class} test images per class in: {args.output_dir}")


if __name__ == "__main__":
    main()
