#!/usr/bin/env python3
"""Run the trained marine debris model on new images with an OpenCV overlay.

This is a local Week 4 inference runner, not a true object detector. The model
classifies the image region we give it. In this script, that region is the whole
image, so the rectangle marks the area that was analysed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ("Marine Life", "Plastic Debris")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/plastic_pulse_week3_tuned.keras"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/week4/live_test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DASHBOARD_WIDTH = 720
MAX_IMAGE_HEIGHT = 560
PANEL_HEIGHT = 112
PADDING = 24


class PredictionResult:
    """Small container for the values we write to the image and summary file."""

    def __init__(self, label: str, confidence: float, plastic_probability: float) -> None:
        self.label = label
        self.confidence = confidence
        self.plastic_probability = plastic_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate one image or a small folder of images with the trained marine debris model."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Path to one image to classify.")
    source.add_argument("--input-dir", type=Path, help="Folder of images to classify recursively.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.5, help="Plastic probability needed to label Plastic Debris.")
    parser.add_argument("--limit", type=int, default=12, help="Maximum number of images to process from --input-dir.")
    parser.add_argument("--display", action="store_true", help="Show each annotated image in an OpenCV window.")
    parser.add_argument("--overwrite", action="store_true", help="Replace files in an existing output folder.")
    return parser.parse_args()


def _image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def _round_robin(groups: list[list[Path]], limit: int | None) -> list[Path]:
    chosen: list[Path] = []
    index = 0
    while True:
        added_this_round = False
        for group in groups:
            if index < len(group):
                chosen.append(group[index])
                added_this_round = True
                if limit is not None and len(chosen) >= limit:
                    return chosen
        if not added_this_round:
            return chosen
        index += 1


def find_images(input_dir: Path, limit: int | None) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder was not found: {input_dir}")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")

    class_folders = [input_dir / "marine_life", input_dir / "plastic"]
    if all(folder.is_dir() for folder in class_folders):
        # For the Week 3 test folder, pull from both classes so the batch is useful evidence.
        images = _round_robin([_image_files(folder) for folder in class_folders], limit)
    else:
        images = _image_files(input_dir)
        if limit is not None:
            images = images[:limit]

    if not images:
        raise FileNotFoundError(f"No supported image files were found in: {input_dir}")
    return images


def expected_label_from_path(image_path: Path) -> str | None:
    parts = set(image_path.parts)
    if "marine_life" in parts:
        return "Marine Life"
    if "plastic" in parts:
        return "Plastic Debris"
    return None


def load_bgr_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read this image: {image_path}")
    return image


def prepare_for_model(bgr_image: np.ndarray) -> np.ndarray:
    """Match the Week 1 preprocessing: RGB, 224 x 224, and pixel values from 0 to 1."""
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def predict_image(model: tf.keras.Model, bgr_image: np.ndarray, threshold: float) -> PredictionResult:
    if not 0.0 < threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    model_input = prepare_for_model(bgr_image)
    plastic_probability = float(model.predict(model_input, verbose=0)[0][0])
    if plastic_probability >= threshold:
        return PredictionResult(CLASS_NAMES[1], plastic_probability, plastic_probability)
    return PredictionResult(CLASS_NAMES[0], 1.0 - plastic_probability, plastic_probability)


def fit_for_dashboard(bgr_image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = bgr_image.shape[:2]
    max_width = DASHBOARD_WIDTH - (PADDING * 2)
    width_scale = max_width / max(width, 1)
    height_scale = MAX_IMAGE_HEIGHT / max(height, 1)
    scale = min(width_scale, height_scale, 1.0)
    display_width = max(1, round(width * scale))
    display_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr_image, (display_width, display_height), interpolation=interpolation)
    return resized, (display_width, display_height)


def draw_prediction(bgr_image: np.ndarray, result: PredictionResult, image_path: Path) -> np.ndarray:
    """Create a simple dashboard screenshot with a box around the analysed region."""
    display_image, (display_width, display_height) = fit_for_dashboard(bgr_image)
    canvas_height = PANEL_HEIGHT + display_height + PADDING
    canvas = np.full((canvas_height, DASHBOARD_WIDTH, 3), 245, dtype=np.uint8)

    color = (0, 150, 0) if result.label == "Marine Life" else (0, 105, 255)
    cv2.rectangle(canvas, (0, 0), (DASHBOARD_WIDTH, PANEL_HEIGHT), color, -1)

    title = "Marine Debris Classifier"
    prediction = f"{result.label} | confidence {result.confidence:.1%}"
    note = "Box shows the full image region analysed by the classifier."
    cv2.putText(canvas, title, (PADDING, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, prediction, (PADDING, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, note, (PADDING, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)

    image_x = (DASHBOARD_WIDTH - display_width) // 2
    image_y = PANEL_HEIGHT
    canvas[image_y : image_y + display_height, image_x : image_x + display_width] = display_image

    # The box marks the full image region passed to the classifier.
    inset = max(5, round(min(display_width, display_height) * 0.025))
    top_left = (image_x + inset, image_y + inset)
    bottom_right = (image_x + display_width - inset, image_y + display_height - inset)
    thickness = max(3, round(min(display_width, display_height) * 0.012))
    cv2.rectangle(canvas, top_left, bottom_right, color, thickness)

    filename = image_path.name[:72]
    footer_y = canvas_height - 10
    cv2.putText(canvas, filename, (PADDING, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1, cv2.LINE_AA)
    return canvas


def output_name(image_path: Path, input_root: Path | None) -> str:
    if input_root is None:
        return f"{image_path.stem}_annotated.jpg"
    relative = image_path.relative_to(input_root)
    safe_stem = "__".join(relative.with_suffix("").parts)
    return f"{safe_stem}_annotated.jpg"


def save_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    csv_path = output_dir / "predictions.csv"
    fieldnames = (
        "image",
        "output",
        "expected_label",
        "prediction",
        "correct",
        "confidence",
        "plastic_probability",
    )
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    checked_rows = [row for row in rows if row["correct"] != ""]
    correct_count = sum(1 for row in checked_rows if row["correct"] is True)
    accuracy = correct_count / len(checked_rows) if checked_rows else None
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "note": "This classifier labels the full analysed image region, not separate detected objects.",
        "images_processed": len(rows),
        "checked_images": len(checked_rows),
        "checked_accuracy": accuracy,
        "predictions": rows,
    }
    (output_dir / "predictions.json").write_text(json.dumps(summary, indent=2) + "\n")


def maybe_display(window_name: str, image: np.ndarray, should_display: bool) -> None:
    if not should_display:
        return
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Trained model was not found: {args.model_path}")

    image_paths = [args.image] if args.image else find_images(args.input_dir, args.limit)
    input_root = args.input_dir if args.input_dir else None
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output folder is not empty: {args.output_dir}. Use --overwrite or choose another folder."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model_path)
    prediction_rows: list[dict[str, object]] = []

    for image_path in image_paths:
        bgr_image = load_bgr_image(image_path)
        result = predict_image(model, bgr_image, args.threshold)
        annotated = draw_prediction(bgr_image, result, image_path)

        destination = args.output_dir / output_name(image_path, input_root)
        if not cv2.imwrite(str(destination), annotated):
            raise OSError(f"Could not save the annotated image to: {destination}")
        maybe_display("Marine Debris Classifier", annotated, args.display)

        expected_label = expected_label_from_path(image_path)
        correct = result.label == expected_label if expected_label is not None else ""
        prediction_rows.append(
            {
                "image": str(image_path),
                "output": str(destination),
                "expected_label": expected_label or "",
                "prediction": result.label,
                "correct": correct,
                "confidence": round(result.confidence, 6),
                "plastic_probability": round(result.plastic_probability, 6),
            }
        )
        print(f"{image_path} -> {result.label} ({result.confidence:.1%})")

    save_summary(prediction_rows, args.output_dir)
    print(f"Saved annotated output to: {args.output_dir}")


if __name__ == "__main__":
    main()
