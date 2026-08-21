#!/usr/bin/env python3
"""Evaluate a saved Plastic-Pulse model on the untouched test images."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path("outputs/.matplotlib-cache").resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_loader import load_test_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the unseen SeaClear test set.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_empty_or_overwrite(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(labels.astype(int), predictions.astype(int), strict=True):
        matrix[actual, predicted] += 1
    return matrix


def save_confusion_matrix(matrix: np.ndarray, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Unseen Test Set Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=(0, 1),
        yticks=(0, 1),
        xticklabels=("Marine life", "Plastic"),
        yticklabels=("Marine life", "Plastic"),
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Saved model was not found: {args.model_path}")

    require_empty_or_overwrite(args.output_dir, args.overwrite)
    test_dataset = load_test_dataset(args.test_dir, batch_size=args.batch_size)
    model = tf.keras.models.load_model(args.model_path)
    keras_metrics = model.evaluate(test_dataset, return_dict=True, verbose=0)

    labels: list[int] = []
    probabilities: list[float] = []
    for images, batch_labels in test_dataset:
        batch_probabilities = model(images, training=False).numpy().reshape(-1)
        labels.extend(batch_labels.numpy().astype(int).reshape(-1).tolist())
        probabilities.extend(batch_probabilities.tolist())

    true_labels = np.array(labels, dtype=int)
    predicted_labels = (np.array(probabilities) >= 0.5).astype(int)
    matrix = confusion_matrix(true_labels, predicted_labels)
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-12)

    results = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "class_order": {"marine_life": 0, "plastic": 1},
        "test_images": int(len(true_labels)),
        "keras_metrics": {name: float(value) for name, value in keras_metrics.items()},
        "threshold": 0.5,
        "confusion_matrix": matrix.tolist(),
        "plastic_precision": precision,
        "plastic_recall": recall,
        "plastic_f1_score": f1_score,
    }
    (args.output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2) + "\n")
    save_confusion_matrix(matrix, args.output_dir / "confusion_matrix.png")

    print("Unseen-data evaluation completed.")
    print(f"Test images: {len(true_labels)}")
    print(f"Loss: {results['keras_metrics']['loss']:.4f}")
    print(f"Accuracy: {results['keras_metrics']['accuracy']:.4f}")
    print(f"Plastic precision / recall / F1: {precision:.4f} / {recall:.4f} / {f1_score:.4f}")
    print(f"Confusion matrix: {matrix.tolist()}")


if __name__ == "__main__":
    main()
