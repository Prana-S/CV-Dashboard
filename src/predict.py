#!/usr/bin/env python3
"""Classify a single image with the Week 2 trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from train_model import CLASS_NAMES, IMAGE_SIZE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict marine life vs. plastic for one image.")
    parser.add_argument("--model", type=Path, default=Path("outputs/best_model.keras"),
                        help="Trained Keras model file (default: outputs/best_model.keras).")
    parser.add_argument("--image", type=Path, required=True, help="Image file to classify.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability above which the image is called plastic (default: 0.5).")
    return parser.parse_args(argv)


def load_image(path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Return a batch of one RGB image; the model rescales pixels internally."""
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = keras.utils.load_img(path, target_size=(image_size, image_size), color_mode="rgb")
    return np.expand_dims(keras.utils.img_to_array(image), axis=0)


def classify(model: keras.Model, image_batch: np.ndarray, threshold: float = 0.5) -> tuple[str, float]:
    probability = float(model.predict(image_batch, verbose=0)[0][0])
    return CLASS_NAMES[int(probability >= threshold)], probability


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Train it first with src/train_model.py."
        )
    model = keras.models.load_model(args.model)
    label, probability = classify(model, load_image(args.image), args.threshold)
    print(f"Image: {args.image}")
    print(f"Prediction: {label}")
    print(f"Plastic probability: {probability:.4f}")


if __name__ == "__main__":
    main()
