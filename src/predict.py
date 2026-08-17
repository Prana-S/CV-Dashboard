#!/usr/bin/env python3
"""Classify a single image with a model trained by ``src/train_model.py``.

Example::

    python3 src/predict.py --model outputs/best_model.keras --image photo.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

CLASS_NAMES = ("marine_life", "plastic")
IMAGE_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict marine life vs. plastic debris.")
    parser.add_argument("--model", type=Path, default=Path("outputs/best_model.keras"),
                        help="Trained .keras model file (default: outputs/best_model.keras).")
    parser.add_argument("--image", type=Path, required=True, help="Image file to classify.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability above which the image is called plastic (default: 0.5).")
    return parser.parse_args()


def load_image(image_path: Path) -> np.ndarray:
    """Load one RGB image as a 224 x 224 batch of raw 0-255 values.

    The model rescales to 0-1 internally, so no normalization happens here.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = tf.keras.utils.load_img(image_path, color_mode="rgb", target_size=IMAGE_SIZE)
    return np.expand_dims(tf.keras.utils.img_to_array(image), axis=0)


def predict(model: tf.keras.Model, image_path: Path, threshold: float = 0.5) -> dict[str, object]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    probability = float(model.predict(load_image(image_path), verbose=0)[0][0])
    is_plastic = probability >= threshold
    return {
        "image": str(image_path),
        "plastic_probability": probability,
        "predicted_class": CLASS_NAMES[1] if is_plastic else CLASS_NAMES[0],
        "confidence": probability if is_plastic else 1.0 - probability,
    }


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Train one first with src/train_model.py."
        )
    model = tf.keras.models.load_model(args.model)
    result = predict(model, args.image, args.threshold)
    print(f"Image:               {result['image']}")
    print(f"Predicted class:     {result['predicted_class']}")
    print(f"Plastic probability: {result['plastic_probability']:.4f}")
    print(f"Confidence:          {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
