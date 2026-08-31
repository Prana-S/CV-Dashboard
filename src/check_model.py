#!/usr/bin/env python3
"""Build, inspect, and optionally save the Week 2 model without training it."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_loader import load_datasets
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the Marine Debris Classifier data loader and model architecture.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--save-model",
        type=Path,
        help="Optional local .keras path for the compiled Week 2 architecture.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        help="Optional local text file for the model summary.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing model or summary file to be replaced.",
    )
    return parser.parse_args()


def require_writable_destination(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def count_trainable_parameters(model: object) -> int:
    """Return the number of parameters that will change during the first training stage."""
    return sum(int(weight.shape.num_elements()) for weight in model.trainable_weights)


def model_summary_text(model: object) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    datasets = load_datasets(args.data_dir, batch_size=args.batch_size)
    images, labels = next(iter(datasets.training))
    model = build_model()
    probabilities = model(images, training=False)

    if images.shape[1:] != (224, 224, 3):
        raise ValueError(f"Unexpected image shape: {images.shape}")
    if probabilities.shape[-1] != 1:
        raise ValueError(f"Unexpected prediction shape: {probabilities.shape}")
    if float(images.numpy().min()) < 0 or float(images.numpy().max()) > 1:
        raise ValueError("Images were not normalized to the expected 0-1 range.")

    summary = model_summary_text(model)
    if args.summary_file:
        require_writable_destination(args.summary_file, args.overwrite)
        args.summary_file.write_text(summary)
    if args.save_model:
        require_writable_destination(args.save_model, args.overwrite)
        # Build Adam's variables before saving so a later reload gets the complete
        # compiled optimizer state, even though Week 2 has not trained the model yet.
        model.optimizer.build(model.trainable_variables)
        model.save(args.save_model)

        # Reopening the saved file proves it is a usable compiled Keras model, not only a diagram.
        import tensorflow as tf

        reloaded_model = tf.keras.models.load_model(args.save_model)
        reloaded_output = reloaded_model(images, training=False)
        if reloaded_output.shape != probabilities.shape:
            raise ValueError("Saved model output shape did not match the original model.")

    print("Week 2 architecture check passed.")
    print(f"Class order: {datasets.class_names} (0 = marine life, 1 = plastic)")
    print(f"Input batch shape: {images.shape}; label shape: {labels.shape}")
    print(f"Prediction shape: {probabilities.shape}")
    print(f"Prediction range: {float(probabilities.numpy().min()):.4f} to {float(probabilities.numpy().max()):.4f}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    if args.save_model:
        print(f"Saved compiled model: {args.save_model}")
    if args.summary_file:
        print(f"Saved model summary: {args.summary_file}")


if __name__ == "__main__":
    main()
