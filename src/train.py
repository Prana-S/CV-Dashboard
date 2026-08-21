#!/usr/bin/env python3
"""Train the Week 2 MobileNetV2 model and save Week 3 learning evidence."""

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
import tensorflow as tf

from data_loader import load_datasets
from model import build_model


class LearningRateHistory(tf.keras.callbacks.Callback):
    """Add the learning rate to Keras history so each experiment can be compared."""

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        del epoch
        if logs is not None:
            logs["learning_rate"] = float(self.model.optimizer.learning_rate.numpy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Plastic-Pulse binary classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout-rate", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_empty_or_overwrite(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)


def write_model_summary(model: tf.keras.Model, destination: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    destination.write_text("\n".join(lines) + "\n")


def save_history(history: tf.keras.callbacks.History, destination: Path) -> dict[str, list[float]]:
    values = {
        name: [float(value) for value in series]
        for name, series in history.history.items()
    }
    destination.write_text(json.dumps(values, indent=2) + "\n")
    return values


def plot_learning_curves(history: dict[str, list[float]], destination: Path) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["accuracy"], marker="o", label="Training")
    axes[0].plot(epochs, history["val_accuracy"], marker="o", label="Validation")
    axes[0].set(title="Model Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, history["loss"], marker="o", label="Training")
    axes[1].plot(epochs, history["val_loss"], marker="o", label="Validation")
    axes[1].set(title="Model Loss", xlabel="Epoch", ylabel="Binary cross-entropy loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.suptitle("Plastic-Pulse Training History")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    require_empty_or_overwrite(args.output_dir, args.overwrite)
    tf.keras.utils.set_random_seed(args.seed)
    datasets = load_datasets(args.data_dir, batch_size=args.batch_size, seed=args.seed)
    model = build_model(args.learning_rate, args.dropout_rate)

    best_model_path = args.output_dir / "best_model.keras"
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        LearningRateHistory(),
    ]

    history = model.fit(
        datasets.training,
        validation_data=datasets.validation,
        epochs=args.epochs,
        callbacks=callbacks,
        # The tf.data loader already shuffles the training set.
        shuffle=False,
        verbose=2,
    )
    model.save(args.output_dir / "final_model.keras")
    history_values = save_history(history, args.output_dir / "training_history.json")
    plot_learning_curves(history_values, args.output_dir / "learning_curves.png")
    write_model_summary(model, args.output_dir / "model_summary.txt")

    configuration = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "class_order": {"marine_life": 0, "plastic": 1},
        "epochs_requested": args.epochs,
        "epochs_completed": len(history_values["loss"]),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout_rate": args.dropout_rate,
        "seed": args.seed,
    }
    (args.output_dir / "training_config.json").write_text(json.dumps(configuration, indent=2) + "\n")

    print("Training completed.")
    print(f"Best validation model: {best_model_path}")
    print(f"Learning curves: {args.output_dir / 'learning_curves.png'}")
    print(f"Training history: {args.output_dir / 'training_history.json'}")


if __name__ == "__main__":
    main()
