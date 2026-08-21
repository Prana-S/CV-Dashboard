#!/usr/bin/env python3
"""Compare two Week 3 training runs using their saved history files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path("outputs/.matplotlib-cache").resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and tuned training experiments.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--tuned-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_history(experiment_dir: Path) -> dict[str, list[float]]:
    history_path = experiment_dir / "training_history.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"Training history was not found: {history_path}")
    return json.loads(history_path.read_text())


def best_epoch(history: dict[str, list[float]]) -> int:
    return min(range(len(history["val_loss"])), key=lambda index: history["val_loss"][index])


def experiment_summary(name: str, history: dict[str, list[float]]) -> dict[str, float | int | str]:
    index = best_epoch(history)
    return {
        "experiment": name,
        "epochs_completed": len(history["loss"]),
        "best_epoch": index + 1,
        "train_accuracy_at_best_epoch": history["accuracy"][index],
        "validation_accuracy_at_best_epoch": history["val_accuracy"][index],
        "train_loss_at_best_epoch": history["loss"][index],
        "validation_loss_at_best_epoch": history["val_loss"][index],
        "learning_rate_at_best_epoch": history["learning_rate"][index],
    }


def plot_validation_curves(
    baseline: dict[str, list[float]],
    tuned: dict[str, list[float]],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, history, color in (("Baseline", baseline, "#4C78A8"), ("Tuned", tuned, "#F58518")):
        epochs = range(1, len(history["val_accuracy"]) + 1)
        axes[0].plot(epochs, history["val_accuracy"], marker="o", color=color, label=name)
        axes[1].plot(epochs, history["val_loss"], marker="o", color=color, label=name)

    axes[0].set(title="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
    axes[1].set(title="Validation Loss", xlabel="Epoch", ylabel="Binary cross-entropy loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Week 3 Experiment Comparison")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Use --overwrite to replace it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_history(args.baseline_dir)
    tuned = load_history(args.tuned_dir)
    summary = {
        "baseline": experiment_summary("baseline", baseline),
        "tuned": experiment_summary("tuned", tuned),
    }
    (args.output_dir / "experiment_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_validation_curves(baseline, tuned, args.output_dir / "validation_comparison.png")

    print("Experiment comparison created.")
    for name, values in summary.items():
        print(
            f"{name}: epoch {values['best_epoch']}, "
            f"val_accuracy={values['validation_accuracy_at_best_epoch']:.4f}, "
            f"val_loss={values['validation_loss_at_best_epoch']:.4f}"
        )


if __name__ == "__main__":
    main()
