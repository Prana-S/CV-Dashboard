#!/usr/bin/env python3
"""Week 2: build, compile, and train the marine-life vs. plastic classifier.

The architecture is a transfer-learning CNN:

    224 x 224 x 3 image
            |
    Rescaling(1/255)                 <- same contract as preprocessing_config.json
            |
    MobileNetV2 (include_top=False)  <- frozen pretrained feature extractor
            |
    GlobalAveragePooling2D
            |
    Dropout
            |
    Dense(1, activation="sigmoid")   <- probability that the image shows plastic

Trained artifacts are written to ``--output-dir`` (``outputs/`` by default),
which is ignored by Git; only this source file is version controlled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

CLASS_NAMES = ("marine_life", "plastic")
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MobileNetV2 transfer-learning classifier on the prepared dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"),
                        help="Dataset directory containing train/ and validation/ folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Where the trained .keras files are written (default: outputs).")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Adam learning rate (default: 0.001).")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate in the classification head (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--no-imagenet-weights", action="store_true",
                        help="Build the backbone without downloading ImageNet weights "
                             "(offline architecture checks only, not real transfer learning).")
    return parser.parse_args()


def build_model(
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    use_imagenet_weights: bool = True,
) -> tf.keras.Model:
    """Return the compiled transfer-learning model."""
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet" if use_imagenet_weights else None,
    )
    # Freezing keeps the pretrained features intact while the small head learns.
    base_model.trainable = False

    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255, name="rescaling")(inputs)
    # training=False keeps the frozen BatchNorm layers in inference mode.
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="plastic_probability")(x)

    model = tf.keras.Model(inputs, outputs, name="plastic_pulse_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def load_datasets(
    data_dir: Path, batch_size: int, seed: int
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load the train/validation folders produced by ``src/prepare_dataset.py``."""
    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"
    for directory in (train_dir, validation_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Missing dataset folder: {directory}. Run src/prepare_dataset.py first."
            )

    def load(directory: Path, shuffle: bool) -> tf.data.Dataset:
        return tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="binary",
            class_names=list(CLASS_NAMES),
            color_mode="rgb",
            batch_size=batch_size,
            image_size=IMAGE_SIZE,
            shuffle=shuffle,
            seed=seed,
        )

    train_ds = load(train_dir, shuffle=True)
    validation_ds = load(validation_dir, shuffle=False)
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        validation_ds.prefetch(autotune),
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    tf.keras.utils.set_random_seed(args.seed)
    train_ds, validation_ds = load_datasets(args.data_dir, args.batch_size, args.seed)

    model = build_model(
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        use_imagenet_weights=not args.no_imagenet_weights,
    )
    model.summary()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "best_model.keras"
    final_model_path = args.output_dir / "final_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=3, restore_best_weights=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(final_model_path)
    history_path = args.output_dir / "training_history.json"
    history_path.write_text(
        json.dumps({key: [float(value) for value in values]
                    for key, values in history.history.items()}, indent=2) + "\n"
    )

    print("Training finished.")
    print(f"Best model:  {best_model_path}")
    print(f"Final model: {final_model_path}")
    print(f"History:     {history_path}")


if __name__ == "__main__":
    main()
