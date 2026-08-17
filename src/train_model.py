#!/usr/bin/env python3
"""Week 2: build, compile, and train the marine-life vs. plastic classifier.

The model reuses a pretrained MobileNetV2 backbone as a frozen feature
extractor (transfer learning) and adds a small custom classification head for
the binary decision.  Preprocessing follows the Week 1 contract stored in
``data/processed/preprocessing_config.json``: RGB images of 224 x 224 pixels
with ``pixel_value / 255.0`` normalization.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = 224
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
# Same order and indices as the Week 1 dataset manifest.
CLASS_NAMES = ("marine_life", "plastic")
CLASS_INDICES = {"marine_life": 0, "plastic": 1}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MobileNetV2 transfer-learning classifier for marine life vs. plastic."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"),
                        help="Folder holding train/ and validation/ subfolders (default: data/processed).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Folder for checkpoints and training history (default: outputs).")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Adam learning rate (default: 0.001).")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate in the classification head (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--no-imagenet-weights", action="store_true",
                        help="Build the backbone without ImageNet weights (offline or testing use).")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only build, compile, and print the architecture; do not train.")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in the range [0, 1)")


def set_seeds(seed: int) -> None:
    """Make CPU runs as repeatable as Keras allows."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def build_model(
    learning_rate: float = 1e-3,
    dropout: float = 0.2,
    use_imagenet_weights: bool = True,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
) -> keras.Model:
    """Return the compiled transfer-learning architecture.

    Pretrained backbone -> global average pooling -> dropout -> sigmoid Dense(1).
    """
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,  # drop the 1000-class ImageNet classifier
        weights="imagenet" if use_imagenet_weights else None,
    )
    # Feature-extraction phase: the pretrained filters stay exactly as they were.
    backbone.trainable = False

    inputs = keras.Input(shape=input_shape, name="image")
    # Week 1 stores plain JPEG pixels, so normalization happens inside the model.
    scaled = layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(inputs)
    # MobileNetV2 was pretrained on inputs in [-1, 1]; map the 0-1 values onto that range.
    scaled = layers.Rescaling(2.0, offset=-1.0, name="mobilenet_preprocess")(scaled)
    features = backbone(scaled, training=False)
    pooled = layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    regularized = layers.Dropout(dropout, name="dropout")(pooled)
    # One neuron with sigmoid: 0 -> marine_life, 1 -> plastic.
    outputs = layers.Dense(1, activation="sigmoid", name="plastic_probability")(regularized)

    model = keras.Model(inputs, outputs, name="marine_vs_plastic_mobilenetv2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def _split_dir(data_dir: Path, split: str) -> Path:
    directory = data_dir / split
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing dataset folder: {directory}. Run src/prepare_dataset.py first "
            "to create data/processed/train and data/processed/validation."
        )
    for class_name in CLASS_NAMES:
        if not (directory / class_name).is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {directory / class_name}. Expected subfolders "
                f"{list(CLASS_NAMES)} created by src/prepare_dataset.py."
            )
    return directory


def load_datasets(
    data_dir: Path, batch_size: int, seed: int, image_size: int = IMAGE_SIZE
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load the Week 1 folders as batched, prefetched binary-label datasets."""
    datasets = []
    for split in ("train", "validation"):
        directory = _split_dir(data_dir, split)
        dataset = keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="binary",
            class_names=list(CLASS_NAMES),  # keeps marine_life=0, plastic=1
            color_mode="rgb",
            image_size=(image_size, image_size),
            batch_size=batch_size,
            shuffle=split == "train",
            seed=seed,
        )
        datasets.append(dataset.prefetch(tf.data.AUTOTUNE))
    return datasets[0], datasets[1]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    set_seeds(args.seed)

    model = build_model(
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        use_imagenet_weights=not args.no_imagenet_weights,
    )
    model.summary()
    if args.summary_only:
        return

    train_dataset, validation_dataset = load_datasets(args.data_dir, args.batch_size, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(args.output_dir / "training_log.csv")),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.output_dir / "final_model.keras")
    history_path = args.output_dir / "training_history.json"
    history_path.write_text(json.dumps(
        {key: [float(value) for value in values] for key, values in history.history.items()},
        indent=2,
    ) + "\n")

    print("Week 2 training finished.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Final model: {args.output_dir / 'final_model.keras'}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
