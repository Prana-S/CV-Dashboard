"""Load the prepared Week 1 images for the Plastic-Pulse classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tensorflow as tf

CLASS_NAMES = ("marine_life", "plastic")
IMAGE_SIZE = (224, 224)


@dataclass(frozen=True)
class DatasetBundle:
    """The two prepared datasets plus the class order used by the model."""

    training: "tf.data.Dataset"
    validation: "tf.data.Dataset"
    class_names: tuple[str, str]


def _class_directories_are_present(data_dir: Path) -> None:
    for split in ("train", "validation"):
        for class_name in CLASS_NAMES:
            expected = data_dir / split / class_name
            if not expected.is_dir():
                raise FileNotFoundError(f"Expected image folder was not found: {expected}")


def normalize_batch(images: object, labels: object) -> tuple[object, object]:
    """Convert JPEG pixel values from 0-255 to the Week 1 0-1 range."""
    import tensorflow as tf

    return tf.cast(images, tf.float32) / 255.0, labels


def load_datasets(
    data_dir: Path = Path("data/processed"),
    batch_size: int = 32,
    seed: int = 42,
) -> DatasetBundle:
    """Load the existing train/validation folders without creating a new split."""
    import tensorflow as tf

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    _class_directories_are_present(data_dir)

    loader_settings = {
        "class_names": list(CLASS_NAMES),
        "color_mode": "rgb",
        "label_mode": "binary",
        "image_size": IMAGE_SIZE,
        "batch_size": batch_size,
    }
    training = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        shuffle=True,
        seed=seed,
        **loader_settings,
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        data_dir / "validation",
        shuffle=False,
        **loader_settings,
    )

    # Week 1 records this 0-1 conversion as the shared preprocessing contract.
    autotune = tf.data.AUTOTUNE
    training = training.map(normalize_batch, num_parallel_calls=autotune).prefetch(autotune)
    validation = validation.map(normalize_batch, num_parallel_calls=autotune).prefetch(autotune)

    return DatasetBundle(training=training, validation=validation, class_names=CLASS_NAMES)
