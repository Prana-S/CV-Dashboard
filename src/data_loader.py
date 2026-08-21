"""Load the prepared Plastic-Pulse image folders for training and evaluation."""

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
    """The training and validation datasets plus their fixed class order."""

    training: "tf.data.Dataset"
    validation: "tf.data.Dataset"
    class_names: tuple[str, str]


def _class_directories_are_present(data_dir: Path) -> None:
    for class_name in CLASS_NAMES:
        expected = data_dir / class_name
        if not expected.is_dir():
            raise FileNotFoundError(f"Expected image folder was not found: {expected}")


def normalize_batch(images: object, labels: object) -> tuple[object, object]:
    """Convert JPEG pixel values from 0-255 to the Week 1 0-1 range."""
    import tensorflow as tf

    return tf.cast(images, tf.float32) / 255.0, labels


def _load_image_directory(
    directory: Path,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> "tf.data.Dataset":
    import tensorflow as tf

    _class_directories_are_present(directory)
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        class_names=list(CLASS_NAMES),
        color_mode="rgb",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    autotune = tf.data.AUTOTUNE
    prepared = dataset.map(normalize_batch, num_parallel_calls=autotune).prefetch(autotune)

    # A small private pool keeps TensorFlow stable on laptops with limited thread resources.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    options.threading.max_intra_op_parallelism = 1
    return prepared.with_options(options)


def load_datasets(
    data_dir: Path = Path("data/processed"),
    batch_size: int = 32,
    seed: int = 42,
) -> DatasetBundle:
    """Load the existing Week 1 train/validation folders without making a new split."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    training = _load_image_directory(data_dir / "train", batch_size, shuffle=True, seed=seed)
    validation = _load_image_directory(data_dir / "validation", batch_size, shuffle=False, seed=seed)
    return DatasetBundle(training=training, validation=validation, class_names=CLASS_NAMES)


def load_test_dataset(
    test_dir: Path = Path("data/test"),
    batch_size: int = 32,
) -> "tf.data.Dataset":
    """Load the untouched test folder used only for the final Week 3 evaluation."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return _load_image_directory(test_dir, batch_size, shuffle=False, seed=42)
