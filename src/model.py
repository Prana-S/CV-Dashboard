"""MobileNetV2 transfer-learning model for Plastic-Pulse."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tensorflow as tf

IMAGE_SHAPE = (224, 224, 3)


def build_model(
    learning_rate: float = 0.001,
    dropout_rate: float = 0.30,
) -> "tf.keras.Model":
    """Build a frozen MobileNetV2 feature extractor with a binary classifier head."""
    import tensorflow as tf

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be between 0 and 1")

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SHAPE, name="image")
    # The loader produces 0-1 values. MobileNetV2 expects the ImageNet range of -1 to 1.
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="mobilenet_v2_preprocessing")(inputs)
    # Keeping training=False also keeps MobileNetV2's batch-normalization layers frozen.
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="classifier_dense")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="plastic_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="plastic_pulse_mobilenet_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model
