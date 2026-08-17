"""Architecture tests for the Week 2 model.

The model is always built with ``use_imagenet_weights=False`` so the tests run
offline and never download the pretrained weight file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_model import INPUT_SHAPE, build_model  # noqa: E402


@pytest.fixture(scope="module")
def model() -> tf.keras.Model:
    return build_model(use_imagenet_weights=False)


def test_input_shape_is_224x224x3(model: tf.keras.Model) -> None:
    assert INPUT_SHAPE == (224, 224, 3)
    assert model.input_shape == (None, 224, 224, 3)


def test_single_sigmoid_output(model: tf.keras.Model) -> None:
    assert model.output_shape == (None, 1)
    assert model.layers[-1].activation is tf.keras.activations.sigmoid


def test_uses_frozen_mobilenetv2_without_top(model: tf.keras.Model) -> None:
    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    assert "mobilenet" in backbone.name.lower()
    assert backbone.trainable is False
    # include_top=False keeps the 7x7 feature map instead of a 1000-class output.
    assert backbone.output_shape[1:3] == (7, 7)


def test_head_has_pooling_and_dropout(model: tf.keras.Model) -> None:
    layer_types = [type(layer) for layer in model.layers]
    assert tf.keras.layers.GlobalAveragePooling2D in layer_types
    assert tf.keras.layers.Dropout in layer_types


def test_only_the_head_is_trainable(model: tf.keras.Model) -> None:
    trainable = sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
    # 1280 pooled features + 1 bias.
    assert trainable == 1281


def test_compiled_with_binary_crossentropy_and_adam(model: tf.keras.Model) -> None:
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    loss = model.loss
    assert getattr(loss, "name", loss) == "binary_crossentropy"


def test_classification_metrics_are_configured(model: tf.keras.Model) -> None:
    metric_names = {metric["config"]["name"] for metric in model.get_compile_config()["metrics"]}
    assert {"accuracy", "precision", "recall", "auc"} <= metric_names


def test_predicts_probability_between_zero_and_one(model: tf.keras.Model) -> None:
    batch = np.zeros((2, *INPUT_SHAPE), dtype="float32")
    predictions = model.predict(batch, verbose=0)
    assert predictions.shape == (2, 1)
    assert np.all((predictions >= 0.0) & (predictions <= 1.0))


def test_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError):
        build_model(dropout=1.0, use_imagenet_weights=False)
    with pytest.raises(ValueError):
        build_model(learning_rate=0.0, use_imagenet_weights=False)
