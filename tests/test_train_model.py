"""Week 2 model tests: architecture and argument checks, no dataset or GPU needed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train_model  # noqa: E402


@pytest.fixture(scope="module")
def model():
    # Random weights keep the test offline; the architecture is identical.
    return train_model.build_model(use_imagenet_weights=False)


def test_input_and_output_shapes(model):
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 1)


def test_head_layers_present(model):
    names = [layer.name for layer in model.layers]
    assert "global_average_pooling" in names
    assert "dropout" in names
    output_layer = model.get_layer("plastic_probability")
    assert output_layer.units == 1
    assert output_layer.activation.__name__ == "sigmoid"


def test_backbone_is_frozen_and_has_no_imagenet_classifier(model):
    backbone = next(layer for layer in model.layers if layer.name.startswith("mobilenetv2"))
    assert backbone.trainable is False
    assert len(backbone.output_shape) == 4  # feature maps, not 1000 class logits


def test_compiled_with_expected_loss_optimizer_and_metrics(model):
    assert model.loss == "binary_crossentropy"
    assert model.optimizer.__class__.__name__ == "Adam"
    # Keras only names the compiled metrics once they have seen a batch.
    results = model.evaluate(
        np.zeros((2, 224, 224, 3), dtype="float32"),
        np.array([[0.0], [1.0]], dtype="float32"),
        verbose=0,
        return_dict=True,
    )
    for expected in ("accuracy", "precision", "recall", "auc"):
        assert expected in results


def test_prediction_is_a_probability(model):
    batch = np.zeros((2, 224, 224, 3), dtype="float32")
    predictions = model.predict(batch, verbose=0)
    assert predictions.shape == (2, 1)
    assert np.all((predictions >= 0) & (predictions <= 1))


def test_class_mapping_matches_week1():
    assert train_model.CLASS_NAMES == ("marine_life", "plastic")
    assert train_model.CLASS_INDICES == {"marine_life": 0, "plastic": 1}


@pytest.mark.parametrize("flag", ["--epochs", "--batch-size", "--learning-rate"])
def test_non_positive_values_are_rejected(flag):
    args = train_model.parse_args([flag, "0"])
    with pytest.raises(ValueError):
        train_model.validate_args(args)


def test_dropout_must_be_below_one():
    with pytest.raises(ValueError):
        train_model.validate_args(train_model.parse_args(["--dropout", "1.0"]))


def test_defaults_are_valid():
    train_model.validate_args(train_model.parse_args([]))


def test_missing_dataset_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_dataset.py"):
        train_model.load_datasets(tmp_path, batch_size=2, seed=0)


def test_missing_class_folder_raises_clear_error(tmp_path):
    (tmp_path / "train" / "marine_life").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="plastic"):
        train_model.load_datasets(tmp_path, batch_size=2, seed=0)
