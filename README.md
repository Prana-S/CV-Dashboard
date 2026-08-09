# Plastic-Pulse Ocean Tracker

A student computer-vision project that classifies an image as **marine life** or **plastic debris**.

## Dataset

This project uses the [SeaClear Marine Debris Detection & Segmentation Dataset](https://data.4tu.nl/datasets/4f1dff25-e157-4399-a5d4-478055461689/1) by Đuraš et al. It is available under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

The original dataset archive, extracted images, and generated training data stay in `data/raw/` and `data/processed/` on your computer. They are ignored by Git and will not be uploaded to GitHub.

## Week 1

Build 500 balanced source crops for each class:

```bash
python3 src/build_seaclear_binary_dataset.py --per-class 500 --overwrite
```

Then create the training-ready dataset:

```bash
python3 src/prepare_dataset.py \
  --marine-source data/raw/marine_life \
  --plastic-source data/raw/plastic \
  --output-dir data/processed \
  --overwrite
```

The scripts keep training and validation images separate, resize images to 224 x 224, create training-only augmentations, and apply `pixel / 255.0` normalization during model loading.
