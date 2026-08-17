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

## Week 2: model architecture

`src/train_model.py` builds and compiles the classifier with transfer learning:

```text
224 x 224 x 3 image
        |
Rescaling(1/255)
        |
MobileNetV2 (include_top=False, ImageNet weights, frozen)
        |
Global Average Pooling
        |
Dropout
        |
Dense(1, sigmoid)  ->  probability that the image shows plastic
```

The pretrained MobileNetV2 backbone already knows generic visual features such as
edges, colours, shapes, and textures, so only the small classification head is
trained. Global average pooling turns the feature maps into one 1280-value
vector, dropout reduces overfitting, and the single sigmoid neuron outputs a
probability: near `0` means marine life and near `1` means plastic debris. The
model is compiled with binary cross-entropy loss, the Adam optimizer, and
accuracy, precision, recall, and AUC metrics.

### Train the model

```bash
pip install -r requirements.txt

python3 src/train_model.py \
  --data-dir data/processed \
  --output-dir outputs \
  --epochs 10 \
  --batch-size 32
```

This downloads the ImageNet weights on first run, so it needs internet access.
Add `--no-imagenet-weights` to build the same architecture offline; that is only
useful for architecture checks and tests, not for real transfer learning.

### Predict a single image

```bash
python3 src/predict.py --model outputs/best_model.keras --image path/to/image.jpg
```

### Source code vs. generated artifacts

- `src/` holds the **model source code** and is tracked in Git.
- `outputs/` holds the **trained model files** that training generates on your
  own computer. It is listed in `.gitignore`, together with `*.keras`, so the
  binaries are never pushed to GitHub unless someone explicitly asks for them.

After the training command finishes you will find, relative to the repository root:

```text
outputs/best_model.keras       # best validation AUC checkpoint
outputs/final_model.keras      # model after the last epoch
outputs/training_history.json  # per-epoch loss and metrics
```

### Run the tests

```bash
python3 -m pytest tests
```

The tests build the model with `use_imagenet_weights=False` so they run offline.
