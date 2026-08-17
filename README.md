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

## Week 2

Install the dependencies (CPU only, no GPU required):

```bash
pip install -r requirements.txt
```

### Model architecture

```text
Input image: 224 x 224 x 3
        ↓
Rescaling: pixel / 255.0
        ↓
Pretrained MobileNetV2 backbone (ImageNet weights, top layer removed, frozen)
        ↓
Global Average Pooling
        ↓
Dropout (0.2)
        ↓
Dense(1) with sigmoid
        ↓
Plastic probability (marine_life = 0, plastic = 1)
```

### Transfer-learning workflow

1. MobileNetV2 is loaded with its ImageNet weights and without its original
   1000-class classification layer, so it is used purely as a feature extractor.
2. The backbone is frozen, so only the new classification head learns during
   this first phase. This keeps training fast on a CPU and works well with a
   small dataset.
3. The custom head reduces the feature maps with global average pooling, drops
   some features during training to limit overfitting, and ends with a single
   sigmoid neuron for the binary decision.
4. The model is compiled with binary cross-entropy loss, the Adam optimizer,
   and accuracy, precision, recall, and AUC metrics.

### Train the model

```bash
python3 src/train_model.py \
  --data-dir data/processed \
  --output-dir outputs \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.001
```

The first run downloads the ImageNet weights (about 9 MB). Without internet
access, add `--no-imagenet-weights` to build the same architecture with random
backbone weights.

Print and check only the compiled architecture, without training:

```bash
python3 src/train_model.py --summary-only
```

The best checkpoint (`outputs/best_model.keras`), the final model, the training
log, and `outputs/training_history.json` are written to the output folder, which
is ignored by Git.

### Predict a single image

```bash
python3 src/predict.py --model outputs/best_model.keras --image path/to/image.jpg
```

### Run the tests

```bash
python3 -m pytest tests
```

### Short explanation for the report

> The classifier uses transfer learning. MobileNetV2, a convolutional neural
> network that was already trained on the large ImageNet collection, is reused
> as the feature extractor. Its early layers recognise edges and colours and its
> deeper layers recognise shapes and textures, so it already knows how to
> describe an image before it sees any underwater data. Its original
> classification layer is removed and its weights are frozen, which means the
> pretrained knowledge is kept and only the new layers are trained. On top of it
> a custom head is added: global average pooling turns each feature map into a
> single number, dropout randomly ignores part of the features so the model does
> not memorise the training images, and one dense neuron with a sigmoid function
> outputs the probability that the image shows plastic. A value near 0 means
> marine life and a value near 1 means plastic debris. The model is compiled with
> binary cross-entropy loss because there are two classes, the Adam optimizer,
> and accuracy, precision, recall, and AUC as metrics.
