# Medical Image Segmentation

This project segments polyps in endoscopy images from [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/). A U-Net trained from scratch predicts a binary mask that labels each pixel as polyp or background.

## Shared BCE comparison

The team uses [the comparison protocol](configs/README.md) and [the shared training configuration](configs/comparison.json) for the scratch and ImageNet encoder experiments. The results below describe the earlier BCE + Tversky run.

## Data and method

- **Exploratory analysis.** Verified all 1,000 image–mask pairs, inspected mask values and polyp area, and created an image-level split of 800 training, 100 validation, and 100 test images.
- **Preprocessing.** Resized RGB images and masks to 256 × 256. Images were scaled to `[0, 1]`. Masks were resized with nearest-neighbor interpolation and converted to binary labels.
- **Model.** The U-Net uses a convolutional encoder, a decoder with transposed convolutions, and skip connections that preserve spatial detail. It outputs one logit per pixel.
- **Loss.** Training combined binary cross-entropy and Tversky loss with equal weights. Tversky used `alpha=0.3` for false positives and `beta=0.7` for false negatives, giving more weight to missed polyp pixels.

## Training and evaluation

The model was trained for 60 epochs on a Kaggle Tesla T4 GPU with Adam, a learning rate of `1e-4`, and a batch size of 4. Weights & Biases logged training and validation loss, Dice, and IoU after each epoch in the team's `BTL_DL_2026` project.

Epoch 59 had the highest validation Dice (**0.8078**). Its checkpoint was evaluated on the 100 test images at 256 × 256 with a probability threshold of 0.5.

| Test metric | Mean across images |
| --- | ---: |
| Dice | **0.8139** |
| IoU | **0.7223** |

The evaluation script saves scores for each test image and visualizes the five lowest- and five highest-Dice cases alongside their ground-truth masks. The reported results use an image-level split. Patient-level separation was not verified.
