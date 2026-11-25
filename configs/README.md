# Kvasir-SEG comparison protocol

Run commands from the repository root. The training script reads `configs/comparison.json` by default. Explicit command-line options override the file. Use the unmodified settings for reported runs.

| Encoder ID | Owner | Initialization |
| --- | --- | --- |
| `scratch` | Team baseline | Existing U-Net, random weights, 64 base channels |
| `resnet34` | Tuan | ImageNet-1K encoder |
| `resnet50` | Hieu | ImageNet-1K encoder |
| `efficientnet-b0` | Dung | ImageNet-1K encoder |

The scratch model is implemented. Register the three pretrained variants in `src.models.build_model` as they are developed. Use the same five-stage U-Net decoder with widths 256, 128, 64, 32 and 16, 2x upsampling and encoder skip connections at matching resolutions. Initialize the decoder from scratch and fine-tune every layer from epoch one. Record the exact encoder weights used. EfficientNet-B2 is an optional additional run.

## Training and scoring

- Use the committed train, validation and test manifests (800/100/100). Shuffle train only. Use no augmentation.
- Resize RGB images to the configured square size with bilinear interpolation, divide by 255, then apply the RGB means and standard deviations from the JSON. Resize masks with nearest-neighbor interpolation and label pixels >= 128 as foreground.
- Train in FP32 with Adam, constant learning rate and no early stopping. Use mean BCEWithLogitsLoss with equal pixel weights on a single output logit per pixel.
- Select the earliest epoch with the highest validation mean Dice. Evaluate its checkpoint with the shared script. Apply sigmoid and the configured threshold, then average per-image Dice and IoU on the resized test masks.
- Log the reported runs to the configured W&B team and project using `--wandb`. Each run saves the resolved config, best checkpoint and final checkpoint in its own output directory.

## Scratch baseline

```sh
python -m scripts.train --config configs/comparison.json --encoder scratch --wandb
python -m scripts.evaluate --checkpoint outputs/unet-scratch-bce-s42/best.pt --output-dir outputs/test-scratch-bce-s42
```

For pretrained runs, use the corresponding encoder ID after its implementation is registered and give each evaluation its own output directory. Retrain the scratch model from initialization for the BCE comparison.
