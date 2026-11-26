"""Unit tests for ResNet34UNet, WandbLogger, and training/evaluation integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.engine.losses import BCEDiceLoss
from src.engine.trainer import Trainer
from src.models import ResNet34UNet
from src.utils.logger import WandbLogger


@pytest.mark.parametrize(
    "in_channels,num_classes,height,width,bilinear",
    [
        (3, 1, 256, 256, True),
        (3, 3, 352, 352, False),
        (1, 1, 256, 256, True),
        (3, 1, 250, 250, True),  # Non-multiple of 32
    ],
)
def test_resnet34_unet_output_shape(
    in_channels: int, num_classes: int, height: int, width: int, bilinear: bool
) -> None:
    model = ResNet34UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=False,
        bilinear=bilinear,
    )
    x = torch.randn(2, in_channels, height, width)
    out = model(x)
    assert out.shape == (2, num_classes, height, width)


def test_freeze_and_unfreeze_encoder() -> None:
    model = ResNet34UNet(pretrained=False, freeze_encoder=True)
    x = torch.randn(1, 3, 128, 128)
    loss = model(x).sum()
    loss.backward()

    for module in model._encoder_modules():
        for param in module.parameters():
            assert not param.requires_grad
            assert param.grad is None

    for module in model._decoder_modules():
        for param in module.parameters():
            assert param.requires_grad
            assert param.grad is not None

    model.unfreeze_encoder()
    for module in model._encoder_modules():
        for param in module.parameters():
            assert param.requires_grad

    groups = model.get_param_groups(encoder_lr=1e-4, decoder_lr=5e-4)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["lr"] == 5e-4


def test_wandb_logger_disabled() -> None:
    logger = WandbLogger(enabled=False)
    assert not logger.enabled
    assert logger.run is None
    logger.log_metrics({"train/loss": 0.5}, step=1)
    logger.log_best_val({"dice": 0.8}, step=1)
    logger.log_test_metrics({"loss": 0.2, "dice": 0.85})
    logger.finish()


@patch("src.utils.logger.wandb")
def test_wandb_logger_project_and_metric_namespaces(mock_wandb: MagicMock) -> None:
    mock_run = MagicMock()
    mock_run.summary = {}
    mock_wandb.init.return_value = mock_run

    logger = WandbLogger(
        entity="noct2306-uet",
        project="BTL_DL_2026",
        run_name="resnet34-unet-test",
        config={"epochs": 1},
        enabled=True,
    )
    mock_wandb.init.assert_called_once_with(
        entity="noct2306-uet",
        project="BTL_DL_2026",
        name="resnet34-unet-test",
        config={"epochs": 1},
        tags=[],
    )

    logger.log_best_val({"dice": 0.82, "loss": 0.18}, step=3)
    mock_wandb.log.assert_called_with(
        {"best_val/dice": 0.82, "best_val/loss": 0.18}, step=3
    )
    assert mock_run.summary["best_val/dice"] == 0.82

    test_metrics = {
        "loss": 0.2,
        "dice": 0.81,
        "iou": 0.70,
        "precision": 0.84,
        "recall": 0.79,
        "accuracy": 0.95,
    }
    logger.log_test_metrics(test_metrics)
    logged_test = mock_wandb.log.call_args.args[0]
    assert set(logged_test.keys()) == {
        "test/loss",
        "test/dice",
        "test/iou",
        "test/precision",
        "test/recall",
        "test/accuracy",
    }

    logger.finish()
    mock_wandb.finish.assert_called_once()


def test_trainer_smoke_with_synthetic_data(tmp_path: Path) -> None:
    model = ResNet34UNet(pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = BCEDiceLoss()
    logger = WandbLogger(enabled=False)

    imgs = torch.randn(4, 3, 64, 64)
    masks = (torch.rand(4, 1, 64, 64) > 0.5).float()
    loader = DataLoader(TensorDataset(imgs, masks), batch_size=2)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        logger=logger,
        checkpoint_dir=tmp_path,
        log_image_every_n_epochs=0,
    )
    best_ckpt = trainer.fit(train_loader=loader, val_loader=loader, epochs=1)
    assert best_ckpt.exists()


def test_trainer_early_stopping(tmp_path: Path) -> None:
    model = ResNet34UNet(pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = BCEDiceLoss()
    logger = WandbLogger(enabled=False)

    imgs = torch.randn(2, 3, 64, 64)
    masks = (torch.rand(2, 1, 64, 64) > 0.5).float()
    loader = DataLoader(TensorDataset(imgs, masks), batch_size=2)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        logger=logger,
        checkpoint_dir=tmp_path,
        log_image_every_n_epochs=0,
        early_stopping_patience=3,
    )

    fixed_val = {
        "loss": 0.5,
        "dice": 0.5,
        "iou": 0.33,
        "precision": 0.5,
        "recall": 0.5,
        "accuracy": 0.5,
    }
    trainer.evaluator.evaluate = MagicMock(return_value=(fixed_val, None))
    trainer.fit(train_loader=loader, val_loader=loader, epochs=10)

    # Epoch 1 sets best_val_dice=0.5, then 3 non-improving epochs (2, 3, 4) trigger early stop
    assert trainer.evaluator.evaluate.call_count == 4
