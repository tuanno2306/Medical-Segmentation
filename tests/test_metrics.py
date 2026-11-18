import unittest

import torch

from src.engine.losses import build_loss
from src.metrics.segmentation_metrics import segmentation_scores


class SegmentationMetricsTest(unittest.TestCase):
    def test_per_image_scores_and_empty_masks(self):
        targets = torch.tensor([[[[1., 1.], [0., 0.]]],
                                [[[0., 0.], [0., 0.]]],
                                [[[1., 0.], [0., 0.]]]])
        logits = torch.tensor([[[[10., -10.], [10., -10.]]],
                               [[[-10., -10.], [-10., -10.]]],
                               [[[-10., -10.], [-10., -10.]]]])
        dice, iou = segmentation_scores(logits, targets)
        torch.testing.assert_close(dice, torch.tensor([0.5, 1., 0.]))
        torch.testing.assert_close(iou, torch.tensor([1 / 3, 1., 0.]))

    def test_loss_produces_gradients(self):
        logits = torch.zeros((1, 1, 2, 2), requires_grad=True)
        loss = build_loss()(logits, torch.ones_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_combined_loss_prefers_correct_predictions(self):
        target = torch.tensor([[[[1., 0.], [1., 0.]]]])
        loss = build_loss()
        correct = torch.tensor([[[[8., -8.], [8., -8.]]]])
        self.assertLess(loss(correct, target), loss(-correct, target))
        empty = torch.zeros_like(target)
        self.assertTrue(torch.isfinite(loss(-correct, empty)))

    def test_bce_baseline_remains_available(self):
        self.assertIsInstance(build_loss("bce"), torch.nn.BCEWithLogitsLoss)


if __name__ == "__main__":
    unittest.main()
