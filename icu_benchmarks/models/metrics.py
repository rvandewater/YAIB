import torch
from typing import Callable
import numpy as np
from ignite.metrics import EpochMetric
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.calibration import calibration_curve
from scipy.spatial.distance import jensenshannon


""""
This file contains metrics that are not available in ignite.metrics. Specifically, it adds transformation capabilities to some
metrics.
"""


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def balanced_accuracy_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    y_true = y_targets.numpy()
    y_pred = np.argmax(y_preds.numpy(), axis=-1)
    return balanced_accuracy_score(y_true, y_pred)


def ece_curve_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    return calibration_curve(y_true, y_pred, n_bins=10)


def mae_with_invert_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor, invert_fn=Callable) -> float:
    y_true = invert_fn(y_targets.numpy().reshape(-1, 1))[:, 0]
    y_pred = invert_fn(y_preds.numpy().reshape(-1, 1))[:, 0]
    return mean_absolute_error(y_true, y_pred)


def JSD_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
    return jensenshannon(abs(y_preds).flatten(), abs(y_targets).flatten()) ** 2


class BalancedAccuracy(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(BalancedAccuracy, self).__init__(
            balanced_accuracy_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
        )


class CalibrationCurve(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(CalibrationCurve, self).__init__(
            ece_curve_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
        )


class MAE(EpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        invert_transform: Callable = lambda x: x,
    ) -> None:
        super(MAE, self).__init__(
            lambda x, y: mae_with_invert_compute_fn(x, y, invert_transform),
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
        )


class JSD(EpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
    ) -> None:
        super(JSD, self).__init__(
            lambda x, y: JSD_fn(x, y),
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
        )
