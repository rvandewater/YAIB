import torch
from typing import Callable
import numpy as np
from ignite.metrics import EpochMetric
from numpy import ndarray
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, confusion_matrix as sk_confusion_matrix
from sklearn.calibration import calibration_curve
from scipy.spatial.distance import jensenshannon
from torchmetrics.classification import BinaryFairness

""""
This file contains custom metrics that can be added to YAIB.
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


class BalancedAccuracy(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(BalancedAccuracy, self).__init__(
            self.balanced_accuracy_compute, output_transform=output_transform, check_compute_fn=check_compute_fn
        )

        def balanced_accuracy_compute(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
            y_true = y_targets.numpy()
            y_pred = np.argmax(y_preds.numpy(), axis=-1)
            return balanced_accuracy_score(y_true, y_pred)


class CalibrationCurve(EpochMetric):
    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(CalibrationCurve, self).__init__(
            self.ece_curve_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn
        )

        def ece_curve_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor, n_bins=10) -> float:
            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            return calibration_curve(y_true, y_pred, n_bins=n_bins)


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

        def mae_with_invert_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor, invert_fn=Callable) -> float:
            y_true = invert_fn(y_targets.numpy().reshape(-1, 1))[:, 0]
            y_pred = invert_fn(y_preds.numpy().reshape(-1, 1))[:, 0]
            return mean_absolute_error(y_true, y_pred)


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

        def JSD_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
            return jensenshannon(abs(y_preds).flatten(), abs(y_targets).flatten()) ** 2


class TorchMetricsWrapper:
    metric = None

    def __init__(self, metric) -> None:
        self.metric = metric

    def update(self, output_tuple) -> None:
        self.metric.update(output_tuple[0], output_tuple[1])

    def compute(self) -> None:
        return self.metric.compute()

    def reset(self) -> None:
        return self.metric.reset()


class BinaryFairnessWrapper(BinaryFairness):
    """
    This class is a wrapper for the BinaryFairness metric from TorchMetrics.
    """

    group_name = None

    def __init__(self, group_name="sex", *args, **kwargs) -> None:
        self.group_name = group_name
        super().__init__(*args, **kwargs)

    def update(self, preds, target, data, feature_names) -> None:
        """ " Standard metric update function"""
        groups = data[:, :, feature_names.index(self.group_name)]
        group_per_id = groups[:, 0]
        return super().update(preds=preds.cpu(), target=target.cpu(), groups=group_per_id.long().cpu())

    def feature_helper(self, trainer, step_prefix):
        """Helper function to get the feature names from the trainer"""
        if step_prefix == "train":
            feature_names = trainer.train_dataloader.dataset.features
        elif step_prefix == "val":
            feature_names = trainer.train_dataloader.dataset.features
        else:
            feature_names = trainer.test_dataloaders.dataset.features
        return feature_names


def confusion_matrix(y_true: ndarray, y_pred: ndarray, normalize=False) -> torch.tensor:
    y_pred = np.rint(y_pred).astype(int)
    confusion = sk_confusion_matrix(y_true, y_pred)
    if normalize:
        confusion = confusion / confusion.sum()
    confusion_dict = {}
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            confusion_dict[f"class_{i}_pred_{j}"] = confusion[i][j]
    return confusion_dict
