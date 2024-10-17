import json
from typing import Dict
from pathlib import Path
from datetime import timedelta
from enum import Enum
from json import JSONEncoder
import gin
import logging
import numpy as np
import torch

from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only
from sklearn.metrics import average_precision_score
from torch.nn import Module
from torch.optim import Optimizer, Adam, SGD, RAdam
from typing import Optional, Union
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, MultiStepLR, ExponentialLR


def save_config_file(log_dir):
    config_path = log_dir / "train_config.gin"
    with config_path.open("w") as f:
        f.write(gin.operative_config_str())


def create_optimizer(name: str, model: Module, lr: float, momentum: float = 0) -> Optimizer:
    """creates the specified optimizer with the given parameters

    Args:
        name (str): str name of optimizer
        model (Module): the model used for training
        lr (float): learning rate
        momentum (float): momentum (only for sgd optimizer)

    Raises:
        ValueError: thrown if optimizer name not known

    Returns:
        Optimizer: the model optimizer
    """
    name = name.lower()
    if name == "adam":
        return Adam(params=model.parameters(), lr=lr)
    elif name == "sgd":
        return SGD(params=model.parameters(), lr=lr, momentum=momentum)
    elif name == "radam":
        return RAdam(params=model.parameters(), lr=lr)
    else:
        raise ValueError(f"No optimizer with name {name} found!")


def create_scheduler(
    scheduler_name: Optional[str],
    optimizer: Optimizer,
    lr_factor: float,
    lr_steps: Optional[list],
    epochs: int,
) -> Union[_LRScheduler, None]:
    """creates a learning rate scheduler with the given parameters

    Args:
        scheduler_name (Optional[str]): str name of scheduler or None, in which case None will be returned
        optimizer (Optimizer): the learning optimizer
        lr_factor (float): the learning rate factor
        lr_steps (Optional[list]): learning rate steps for the scheduler to take (only supported for step scheduler)
        epochs (int): number of scheduler epochs (only supported for cosine scheduler)

    Raises:
        ValueError: thrown if step scheduler was chosen but no steps were passed
        ValueError: thrown if scheduler name not known and not None

    Returns:
        Union[_LRScheduler, None]: either the learning rate scheduler object or None if scheduler_name was None
    """
    if scheduler_name == "step":
        if not lr_steps:
            raise ValueError("step scheduler chosen but no lr steps passed!")
        return MultiStepLR(optimizer, lr_steps, lr_factor)
    elif scheduler_name == "exponential":
        return ExponentialLR(optimizer, lr_factor)
    elif scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, epochs)
    elif not scheduler_name:
        return None
    else:
        raise ValueError(f"no scheduler with name {scheduler_name} found!")


class JsonResultLoggingEncoder(JSONEncoder):
    """JSON converter for objects that are not serializable by default."""

    # Serializes foreign datatypes
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, tuple):
            if isinstance(obj)[0] is torch.Tensor or isinstance(obj)[0] is np.ndarray:
                return map(lambda item: item.tolist(), obj)
        if isinstance(obj, timedelta):
            return str(obj)
        return JSONEncoder.default(self, obj)


class Align(str, Enum):
    LEFT = "<"
    CENTER = "^"
    RIGHT = ">"


def log_table_row(
    cells: list,
    level: int = logging.INFO,
    widths: list[int] = None,
    header: list[str] = None,
    align: Align = Align.LEFT,
    highlight: bool = False,
):
    """Logs a table row.

    Args:
        cells: List of cells to log.
        level: Logging level.
        widths: List of widths for each cell.
        header: List of headers to calculate widths if widths not supplied.
        highlight: If set to true, highlight the row.
    """
    table_cells = cells
    if not widths and header:
        widths = [len(head) for head in header]
    if widths:
        table_cells = []
        for cell, width in zip(cells, widths):
            cell = str(cell)[:width]  # truncate cell if it is too long
            table_cells.append("{: {align}{width}}".format(cell, align=align.value, width=width))
    table_row = " | ".join([f"{cell}" for cell in table_cells])
    if highlight:
        table_row = f"\x1b[31;32m{table_row}\x1b[0m"
    logging.log(level, table_row)


class JSONMetricsLogger(Logger):
    def __init__(self, output_dir=None, **kwargs):
        super().__init__(**kwargs)
        if output_dir is None:
            output_dir = Path.cwd() / "metrics"
        logging.info(f"logging metrics to file: {str(output_dir.resolve())}")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self):
        return "json_metrics_logger"

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        old_metrics = {}
        stage_metrics = {
            "train": {"/".join(key.split("/")[1:]): value for key, value in metrics.items() if key.startswith("train/")},
            "val": {"/".join(key.split("/")[1:]): value for key, value in metrics.items() if key.startswith("val/")},
            "test": {"/".join(key.split("/")[1:]): value for key, value in metrics.items() if key.startswith("test/")},
        }
        for stage, metrics in stage_metrics.items():
            if metrics:
                output_file = self.output_dir / f"{stage}_metrics.json"
                old_metrics = {}
                if output_file.exists():
                    try:
                        with output_file.open("r") as f:
                            old_metrics = json.load(f)
                        logging.debug(f"updating {stage} metrics file...")
                    except json.decoder.JSONDecodeError:
                        logging.warning("could not decode json file, overwriting...")

                old_metrics.update(metrics)
                with output_file.open("w") as f:
                    json.dump(old_metrics, f, cls=JsonResultLoggingEncoder, indent=4)

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass


class scorer_wrapper:
    """
    Wrapper that flattens the binary classification input such that we can use a broader range of sklearn metrics.
    """

    def __init__(self, scorer=average_precision_score):
        self.scorer = scorer

    def __call__(self, y_true, y_pred):
        if len(np.unique(y_true)) <= 2 and y_pred.ndim > 1:
            y_pred_argmax = np.argmax(y_pred, axis=1)
            return self.scorer(y_true, y_pred_argmax)
        else:
            return self.scorer(y_true, y_pred)

    def __name__(self):
        return "scorer_wrapper"


# Source: https://github.com/ratschlab/tls
@gin.configurable("get_smoothed_labels")
def get_smoothed_labels(
    label, event, smoothing_fn=gin.REQUIRED, h_true=gin.REQUIRED, h_min=gin.REQUIRED, h_max=gin.REQUIRED, delta_h=12, gamma=0.1
):
    diffs = np.concatenate([np.zeros(1), event[1:] - event[:-1]], axis=-1)
    pos_event_change_full = np.where((diffs == 1) & (event == 1))[0]

    multihorizon = isinstance(h_true, list)
    if multihorizon:
        label_for_event = label[0]
        h_for_event = h_true[0]
    else:
        label_for_event = label
        h_for_event = h_true
    diffs_label = np.concatenate([np.zeros(1), label_for_event[1:] - label_for_event[:-1]], axis=-1)

    # Event that occurred after the end of the stay for M3B.
    # In that case event are equal to the number of hours after the end of stay when the event occured.
    pos_event_change_delayed = np.where((diffs >= 1) & (event > 1))[0]
    if len(pos_event_change_delayed) > 0:
        delays = event[pos_event_change_delayed] - 1
        pos_event_change_delayed += delays.astype(int)
        pos_event_change_full = np.sort(np.concatenate([pos_event_change_full, pos_event_change_delayed]))

    last_know_label = label_for_event[np.where(label_for_event != -1)][-1]
    last_know_idx = np.where(label_for_event == last_know_label)[0][-1]

    # Need to handle the case where the ts was truncatenated at 2016 for HiB
    if ((last_know_label == 1) and (len(pos_event_change_full) == 0)) or (
        (last_know_label == 1) and (last_know_idx >= pos_event_change_full[-1])
    ):
        last_know_event = 0
        if len(pos_event_change_full) > 0:
            last_know_event = pos_event_change_full[-1]

        last_known_stable = 0
        known_stable = np.where(label_for_event == 0)[0]
        if len(known_stable) > 0:
            last_known_stable = known_stable[-1]

        pos_change = np.where((diffs_label >= 1) & (label_for_event == 1))[0]
        last_pos_change = pos_change[np.where(pos_change > max(last_know_event, last_known_stable))][0]
        pos_event_change_full = np.concatenate([pos_event_change_full, [last_pos_change + h_for_event]])

    # No event case
    if len(pos_event_change_full) == 0:
        pos_event_change_full = np.array([np.inf])

    time_array = np.arange(len(label))
    dist = pos_event_change_full.reshape(-1, 1) - time_array
    dte = np.where(dist > 0, dist, np.inf).min(axis=0)
    if multihorizon:
        smoothed_labels = []
        for k in range(label.shape[-1]):
            smoothed_labels.append(
                np.array(
                    list(
                        map(
                            lambda x: smoothing_fn(
                                x, h_true=h_true[k], h_min=h_min[k], h_max=h_max[k], delta_h=delta_h, gamma=gamma
                            ),
                            dte,
                        )
                    )
                )
            )
        return np.stack(smoothed_labels, axis=-1)
    else:
        return np.array(
            list(map(lambda x: smoothing_fn(x, h_true=h_true, h_min=h_min, h_max=h_max, delta_h=delta_h, gamma=gamma), dte))
        )
