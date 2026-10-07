"""Single-target affine normalization shared by preprocessing, ML and DL."""

import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler

from .constants import DataSegment, DataSplit


def target_scaling_policy(policy, lower, upper):
    """Resolve legacy bounds without silently accepting partial or conflicting settings."""
    has_bounds = lower is not None or upper is not None
    if policy is None:
        policy = "fixed" if has_bounds else "minmax"
        if has_bounds:
            warnings.warn("Set target_scaling='fixed' when using outcome bounds.", DeprecationWarning, stacklevel=2)
    if policy not in ("minmax", "fixed", "none"):
        raise ValueError(f"Unknown target scaling policy: {policy}")
    if policy == "fixed":
        if lower is None or upper is None or not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("Fixed target scaling requires finite outcome_min < outcome_max.")
    elif has_bounds:
        raise ValueError("Outcome bounds are only valid with target_scaling='fixed'.")
    return policy


@dataclass(frozen=True)
class TargetTransform:
    """z = y * scale + offset. Scalar arithmetic preserves NumPy/Torch shape and device."""

    label: str
    policy: str
    scale: float
    offset: float
    minimum: float
    maximum: float
    version: int = 1
    metric_space: str = "original_target_units"

    def __post_init__(self):
        if self.version != 1 or self.metric_space != "original_target_units":
            raise ValueError("Unsupported target transform version or metric space.")
        if not isinstance(self.label, str) or not self.label or self.policy not in ("minmax", "fixed", "none"):
            raise ValueError("Invalid target transform label or policy.")
        if not np.isfinite([self.scale, self.offset, self.minimum, self.maximum]).all() or self.scale <= 0:
            raise ValueError("Target transform must have finite coefficients and positive scale.")
        if self.minimum > self.maximum:
            raise ValueError("Invalid target transform range.")

    @classmethod
    def fit(cls, values, label, policy="minmax", lower=None, upper=None):
        policy = target_scaling_policy(policy, lower, upper)
        values = np.asarray(values, dtype=float)
        if np.isinf(values).any():
            raise ValueError("Infinite regression targets are not supported.")
        observed = values[np.isfinite(values)]
        if not observed.size:
            raise ValueError("No observed training targets to fit.")
        if policy == "fixed":
            observed = np.array([lower, upper])
        scaler = MinMaxScaler(clip=False).fit(observed.reshape(-1, 1))
        if scaler.data_range_[0] == 0:
            logging.warning("Training targets are constant; using sklearn's constant-column scaling.")
        return cls(
            label=label,
            policy=policy,
            scale=float(scaler.scale_[0]) if policy != "none" else 1.0,
            offset=float(scaler.min_[0]) if policy != "none" else 0.0,
            minimum=float(scaler.data_min_[0]),
            maximum=float(scaler.data_max_[0]),
        )

    def transform(self, values):
        return values * self.scale + self.offset

    def inverse_transform(self, values):
        return (values - self.offset) / self.scale

    def save(self, directory):
        (Path(directory) / "target_transform.json").write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, directory):
        path = Path(directory) / "target_transform.json"
        if not path.exists():
            raise ValueError(f"Missing {path}; regression evaluation requires the training transform. Retrain legacy models.")
        return cls(**json.loads(path.read_text()))


def transform_outcomes(data, label, policy, lower, upper, target_transform=None):
    """Fit once before padding; reuse on all splits (or use a saved source transform)."""
    if isinstance(label, list) and len(label) == 1:
        label = label[0]
    if not isinstance(label, str):
        raise ValueError("Target scaling currently supports one selected regression label.")
    if target_transform is None:
        target_transform = TargetTransform.fit(
            data[DataSplit.train][DataSegment.outcome][label].to_numpy(), label, policy, lower, upper
        )
    if target_transform.label != label:
        raise ValueError(f"Target label {label!r} does not match saved label {target_transform.label!r}.")
    for split in data.values():
        outcome = split[DataSegment.outcome]
        if np.isinf(outcome[label].to_numpy()).any():
            raise ValueError("Infinite regression targets are not supported.")
        if isinstance(outcome, pl.DataFrame):
            split[DataSegment.outcome] = outcome.with_columns(target_transform.transform(pl.col(label)))
        else:
            split[DataSegment.outcome] = outcome.assign(**{label: target_transform.transform(outcome[label])})
    return target_transform
