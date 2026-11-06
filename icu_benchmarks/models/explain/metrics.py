"""Explanation-quality metrics for variable-level attributions.

All metrics work on one attribution vector per explained hour (one value per variable) and remove a variable by setting
its whole history, value and missingness indicator together, to the "not measured" baseline.
"""

import numpy as np


def sparsity(attributions: np.ndarray) -> np.ndarray:
    """Gini index of the absolute attributions of each explained hour (Chalasani et al., 2020).

    Args:
        attributions: Attributions of shape (hours, variables).

    Returns:
        One value per explained hour: 0 when every variable gets the same share, close to 1 when one variable gets all of
        it. NaN for an all-zero attribution.
    """
    values = np.sort(np.abs(attributions), axis=1)
    n = values.shape[1]
    total = values.sum(axis=1)
    weights = 2 * np.arange(1, n + 1) - n - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        gini = (values * weights).sum(axis=1) / (n * total)
    return np.where(total > 0, gini, np.nan)


def faithfulness_correlation(attributions: np.ndarray, subsets: np.ndarray, output_drops: np.ndarray) -> np.ndarray:
    """Pearson correlation between the attribution of random variable subsets and the output drop when they are removed.

    Bhatt et al. (2020): a faithful attribution assigns more to subsets whose removal changes the output more.

    Args:
        attributions: Attributions of shape (hours, variables).
        subsets: Boolean masks of the removed variables, shape (hours, subsets, variables).
        output_drops: Model output minus the output without the subset, shape (hours, subsets).

    Returns:
        One correlation per explained hour; NaN where the attributions or the drops do not vary across the subsets.
    """
    sums = np.einsum("hv,hsv->hs", attributions, subsets.astype(attributions.dtype))
    sums = sums - sums.mean(axis=1, keepdims=True)
    drops = output_drops - output_drops.mean(axis=1, keepdims=True)
    denominator = np.sqrt((sums**2).sum(axis=1) * (drops**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = (sums * drops).sum(axis=1) / denominator
    return np.where(denominator > 1e-12, correlation, np.nan)


def deletion_aopc(output: np.ndarray, deleted_outputs: np.ndarray) -> np.ndarray:
    """Area over the perturbation curve (Samek et al., 2017) for removing the k highest-attributed variables, k = 1..K.

    Args:
        output: Model output of each explained hour, shape (hours,).
        deleted_outputs: Output after removing the top k variables, shape (hours, K).

    Returns:
        The mean output drop over k for each explained hour. Higher means the attribution found the variables that drive
        the output.
    """
    return (output[:, None] - deleted_outputs).mean(axis=1)
