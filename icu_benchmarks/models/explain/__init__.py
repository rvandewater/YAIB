"""Feature attributions and explanation-quality metrics for the deep learning prediction models.

``train_common`` calls :func:`explain_predictions` after testing a deep learning model when ``--explain-features`` is
set. It explains every labeled hour of a sample of test stays and saves, per fold:

- ``test_attributions.parquet``: one row per method and explained hour, with one column per variable.
- ``test_feature_importance.parquet``: the mean absolute attribution of each variable, per method.
- ``test_explanation_metrics.json``: faithfulness, deletion and sparsity of each method, averaged over the explained
  hours. ``aggregate_results`` averages them over the folds like the other test metrics.

Integrated Gradients needs the optional ``captum`` package: ``uv sync --extra explain``.
"""

import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import gin
import numpy as np
import polars as pl
import torch

from icu_benchmarks.models.explain.attribution import (
    chunks,
    feature_ablation,
    integrated_gradients,
    missing_baseline,
    output_all_hours,
    random_attribution,
    remove_variables,
    variable_groups,
)
from icu_benchmarks.models.explain.metrics import deletion_aopc, faithfulness_correlation, sparsity

METHODS = ("IntegratedGradients", "FeatureAblation", "Random")
METRICS = ("faithfulness", "deletion", "sparsity")


def require_captum() -> None:
    """Fails early, with the install command, when the optional captum package is missing."""
    try:
        import captum  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Explaining deep learning models needs the optional captum package. Install it with `uv sync --extra explain`."
        ) from error


def stay_times(dataset) -> dict:
    """The sequence value (e.g. the hour) of each input row, per stay, in the dataset's input order."""
    group, sequence = dataset.vars["GROUP"], dataset.vars["SEQUENCE"]
    rows = dataset.row_indicators
    if not isinstance(rows, pl.DataFrame) or sequence not in rows.columns:
        return {}
    by_stay = rows.sort([group, sequence]).group_by(group, maintain_order=True).agg(pl.col(sequence))
    return dict(by_stay.iter_rows())


@gin.configurable("explain_predictions")
def explain_predictions(
    model,
    dataset,
    log_dir: Path,
    methods: tuple[str, ...] = METHODS,
    metrics: tuple[str, ...] = METRICS,
    max_stays: int = 100,
    ig_steps: int = 32,
    batch_size: int = 64,
    faithfulness_subsets: int = 20,
    faithfulness_subset_size: float = 0.1,
    deletion_steps: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Explains every labeled hour of a sample of test stays and saves the attributions and their quality metrics.

    Args:
        model: A trained deep learning prediction model (binary classification or regression).
        dataset: The test dataset (``PredictionPolarsDataset``).
        log_dir: The fold directory to save the results in.
        methods: Attribution methods: ``IntegratedGradients``, ``FeatureAblation`` and/or ``Random`` (a reference).
        metrics: Quality metrics: ``faithfulness``, ``deletion`` and/or ``sparsity``.
        max_stays: Number of test stays to explain, drawn at random; every labeled hour of each is explained.
        ig_steps: Integration steps of Integrated Gradients.
        batch_size: Number of inputs per forward pass.
        faithfulness_subsets: Number of random variable subsets per stay for the faithfulness correlation.
        faithfulness_subset_size: Share of the variables in each subset.
        deletion_steps: Number of top variables removed one by one for the deletion curve.
        seed: Seed for drawing the stays, the subsets and the random reference.

    Returns:
        The explanation metrics, keyed ``explain/<method>/<metric>``.
    """
    unknown = set(methods) - set(METHODS) or set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f"Unknown explanation methods or metrics: {sorted(unknown)}.")
    if "IntegratedGradients" in methods:
        require_captum()
    if not hasattr(dataset, "get_input_names"):
        raise NotImplementedError("Explanations need the Polars prediction dataset.")
    start = time.time()
    rng = np.random.default_rng(seed)

    columns = dataset.get_input_names()
    names, group_of = variable_groups(columns)
    n_variables = len(names)
    group_of = torch.as_tensor(group_of)
    baseline = missing_baseline(columns)

    selected = np.sort(rng.choice(len(dataset), size=min(max_stays, len(dataset)), replace=False))
    items = [dataset[int(index)] for index in selected]
    stay_inputs = torch.stack([item[0] for item in items]).float()
    labeled = torch.stack([item[2] for item in items]).bool()
    if stay_inputs.shape[-1] != len(columns):
        raise ValueError(f"The inputs have {stay_inputs.shape[-1]} columns, but the dataset names {len(columns)}.")
    stays, hours = (index.long() for index in torch.nonzero(labeled, as_tuple=True))
    if len(stays) == 0:
        logging.warning("No labeled hours in the sampled test stays; skipping explanations.")
        return {}
    explained = list(zip(stays.tolist(), hours.tolist()))
    logging.info(f"Explaining {len(explained)} labeled hours of {len(selected)} test stays with {', '.join(methods)}.")

    was_training = model.training
    model.eval()
    # cuDNN implements RNN backward passes only in training mode; the plain CUDA kernels work in eval mode too
    cudnn_off = torch.backends.cudnn.flags(enabled=False) if torch.cuda.is_available() else nullcontext()
    try:
        with torch.enable_grad(), cudnn_off:
            stay_outputs = output_all_hours(model, stay_inputs, batch_size)  # (stays, hours)
            output = stay_outputs[stays, hours].numpy()
            attributions, results = {}, {}
            for method in methods:
                if method == "IntegratedGradients":
                    attributions[method], deltas = integrated_gradients(
                        model, stay_inputs, stays, hours, baseline, group_of, n_variables, ig_steps, batch_size
                    )
                    results[f"explain/{method}/convergence_delta"] = float(np.mean(deltas))
                elif method == "FeatureAblation":
                    attributions[method] = feature_ablation(
                        stay_inputs, stay_outputs, explained, model, baseline, group_of, n_variables, batch_size
                    )
                else:
                    attributions[method] = random_attribution(len(explained), n_variables, rng)

            if "faithfulness" in metrics:
                # The same subsets for every method, so that their scores are comparable
                size = max(1, round(faithfulness_subset_size * n_variables))
                stay_subsets = np.zeros((len(selected), faithfulness_subsets, n_variables), dtype=bool)
                for stay in range(len(selected)):
                    for subset in range(faithfulness_subsets):
                        stay_subsets[stay, subset, rng.choice(n_variables, size=size, replace=False)] = True
                subset_outputs = torch.stack(
                    [
                        output_all_hours(
                            model,
                            remove_variables(stay_inputs[stay].unsqueeze(0), torch.as_tensor(subsets), group_of, baseline),
                            batch_size,
                        )
                        for stay, subsets in enumerate(stay_subsets)
                    ]
                )  # (stays, subsets, hours)
                drops = output[:, None] - subset_outputs[stays, :, hours].numpy()
                for method, attribution in attributions.items():
                    scores = faithfulness_correlation(attribution, stay_subsets[stays.numpy()], drops)
                    results[f"explain/{method}/faithfulness"] = (
                        float(np.nanmean(scores)) if np.isfinite(scores).any() else None
                    )

            if "deletion" in metrics:
                steps = min(deletion_steps, n_variables)
                for method, attribution in attributions.items():
                    top = torch.as_tensor(np.argsort(-attribution, axis=1)[:, :steps])  # (explained, steps)
                    # Step k removes the top k + 1 variables
                    removed = torch.zeros(len(explained), steps, n_variables, dtype=torch.bool)
                    rows = torch.arange(len(explained))
                    for k in range(steps):
                        if k:
                            removed[:, k] = removed[:, k - 1]
                        removed[rows, k, top[:, k]] = True
                    deleted = []
                    for part in chunks(len(explained), max(1, batch_size // steps)):
                        x = remove_variables(stay_inputs[stays[part]].unsqueeze(1), removed[part], group_of, baseline)
                        out = output_all_hours(model, x.flatten(0, 1), batch_size).view(len(x), steps, -1)
                        deleted.append(out[torch.arange(len(x)), :, hours[part]])
                    scores = deletion_aopc(output, torch.cat(deleted).numpy())
                    results[f"explain/{method}/deletion"] = float(np.mean(scores)) / scale(model)

            if "sparsity" in metrics:
                for method, attribution in attributions.items():
                    scores = sparsity(attribution)
                    results[f"explain/{method}/sparsity"] = float(np.nanmean(scores)) if np.isfinite(scores).any() else None
    finally:
        model.train(was_training)

    save_explanations(log_dir, dataset, selected, explained, names, attributions, scale(model), results)
    logging.info(f"Explanations saved to {log_dir} in {time.time() - start:.1f} s.")
    return results


def scale(model) -> float:
    """The factor from the model's output space to native target units: regression models predict scaled targets."""
    transform = getattr(model, "target_transform", None)
    return transform.scale if transform is not None else 1.0


def save_explanations(log_dir, dataset, selected, explained, names, attributions, output_scale, results) -> None:
    """Writes the attributions (in native units for regression), the feature importances and the metrics."""
    group, sequence = dataset.vars["GROUP"], dataset.vars["SEQUENCE"]
    stay_ids = dataset.get_stay_ids()
    times = stay_times(dataset)
    ids = [stay_ids[int(selected[stay])] for stay, _ in explained]
    hour_values = [times[stay_id][hour] if stay_id in times else hour for stay_id, (_, hour) in zip(ids, explained)]
    frames, importance = [], []
    for method, attribution in attributions.items():
        native = (attribution / output_scale).astype(np.float32)
        frame = pl.DataFrame(native, schema=names, orient="row")
        frames.append(
            frame.with_columns(pl.Series(group, ids), pl.Series(sequence, hour_values), pl.lit(method).alias("method"))
        )
        importance.append(
            pl.DataFrame(
                {
                    "method": method,
                    "variable": names,
                    "mean_abs_attribution": np.abs(native).mean(axis=0),
                    "mean_attribution": native.mean(axis=0),
                }
            )
        )
    attributions_frame = pl.concat(frames)
    attributions_frame.select(group, sequence, "method", *names).write_parquet(log_dir / "test_attributions.parquet")
    pl.concat(importance).write_parquet(log_dir / "test_feature_importance.parquet")
    with open(log_dir / "test_explanation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
