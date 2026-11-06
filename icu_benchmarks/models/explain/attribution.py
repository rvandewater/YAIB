"""Attributions of deep learning predictions to the input variables, for every labeled hour.

A variable is a value column together with its ``MissingIndicator_<name>`` column, so that removing a variable means
"not measured": its value is set to 0 (the training mean, after YAIB's standardisation) and its indicator to 1. All
methods return one attribution per explained hour and variable, summed over the hours before it and over the variable's
columns.

The explained output is the log-odds of the positive class for binary classification (logit 1 minus logit 0) and the
predicted value for regression, at the hour being explained.
"""

from typing import Iterator

import numpy as np
import torch
from torch import Tensor

INDICATOR_PREFIX = "MissingIndicator_"


def variable_groups(columns: list[str]) -> tuple[list[str], np.ndarray]:
    """Groups each value column with its missingness indicator.

    Args:
        columns: The model's input columns, in input order.

    Returns:
        The variable names, and for each column the index of its variable. An indicator without a matching value column
        is its own variable.
    """
    names: list[str] = []
    index: dict[str, int] = {}
    group_of = []
    present = set(columns)
    for column in columns:
        base = column[len(INDICATOR_PREFIX) :] if column.startswith(INDICATOR_PREFIX) else column
        variable = base if base in present else column
        if variable not in index:
            index[variable] = len(names)
            names.append(variable)
        group_of.append(index[variable])
    return names, np.asarray(group_of)


def missing_baseline(columns: list[str]) -> Tensor:
    """The input of a variable that was never measured: 0 for values, 1 for missingness indicators."""
    return torch.tensor([1.0 if column.startswith(INDICATOR_PREFIX) else 0.0 for column in columns])


def explained_output(output: Tensor | tuple) -> Tensor:
    """Reduces the model output of shape (..., classes) to the explained scalar: binary log-odds or the regression value."""
    if isinstance(output, tuple):  # Models with an auxiliary loss return (prediction, aux_loss)
        output = output[0]
    if output.shape[-1] == 2:
        return output[..., 1] - output[..., 0]
    if output.shape[-1] == 1:
        return output[..., 0]
    raise NotImplementedError(f"Explanations support binary classification and regression, not {output.shape[-1]} classes.")


def output_at_hours(model: torch.nn.Module, x: Tensor, hours: Tensor) -> Tensor:
    """The explained output of each sample at its own hour, shape (samples,). Differentiable, for Captum."""
    output = explained_output(model(x))
    return output[torch.arange(len(hours), device=output.device), hours]


@torch.no_grad()
def output_all_hours(model: torch.nn.Module, x: Tensor, batch_size: int) -> Tensor:
    """The explained output at every hour, shape (samples, hours), computed in batches without gradients."""
    device = next(model.parameters()).device
    outputs = [explained_output(model(chunk.to(device))).cpu() for chunk in x.split(batch_size)]
    return torch.cat(outputs)


def remove_variables(x: Tensor, removed: Tensor, group_of: Tensor, baseline: Tensor) -> Tensor:
    """Sets the whole history of the removed variables to the baseline.

    Args:
        x: Inputs of shape (..., hours, columns).
        removed: Boolean masks of shape (..., variables), broadcastable to the leading dimensions of x.
        group_of: The variable index of each column, shape (columns,).
        baseline: The baseline value of each column, shape (columns,).
    """
    columns = removed[..., group_of].unsqueeze(-2)  # (..., 1, columns)
    return torch.where(columns, baseline, x)


def chunks(n: int, size: int) -> Iterator[slice]:
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def integrated_gradients(
    model: torch.nn.Module,
    stay_inputs: Tensor,
    stays: Tensor,
    hours: Tensor,
    baseline: Tensor,
    group_of: Tensor,
    n_variables: int,
    steps: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrated Gradients (Sundararajan et al., 2017) from the "not measured" baseline, with Captum.

    Args:
        model: The model, in eval mode.
        stay_inputs: The inputs of the explained stays, shape (stays, hours, columns).
        stays: The stay index of each explained hour.
        hours: The hour to explain, for each explained hour.
        baseline: The baseline value of each column.
        group_of: The variable index of each column.
        n_variables: The number of variables.
        steps: The number of integration steps.
        batch_size: The number of explained hours per Captum call.

    Returns:
        The attributions of shape (explained hours, variables), and the absolute convergence delta of each explained
        hour (how far the attributions are from summing to the output difference).
    """
    from captum.attr import IntegratedGradients

    device = next(model.parameters()).device
    explainer = IntegratedGradients(lambda inputs, at: output_at_hours(model, inputs, at))
    group_one_hot = torch.nn.functional.one_hot(group_of, n_variables).float().to(device)  # (columns, variables)
    attributions, deltas = [], []
    for part in chunks(len(stays), batch_size):
        inputs = stay_inputs[stays[part]].to(device)
        attribution, delta = explainer.attribute(
            inputs,
            baselines=baseline.to(device).expand_as(inputs),
            additional_forward_args=(hours[part].to(device),),
            n_steps=steps,
            internal_batch_size=batch_size,
            return_convergence_delta=True,
        )
        attributions.append((attribution.sum(dim=1) @ group_one_hot).detach().cpu())
        deltas.append(delta.abs().detach().cpu())
    return torch.cat(attributions).numpy(), torch.cat(deltas).numpy()


def feature_ablation(
    stay_inputs: Tensor,
    stay_outputs: Tensor,
    explained: list[tuple[int, int]],
    model: torch.nn.Module,
    baseline: Tensor,
    group_of: Tensor,
    n_variables: int,
    batch_size: int,
) -> np.ndarray:
    """Occlusion of each variable's whole history: the output drop when the variable is set to "not measured".

    One forward pass per stay and variable gives the drop at every hour of the stay at once.

    Args:
        stay_inputs: The inputs of the explained stays, shape (stays, hours, columns).
        stay_outputs: Their explained outputs at every hour, shape (stays, hours).
        explained: The (stay index, hour) of each explained hour.
        model: The model, in eval mode.
        baseline: The baseline value of each column.
        group_of: The variable index of each column.
        n_variables: The number of variables.
        batch_size: The number of inputs per forward pass.

    Returns:
        The attributions of shape (explained hours, variables).
    """
    removed = torch.eye(n_variables, dtype=torch.bool)  # one variable per row
    drops = {}
    for stay in sorted({stay for stay, _ in explained}):
        ablated = remove_variables(stay_inputs[stay].unsqueeze(0), removed, group_of, baseline)  # (variables, hours, cols)
        drops[stay] = stay_outputs[stay].unsqueeze(0) - output_all_hours(model, ablated, batch_size)  # (variables, hours)
    return np.stack([drops[stay][:, hour].numpy() for stay, hour in explained])


def random_attribution(n_explained: int, n_variables: int, rng: np.random.Generator) -> np.ndarray:
    """Random attributions, the reference that every metric should beat."""
    return rng.standard_normal((n_explained, n_variables))
