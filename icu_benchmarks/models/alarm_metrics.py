import numpy as np
from numpy import ndarray
import torch
from sklearn.metrics import precision_score

# def convert_to_alarm(y_true: ndarray, y_pred: ndarray, normalize=False) -> torch.tensor:
#     y_pred = np.rint(y_pred).astype(int)
#     confusion = sk_confusion_matrix(y_true, y_pred)
#
#     return y_pred


def convert_to_alarm(ground_truth, predictions, grace_horizon=12, silencing_length=6):
    """Convert the predictions to alarm predictions based on the grace horizon and silencing length."""
    # Silence positive predictions
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and predictions must have the same length.")
    if len(ground_truth) < grace_horizon:
        raise ValueError("Length of the ground truth must be greater than the grace horizon.")
    if not isinstance(ground_truth, np.ndarray):
        ground_truth = np.array(ground_truth)
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    threshold = 0.5
    predictions = np.where(predictions >= threshold, 1, 0)
    silenced_predictions = silence_positives(ground_truth, predictions, grace_horizon, silencing_length)
    # Fill gaps in the predictions
    filled_predictions = fill_gaps(silenced_predictions, grace_horizon)
    return filled_predictions, ground_truth


def silence_positives(ground_truth, predictions, grace_horizon=12, silencing_length=6):
    """Silence positive predictions in the predictions array based on the grace horizon and silencing length."""
    # Find all positive indices in the rounded predictions
    positive_indices = np.where(predictions == 1)[0]
    positive_indices = positive_indices[positive_indices < len(ground_truth) - grace_horizon]
    # print(positive_indices)
    # Create the silence array
    silence_array = np.ones_like(ground_truth)

    while len(positive_indices) > 0:
        positive_index = positive_indices[0]
        if silence_array[positive_indices[0]] == 0:
            # print(f"Already silenced: {positive_index}")
            positive_indices = positive_indices[1:]
            continue
        silence_array[positive_index + 1 : positive_index + silencing_length] = 0
        positive_indices = positive_indices[1:]
    # print(predictions)
    # print(silence_array)
    silenced_predictions = predictions * silence_array
    # print(silenced_predictions)
    return silenced_predictions


def fill_gaps(predictions, ground_truth, grace_horizon=12):
    """Fill gaps in the predictions by taking the maximum value between ground truth and predictions."""
    if grace_horizon > len(predictions):
        grace_horizon = len(predictions)
    # Take the last grace_horizon values
    grace_values = predictions[-grace_horizon:]
    # print(grace_values)
    # Find the first occurrence of 1 in the grace_values
    first_one_index = np.where(grace_values == 1)
    if first_one_index[0].size > 0:
        # Make all subsequent values after the first 1 positive
        first_one_index = first_one_index[0][0]
        grace_values[first_one_index:] = 1
    # Update the original prediction array
    # print(grace_values)
    predictions[-grace_horizon:] = grace_values
    ground_truth[-grace_horizon:] = np.maximum(predictions[-grace_horizon:], ground_truth[-grace_horizon:])
    return predictions, ground_truth
