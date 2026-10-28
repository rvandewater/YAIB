"""The prediction datasets must pad, mask and align stays the same way for the DL and ML models."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch

from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.loader import ImputationPredictionDataset, PredictionPandasDataset, PredictionPolarsDataset

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x", "y"], "STATIC": []}


def splits(hourly: bool, frame=pl.DataFrame) -> dict:
    """Stay 1 has three hours, stay 2 has two hours; features equal stay_id * 10 + time."""
    stay_id, time = [1, 1, 1, 2, 2], [0, 1, 2, 0, 1]
    features = frame(
        {"stay_id": stay_id, "time": time, "y": [s * 10.0 + t for s, t in zip(stay_id, time)], "x": [1.0, 2, 3, 4, 5]}
    )
    if hourly:
        outcome = frame({"stay_id": stay_id, "time": time, "label": [0.0, np.nan, 1, 0, 0]})
    else:
        outcome = frame({"stay_id": [1, 2], "label": [1.0, 0.0]})
    return {Split.train: {Segment.features: features, Segment.outcome: outcome}}


@pytest.mark.parametrize("ram_cache", [True, False])
def test_polars_items_are_padded_and_masked(ram_cache):
    dataset = PredictionPolarsDataset(splits(hourly=True), vars=VARS, ram_cache=ram_cache)
    assert len(dataset) == 2 and dataset.maxlen == 3
    # Feature columns are sorted alphabetically, the group column is not a feature
    assert dataset.get_feature_names() == ["x", "y"]
    data, labels, mask = dataset[1]
    torch.testing.assert_close(data, torch.tensor([[4.0, 20], [5, 21], [0, 0]]))
    torch.testing.assert_close(labels, torch.tensor([0.0, 0, 0]))
    assert mask.tolist() == [True, True, False]
    # A missing hourly label is masked out and marked -1
    _, labels, mask = dataset[0]
    assert labels.tolist() == [0, -1, 1] and mask.tolist() == [True, False, True]


def test_polars_per_stay_label_is_aligned_with_last_hour():
    dataset = PredictionPolarsDataset(splits(hourly=False), vars=VARS)
    _, labels, mask = dataset[0]
    assert labels.tolist() == [-1, -1, 1] and mask.tolist() == [False, False, True]
    rep, labels, rows = dataset.get_data_and_labels()
    # One row per stay with the features of the last hour
    np.testing.assert_array_equal(rep, [[3, 12], [5, 21]])
    np.testing.assert_array_equal(labels, [1, 0])
    np.testing.assert_array_equal(rows.ravel(), [1, 2])
    data, tensor_labels, _ = dataset.to_tensor()
    assert data.shape == (2, 2) and tensor_labels.tolist() == [1, 0]


def test_polars_hourly_ml_rows_drop_missing_labels():
    dataset = PredictionPolarsDataset(splits(hourly=True), vars=VARS)
    rep, labels, rows = dataset.get_data_and_labels()
    assert rep.shape == (4, 2) and labels.tolist() == [0, 1, 0, 0]
    assert rows.tolist() == [[1, 0], [1, 2], [2, 0], [2, 1]]


def test_balance_weights_inverse_class_frequency():
    data = splits(hourly=False)
    data[Split.train][Segment.outcome] = pl.DataFrame({"stay_id": [1, 2], "label": [1.0, 1.0]})
    assert PredictionPolarsDataset(data, vars=VARS).get_balance() == pytest.approx([1.0])
    data[Split.train][Segment.outcome] = pl.DataFrame({"stay_id": [1, 2, 3, 4], "label": [1.0, 0.0, 0.0, 0.0]})
    data[Split.train][Segment.features] = pl.DataFrame({"stay_id": [1, 2, 3, 4], "time": [0] * 4, "x": [0.0] * 4})
    # One positive in four stays: one weight is three times the other (sorted, the order is not fixed by the label)
    assert sorted(PredictionPolarsDataset(data, vars=VARS).get_balance()) == pytest.approx([2 / 3, 2.0])


def test_vars_must_be_dict():
    with pytest.raises(ValueError, match="dict"):
        PredictionPolarsDataset(splits(hourly=False), vars=["stay_id"])


def test_pandas_dataset_matches_polars():
    with pytest.warns(DeprecationWarning):
        pandas = PredictionPandasDataset(splits(hourly=False, frame=pd.DataFrame), vars=VARS)
    polars = PredictionPolarsDataset(splits(hourly=False), vars=VARS)
    assert len(pandas) == len(polars) == 2
    for (a, b, c), (d, e, f) in zip(pandas, polars):
        # The pandas dataset keeps the input column order
        torch.testing.assert_close(a[:, [1, 0]].float(), d)
        torch.testing.assert_close(b, e)
        assert c.tolist() == f.tolist()
    rep, labels, _ = pandas.get_data_and_labels()
    np.testing.assert_array_equal(rep[:, [1, 0]], polars.get_data_and_labels()[0])
    assert sorted(pandas.get_balance()) == pytest.approx([1.0, 1.0])
    assert pandas.to_tensor()[0].shape == (2, 2)


def test_imputation_prediction_dataset_groups_rows_per_stay():
    frame = pd.DataFrame({"stay_id": [1, 1, 2], "x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, 6.0]})
    dataset = ImputationPredictionDataset(frame, "stay_id", ["x"], ram_cache=False)
    assert len(dataset) == 2 and dataset.maxlen == 2
    assert dataset[0].shape == (2, 1) and dataset[1].tolist() == [[3.0]]
