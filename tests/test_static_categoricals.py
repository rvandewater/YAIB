"""Static categoricals: nulls are imputed with the train mode and unseen categories do not crash val/test (#251)."""

import polars as pl

from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.preprocessor import PolarsClassificationPreprocessor

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["hr"], "STATIC": ["age", "sex"]}


def segments(stays, sex):
    n = len(stays)
    return {
        Segment.static: pl.DataFrame({"stay_id": stays, "age": [50.0 + i for i in range(n)], "sex": sex}),
        Segment.dynamic: pl.DataFrame({"stay_id": stays, "time": [0.0] * n, "hr": [80.0] * n}),
        Segment.outcome: pl.DataFrame({"stay_id": stays, "label": [0] * n}),
    }


def encoded_sex(train, val, test):
    data = {
        Split.train: segments([1, 2, 3, 4, 5], train),
        Split.val: segments([6, 7], val),
        Split.test: segments([8, 9], test),
    }
    data = PolarsClassificationPreprocessor().apply(data, VARS)
    return {split: data[split][Segment.features].sort("stay_id")["sex"].to_list() for split in data}


def test_null_static_categorical_imputed_with_train_mode():
    # Female -> 0, Male -> 1; a missing sex takes the train mode (Male) instead of becoming a category of its own
    sex = encoded_sex(["Male", "Male", None, "Female", "Male"], ["Male", None], [None, "Female"])
    assert sex == {Split.train: [1, 1, 1, 0, 1], Split.val: [1, 1], Split.test: [1, 0]}


def test_unseen_static_categorical_encoded_as_unknown():
    sex = encoded_sex(["Male", "Male", "Female", "Female", "Male"], ["Other", "Male"], ["Female", "Other"])
    assert sex == {Split.train: [1, 1, 0, 0, 1], Split.val: [-1, 1], Split.test: [0, -1]}
