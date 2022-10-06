import pandas as pd
import pyarrow as pa
from sklearn.base import TransformerMixin

from icu_benchmarks.common import constants

VARS = constants.VARS
FILE_NAMES = constants.FILE_NAMES

# TODO make function take df
# TODO make proportions of splits as parameter
def generate_splits(sta_path, outc_path, dyn_path, static_splits_path, labels_splits_path, dyn_splits_path):
    """
    1. Generates training, validation and test splits in the data.
    2. Merges dynamic data with splits for easy accessing.
    """
    static_df = pa.parquet.read_table(sta_path).to_pandas()

    train_df = static_df.sample(frac=0.7, random_state=3333)
    train_df["split"] = "train"

    validation_df = static_df.drop(train_df.index).sample(frac=0.5, random_state=25)
    validation_df["split"] = "val"

    test_df = static_df.drop(train_df.index).drop(validation_df.index)
    test_df["split"] = "test"

    static_splits_df = pd.concat([train_df, validation_df, test_df]).sort_index()
    splits_reindexed_df = static_splits_df.set_index("split")
    pa.parquet.write_table(pa.Table.from_pandas(splits_reindexed_df), static_splits_path)

    splits_df = static_splits_df[["split", VARS["STAY_ID"]]]

    dyn_df = pa.parquet.read_table(dyn_path).to_pandas()
    dyn_with_splits_df = dyn_df.merge(splits_df, on=VARS["STAY_ID"]).set_index(["split"])

    # TODO check whether this works for different label format too
    outc_df = pa.parquet.read_table(outc_path).to_pandas()
    labels_splits_df = outc_df.merge(splits_df, on=VARS["STAY_ID"]).set_index(["split"])
    pa.parquet.write_table(pa.Table.from_pandas(labels_splits_df), labels_splits_path)

    return pa.parquet.write_table(pa.Table.from_pandas(dyn_with_splits_df), dyn_splits_path)
