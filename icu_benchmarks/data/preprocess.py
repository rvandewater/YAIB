import gc

import logging
import numpy as np
import pandas as pd
import tables
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from icu_benchmarks.common import constants


def gather_cat_values(common_path, cat_values):
    # not too many, so read all of them
    df_cat = pd.read_parquet(common_path, columns=list(cat_values))

    d = {}
    for c in df_cat.columns:
        d[c] = [x for x in df_cat[c].unique() if not np.isnan(x)]
    return d


def gather_stats_over_dataset(parts, to_standard_scale, to_min_max_scale, train_split_pids, fill_string):
    minmax_scaler = MinMaxScaler()

    for p in parts:
        df_part = impute_df(pd.read_parquet(p, engine='pyarrow', columns=to_min_max_scale + [constants.PID],
                                            filters=[(constants.PID, "in", train_split_pids)]), fill_string=fill_string)
        df_part = df_part.replace(np.inf, np.nan).replace(-np.inf, np.nan)
        minmax_scaler.partial_fit(df_part[to_min_max_scale])

        gc.collect()

    means = []
    stds = []
    # cannot read all to_standard_scale columns in memory, one-by-one would very slow, so read a certain number
    # of columns at a time
    batch_size = 20
    batches = (to_standard_scale[pos:pos + batch_size] for pos in range(0, len(to_standard_scale), batch_size))
    for s in batches:
        dfs = impute_df(pd.read_parquet(parts[0].parent, engine='pyarrow', columns=[constants.PID] + s,
                                        filters=[(constants.PID, "in", train_split_pids)]),
                        fill_string=fill_string)
        dfs = dfs.replace(np.inf, np.nan).replace(-np.inf, np.nan)

        # don't rely on sklearn StandardScaler as partial_fit does not seem to work correctly
        # if in one iteration all values of a column are nan (i.e. the then mean becomes nan)
        means.extend(dfs[s].mean())
        stds.extend(dfs[s].std(ddof=0))  # ddof=0 to be consistent with sklearn StandardScalar
        gc.collect()

    return (means, stds), minmax_scaler


def _normalize_cols(df, output_cols):
    cols_to_drop = [c for c in set(df.columns).difference(output_cols) if c != constants.PID]
    if cols_to_drop:
        logging.warning(f"Dropping columns {cols_to_drop} as they don't appear in output columns")
    df = df.drop(columns=cols_to_drop)

    cols_to_add = sorted(set(output_cols).difference(df.columns))

    if cols_to_add:
        logging.warning(f"Adding dummy columns {cols_to_add}")
        df[cols_to_add] = 0.0

    col_order = [constants.DATETIME] + sorted([c for c in df.columns if c != constants.DATETIME])
    df = df[col_order]

    cmp_list = list(c for c in df.columns if c != constants.PID)
    assert cmp_list == output_cols

    return df


def to_ml(save_path, parts, labels, features, endpoint_names, df_var_ref, fill_string, output_cols, split_path=None,
          random_seed=42):
    df_part = pd.read_parquet(parts[0])
    data_cols = df_part.columns

    common_path = parts[0].parent
    df_pid_and_time = pd.read_parquet(common_path, columns=[constants.PID, constants.DATETIME])

    # list of patients for every split
    split_ids = get_splits(df_pid_and_time, split_path, random_seed)

    cat_values, binary_values, to_standard_scale, to_min_max_scale = get_var_types(data_cols, df_var_ref)
    to_standard_scale = [c for c in to_standard_scale if c in set(output_cols)]
    to_min_max_scale = [c for c in to_min_max_scale if c in set(output_cols)]

    cat_vars_levels = gather_cat_values(common_path, cat_values)

    (means, stds), minmax_scaler = gather_stats_over_dataset(parts, to_standard_scale, to_min_max_scale,
                                                             split_ids['train'], fill_string)

    # for every train, val, test split keep how many records
    # have already been written (needed to compute correct window position)
    output_offsets = {}

    features_available = features
    if not features_available:
        features = [None] * len(parts)

    for p, l, f in zip(parts, labels, features):
        df = impute_df(pd.read_parquet(p), fill_string=fill_string)
        df_feat = pd.read_parquet(f) if f else pd.DataFrame(columns=[constants.PID])

        df_label = pd.read_parquet(l)[
            [constants.PID, constants.REL_DATETIME] + list(endpoint_names)]
        df_label = df_label.rename(columns={constants.REL_DATETIME: constants.DATETIME})
        df_label[constants.DATETIME] = df_label[constants.DATETIME] / 60.0

        # align indices between labels df and common df
        df_label = df_label.set_index([constants.PID, constants.DATETIME])
        df_label = df_label.reindex(index=zip(df[constants.PID].values, df[constants.DATETIME].values))
        df_label = df_label.reset_index()

        for cat_col in cat_values:
            df[cat_col] = pd.Categorical(df[cat_col], cat_vars_levels[cat_col])

        for bin_col in binary_values:
            bin_vals = [0.0, 1.0]
            if bin_col == 'sex':
                bin_vals = ['F', 'M']
            df[bin_col] = pd.Categorical(df[bin_col], bin_vals)

        if cat_values:
            df = pd.get_dummies(df, columns=cat_values)
        if binary_values:
            df = pd.get_dummies(df, columns=binary_values, drop_first=True)

        df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan)

        # reorder columns and making sure columns correspond to output_cols
        df = _normalize_cols(df, output_cols)

        split_dfs = {}
        split_labels = {}
        split_features = {}
        for split in split_ids.keys():
            split_dfs[split] = df[df[constants.PID].isin(split_ids[split])]
            split_labels[split] = df_label[df_label[constants.PID].isin(split_ids[split])]
            split_features[split] = df_feat[df_feat[constants.PID].isin(split_ids[split])]

        # windows computation: careful with offset!
        split_windows = {}
        for split, df in split_dfs.items():
            if df.empty:
                split_windows[split] = np.array([])
                continue
            split_windows[split] = get_windows_split(df, offset=output_offsets.get(split, 0))

            assert np.all(
                split_windows[split] == get_windows_split(split_labels[split], offset=output_offsets.get(split, 0)))
            split_dfs[split] = df.drop(columns=[constants.PID])
            split_labels[split] = split_labels[split].drop(columns=[constants.PID])
            split_features[split] = split_features[split].drop(columns=[constants.PID])

            output_offsets[split] = output_offsets.get(split, 0) + len(df)

        for split_df in split_dfs.values():
            if split_df.empty:
                continue
            split_df[to_standard_scale] = (split_df[to_standard_scale].values - means) / stds
            split_df[to_min_max_scale] = minmax_scaler.transform(split_df[to_min_max_scale])
            split_df.replace(np.inf, np.nan, inplace=True)
            split_df.replace(-np.inf, np.nan, inplace=True)

        split_arrays = {}
        label_arrays = {}
        feature_arrays = {}
        for split, df in split_dfs.items():
            array_split = df.values
            array_label = split_labels[split].values

            np.place(array_split, mask=np.isnan(array_split), vals=0.0)

            split_arrays[split] = array_split
            label_arrays[split] = array_label

            if features_available:
                array_features = split_features[split].values
                np.place(array_features, mask=np.isnan(array_features), vals=0.0)
                feature_arrays[split] = array_features

            assert len(df.columns) == split_arrays[split].shape[1]

        tasks = list(split_labels['train'].columns)

        output_cols = [c for c in df.columns if c != constants.PID]

        feature_names = list(split_features['train'].columns)

        save_to_h5_with_tasks(save_path, output_cols, tasks, feature_names,
                              split_arrays, label_arrays,
                              feature_arrays if features_available else None,
                              split_windows)

        gc.collect()


def _write_data_to_hdf(data, dataset_name, node, f, first_write, nr_cols, expectedrows=1000000):
    filters = tables.Filters(complevel=5, complib='blosc:lz4')

    if first_write:
        ea = f.create_earray(node, dataset_name,
                             atom=tables.Atom.from_dtype(data.dtype),
                             expectedrows=expectedrows,
                             shape=(0, nr_cols),
                             filters=filters)
        if len(data) > 0:
            ea.append(data)
    elif len(data) > 0:
        node[dataset_name].append(data)


def save_to_h5_with_tasks(save_path, col_names, task_names, feature_names, data_dict, label_dict, features_dict,
                          patient_windows_dict):
    """
    Save a dataset with the desired format as h5.
    Args:
        save_path: Path to save the dataset to.
        col_names: List of names the variables in the dataset.
        task_names: List of names for the tasks in the dataset.
        feature_names: List of names for the features in the dataset.
        data_dict: Dict with an array for each split of the data
        label_dict: (Optional) Dict with each split and labels array in same order as lookup_table.
        features_dict: Dict with each split and features array in same order as lookup_table.
        patient_windows_dict: Dict containing a array for each split such that each row of the array is of the type
        [start_index, stop_index, patient_id].
    Returns:
    """

    # data labels windows

    first_write = not save_path.exists()
    mode = 'w' if first_write else 'a'

    with tables.open_file(save_path, mode) as f:
        if first_write:
            n_data = f.create_group("/", 'data', 'Dataset')
            f.create_array(n_data, 'columns', obj=[str(k).encode('utf-8') for k in col_names])
        else:
            n_data = f.get_node('/data')

        splits = ['train', 'val', 'test']
        for split in splits:
            _write_data_to_hdf(data_dict[split].astype(float), split, n_data, f, first_write,
                               data_dict['train'].shape[1])

        if label_dict is not None:
            if first_write:
                labels = f.create_group("/", 'labels', 'Labels')
                f.create_array(labels, 'tasks', obj=[str(k).encode('utf-8') for k in task_names])
            else:
                labels = f.get_node('/labels')

            for split in splits:
                _write_data_to_hdf(label_dict[split].astype(float), split, labels, f, first_write,
                                   label_dict['train'].shape[1])

        if features_dict is not None:
            if first_write:
                features = f.create_group("/", 'features', 'Features')
                f.create_array(features, 'name_features', obj=[str(k).encode('utf-8') for k in feature_names])
            else:
                features = f.get_node('/features')

            for split in splits:
                _write_data_to_hdf(features_dict[split].astype(float), split, features, f, first_write,
                                   features_dict['train'].shape[1])

        if patient_windows_dict is not None:
            if first_write:
                p_windows = f.create_group("/", 'patient_windows', 'Windows')
            else:
                p_windows = f.get_node('/patient_windows')

            for split in splits:
                _write_data_to_hdf(patient_windows_dict[split].astype(int), split, p_windows, f, first_write,
                                   patient_windows_dict['train'].shape[1])

        if not len(col_names) == data_dict['train'].shape[-1]:
            raise Exception(
                "We saved to data but the number of columns ({}) didn't match the number of features {} ".format(
                    len(col_names), data_dict['train'].shape[-1]))


def impute_df(df, fill_string='ffill'):
    df = df.groupby(constants.PID).apply(lambda x: x.fillna(method=fill_string))
    return df


def get_var_types(columns, df_var_ref):
    cat_ref = list(df_var_ref[df_var_ref.variableunit == 'Categorical']['metavariablename'].values)
    cat_values = [c for c in cat_ref if c in columns]
    binary_values = list(np.unique(df_var_ref[df_var_ref['metavariableunit'] == 'Binary']['metavariablename']))
    binary_values += ['sex']
    to_standard_scale = [k for k in np.unique(df_var_ref['metavariablename'].astype(str)) if
                         k not in cat_values + binary_values] + ['age', 'height']
    to_standard_scale = [c for c in to_standard_scale if c in columns]

    to_min_max_scale = [constants.DATETIME, 'admissiontime', 'height']
    return cat_values, binary_values, to_standard_scale, to_min_max_scale


def get_splits(df, split_path, random_seed):
    if split_path:
        split_df = pd.read_csv(split_path, sep='\t')
        split_ids = {}
        for split in split_df['split'].unique():
            split_ids[split] = split_df.loc[split_df['split'] == split, constants.PID].values
    else:
        split_ids = {}
        train_val_ids, split_ids['test'] = train_test_split(np.unique(df[constants.PID]), test_size=0.15,
                                                            random_state=random_seed)
        split_ids['train'], split_ids['val'] = train_test_split(train_val_ids, test_size=(0.15 / 0.85),
                                                                random_state=random_seed)
    return split_ids


def get_windows_split(df_split, offset=0):
    pid_array = df_split[constants.PID]
    starts = sorted(np.unique(pid_array, return_index=True)[1])
    stops = np.concatenate([starts[1:], [df_split.shape[0]]])
    ids = pid_array.values[starts]
    return np.stack([np.array(starts) + offset, np.array(stops) + offset, ids], axis=1)
