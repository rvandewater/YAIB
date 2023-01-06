import numpy as np


def value_empty(size, default_val, dtype=None):
    """ Returns a vector filled with elements of a specific value"""

    if dtype is not None:
        tmp_arr = np.empty(size, dtype=dtype)
    else:
        tmp_arr = np.empty(size)

    tmp_arr[:] = default_val
    return tmp_arr


def empty_nan(sz):
    """ Returns an empty NAN vector of specified size"""
    arr = np.empty(sz)
    arr[:] = np.nan
    return arr


""" ONLY FORWARD FILLING imputation schema"""


def impute_forward_fill_simple(raw_ts, raw_values, timegrid_pred, global_mean, fill_interval_secs=np.inf):
    """
    Simple forward filling algorithm used in the respiratory failure endpoints
    """

    pred_values = np.zeros_like(timegrid_pred)
    cum_count_ms = np.zeros_like(timegrid_pred)
    time_to_last_ms = value_empty(timegrid_pred.size, -1.0)
    input_ts = 0
    cum_real_meas = 0
    last_real_ms = None

    for idx, ts in np.ndenumerate(timegrid_pred):

        while input_ts < raw_ts.size and raw_ts[input_ts] <= ts:
            cum_real_meas += 1
            last_real_ms = input_ts
            input_ts += 1

        # No value has been observed before the current time-grid point. We have to fill in using the global mean
        if input_ts == 0:
            pred_values[idx[0]] = global_mean
            continue

        ext_offset = ts - raw_ts[input_ts - 1]

        if last_real_ms is None:
            real_offset = -1.0
        else:
            real_offset = ts - raw_ts[last_real_ms]

        assert (ext_offset >= 0)

        # Fill with normal value after forward filling horizon
        if ext_offset > fill_interval_secs:
            pred_values[idx[0]] = global_mean
        else:

            # Handle the special case where the same variable was observed at the exact time stamp in two tables
            if input_ts > 1 and raw_ts[input_ts - 1] == raw_ts[input_ts - 2]:
                gen_val = np.mean(raw_values[input_ts - 2:input_ts])
                pred_values[idx[0]] = gen_val
            else:
                gen_val = raw_values[input_ts - 1]
                pred_values[idx[0]] = gen_val

        cum_count_ms[idx[0]] = cum_real_meas
        time_to_last_ms[idx[0]] = real_offset

    return (pred_values, cum_count_ms, time_to_last_ms)
