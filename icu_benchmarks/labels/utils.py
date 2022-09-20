import numpy as np
from icu_benchmarks.common.constants import STEPS_PER_HOUR, BINARY_TSH_URINE


def get_hr_status(hr_col):
    """Return a presence feature on HR given the cumulative counts of HR.

    Presence at time t is valid if there is a HR measurement in th 15min surrounding the index.
    i.e in [idx-1, idx. idx+1]
    """

    hr_status_arr = np.zeros_like(hr_col)
    for jdx in range(hr_col.size):
        subarr = hr_col[max(0, jdx - 2):min(hr_col.size - 1, jdx + 2)]
        three_steps_hr_diff = subarr[-1] - subarr[0]
        hr_status_arr[jdx] = 1 if three_steps_hr_diff > 0 else 0

    return hr_status_arr


def get_any_resp_label(pre_resp_arr):
    """Transforms multiple level of severity annotation of respiratory failure events to a unique binary label"""

    ann_resp_arr = np.zeros_like(pre_resp_arr)
    for jdx in range(ann_resp_arr.size):
        if pre_resp_arr[jdx] in ["event_1", "event_2", "event_3"]:
            ann_resp_arr[jdx] = 1
        elif pre_resp_arr[jdx] in ["UNKNOWN"]:
            ann_resp_arr[jdx] = np.nan
    return ann_resp_arr


def convolve_hr(in_arr, hr_status_arr):
    """ Convolve an array with a HR status arr"""
    out_arr = np.copy(in_arr)
    out_arr[hr_status_arr == 0] = np.nan
    return out_arr


def transition_to_abs(score_arr, target, lhours, rhours):
    """ Transition to an absolute value from a value below the target"""
    out_arr = np.zeros_like(score_arr)
    for jdx in range(out_arr.size):
        if score_arr[jdx] >= target:
            out_arr[jdx] = np.nan
            continue
        low_idx = min(score_arr.size, jdx + 1 + STEPS_PER_HOUR * lhours)
        high_idx = min(score_arr.size, jdx + 1 + STEPS_PER_HOUR * rhours)
        future_arr = score_arr[low_idx:high_idx]
        if future_arr.size == 0:
            out_arr[jdx] = np.nan
        elif np.sum(future_arr >= target) > 0:
            out_arr[jdx] = 1.0
    return out_arr


def unique_label_at_hours(stay_length, status, at_hours=24):
    """ Unique Label assigned at a fixed time-point"""
    out_arr = np.zeros(stay_length)
    out_arr[:] = np.nan
    if stay_length >= at_hours * STEPS_PER_HOUR:
        out_arr[at_hours * STEPS_PER_HOUR - 1] = status
    return out_arr


def transition_to_failure(ann_col, lhours=None, rhours=None):
    """ Transition to failure defined on a binary annotation column"""
    out_arr = np.zeros_like(ann_col)
    for jdx in range(len(out_arr)):
        if np.isnan(ann_col[jdx]) or ann_col[jdx] == 1:
            out_arr[jdx] = np.nan
        elif ann_col[jdx] == 0:
            low_idx = min(ann_col.size, jdx + 1 + lhours * STEPS_PER_HOUR)
            high_idx = min(ann_col.size, jdx + 1 + rhours * STEPS_PER_HOUR)
            fut_arr = ann_col[low_idx:high_idx]
            if (fut_arr == 1.0).any():
                out_arr[jdx] = 1
    return out_arr


def future_urine_output(urine_col, urine_meas_arr, weight_col, rhours=None):
    """ Regression and binary classification problems on urine output in the future"""
    reg_out_arr = np.zeros_like(urine_col)
    binary_out_arr = np.zeros_like(urine_col)

    for jdx in range(urine_col.size):

        # No valid urine measurement anchor in 2 hours, the task is invalid
        measurement_idx = min(jdx + STEPS_PER_HOUR * rhours, urine_col.size - 1)

        end_of_stay_before_2h = measurement_idx == urine_col.size - 1
        urine_diff = urine_meas_arr[measurement_idx] - urine_meas_arr[measurement_idx - 1]
        no_measurement_in_2h = urine_diff <= 0

        if end_of_stay_before_2h or no_measurement_in_2h:
            binary_out_arr[jdx] = np.nan
            reg_out_arr[jdx] = np.nan
            continue

        current_weight = weight_col[jdx]

        cum_increase = 0
        for kdx in range(1, rhours * STEPS_PER_HOUR + 1):
            if jdx + kdx >= urine_col.size:
                break
            cum_increase += urine_col[jdx + kdx]
        std_cum_increase = cum_increase / (current_weight * STEPS_PER_HOUR * rhours)
        reg_out_arr[jdx] = std_cum_increase

        # More than 0.5 ml/kg/h
        if std_cum_increase >= BINARY_TSH_URINE:
            binary_out_arr[jdx] = 1.0

    return reg_out_arr, binary_out_arr


def merge_apache_groups(apache_ii_group, apache_iv_group, apache_ii_map, apache_iv_map):
    """Merges apache II and IV definition to have a unique APACHE label"""

    if np.isfinite(apache_ii_group) and int(apache_ii_group) in apache_ii_map.keys():
        apache_pat_group = apache_ii_map[int(apache_ii_group)]
    elif np.isfinite(apache_iv_group) and int(apache_iv_group) in apache_iv_map.keys():
        apache_pat_group = apache_iv_map[int(apache_iv_group)]
    else:
        apache_pat_group = np.nan
    return apache_pat_group
