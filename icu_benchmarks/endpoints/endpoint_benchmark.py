""" Functions to generate the set of endpoints for the time series
    benchmark on the HiRID database"""

import glob
import logging
import math
import os
import os.path
import sys
import gin

import numpy as np
import pandas as pd
import skfda.preprocessing.smoothing.kernel_smoothers as skks
import skfda.representation.grid as skgrid
from icu_benchmarks.common.constants import STEPS_PER_HOUR, LEVEL1_RATIO_RESP, LEVEL2_RATIO_RESP, LEVEL3_RATIO_RESP, \
    FRACTION_TSH_CIRC, FRACTION_TSH_RESP, DATETIME, PID, REL_DATETIME, SPO2_NORMAL_VALUE, NIV_VENT_MODE, \
    SUPPOX_TO_FIO2, PAO2_MIX_SCALE, ABGA_WINDOW, SUPPOX_MAX_FFILL, AMBIENT_FIO2, EVENT_SEARCH_WINDOW, \
    FI02_SEARCH_WINDOW, PA02_SEARCH_WINDOW, PEEP_SEARCH_WINDOW, HR_SEARCH_WINDOW, VENT_VOTE_TSH, PEEP_TSH, \
    FRACTION_VENT_HR_GAP, SPO2_PERCENTILE, SPO2_MIN_WINDOW, SHORT_GAP_TSH, SHORT_EVENT_TSH, PAO2_BW, \
    PF_MERGE_THRESHOLD, OFFSET_RESP, VAR_IDS_EP

MINS_PER_STEP = 60 // STEPS_PER_HOUR
MAX_SUPPOX_KEY = np.array(list(SUPPOX_TO_FIO2.keys())).max()
MAX_SUPPOX_TO_FIO2_VAL = SUPPOX_TO_FIO2[MAX_SUPPOX_KEY]

def mix_real_est_pao2(pao2_col, pao2_meas_cnt, pao2_est_arr):
    """ Mix real PaO2 measurement and PaO2 estimates using a Gaussian kernel
     
    INPUTS:
    pao2_col: Imputed PaO2 time series
    pao2_meas_cnt: Cumulative real measurement counts of PaO2
    pao2_est_arr: PaO2 estimates at each point computed by another function

    RETURNS: A 1D time series of the final mixed PaO2 estimate
    """
    final_pao2_arr = np.copy(pao2_est_arr)
    sq_scale = PAO2_MIX_SCALE  # 1 hour has mass 1/3 approximately

    for idx in range(final_pao2_arr.size):
        meas_ref = pao2_meas_cnt[idx]
        real_val = None
        real_val_dist = None

        # Search forward and backward with priority giving to backward if equidistant
        for sidx in range(4 * STEPS_PER_HOUR):
            if not idx - sidx < 0 and pao2_meas_cnt[idx - sidx] < meas_ref:
                real_val = pao2_col[idx - sidx + 1]
                real_val_dist = MINS_PER_STEP * sidx
                break
            elif not idx + sidx >= final_pao2_arr.size and pao2_meas_cnt[idx + sidx] > meas_ref:
                real_val = pao2_col[idx + sidx]
                real_val_dist = MINS_PER_STEP * sidx
                break

        if real_val is not None:
            alpha_mj = math.exp(-real_val_dist ** 2 / sq_scale)
            alpha_ej = 1 - alpha_mj
            final_pao2_arr[idx] = alpha_mj * real_val + alpha_ej * pao2_est_arr[idx]

    return final_pao2_arr


def kernel_smooth_arr(input_arr, bandwidth=None):
    """ Kernel smooth an input array with a Nadaraya-Watson kernel smoother
    
    INPUTS:
    input_arr: Input array to be smoothed

    RETURNS: Input array smoothed with kernel
    """
    output_arr = np.copy(input_arr)
    fin_arr = output_arr[np.isfinite(output_arr)]
    time_axis = MINS_PER_STEP * np.arange(len(output_arr))
    fin_time = time_axis[np.isfinite(output_arr)]

    # Return the unsmoothed array if fewer than 2 observations
    if fin_arr.size < 2:
        return output_arr

    smoother = skks.NadarayaWatsonSmoother(smoothing_parameter=bandwidth)
    fgrid = skgrid.FDataGrid([fin_arr], fin_time)
    fd_smoothed = smoother.fit_transform(fgrid)
    output_smoothed = fd_smoothed.data_matrix.flatten()
    output_arr[np.isfinite(output_arr)] = output_smoothed
    return output_arr


def percentile_smooth(signal_col, percentile, win_scope_mins):
    """ Window percentile smoother, where percentile is in the interval [0,100]
    
    INPUTS:
    signal_col: Input time series to be smoothed
    percentile: Which percentile should be used to take from the centralized window?
    win_scope_mins: Length of smoothing windows in minutes

    RETURNS: Smoothed input array
    """
    out_arr = np.zeros_like(signal_col)
    mins_per_window = MINS_PER_STEP
    search_range = int(win_scope_mins / mins_per_window / 2)
    for jdx in range(out_arr.size):
        search_arr = signal_col[max(0, jdx - search_range):min(out_arr.size, jdx + search_range)]
        out_arr[jdx] = np.percentile(search_arr, percentile)
    return out_arr


def assign_resp_levels(pf_event_est_arr=None, vent_status_arr=None,
                       sz_window=None, peep_status_arr=None, peep_threshold_arr=None,
                       offset_back_windows=None):
    """Now label based on the array of estimated Horowitz indices

    INPUTS: 
    event_status_arr: The array to be filled with the ventilation status (OUTPUT)
    pf_event_est_arr: Array of estimated P/F ratios at each time step
    vent_status_arr: Array of estimated ventilation status at each time-step
    sz_window: Size of forward windows specified as multiple of time series gridding size
    peep_status_arr: Status array of available PEEP measurement at a time-point
    peep_threshold_arr: Binary array indicating if PEEP threshold was reached at a time-point
    offset_back_windows: Do not compute respiratory level for incomplete windows at the end of stay

    RETURNS: Filled event status array
    """
    n_steps = len(pf_event_est_arr)
    new_event_status_arr = np.zeros(n_steps, dtype="<S10")
    new_event_status_arr.fill("UNKNOWN")
    for idx in range(0, len(new_event_status_arr) - offset_back_windows):
        # We search in the future for resp failure (different from circ failure which is centered)
        est_idx = pf_event_est_arr[idx:min(n_steps, idx + sz_window)]
        est_vent = vent_status_arr[idx:min(n_steps, idx + sz_window)]
        est_peep_dense = peep_status_arr[idx:min(n_steps, idx + sz_window)]
        est_peep_threshold = peep_threshold_arr[idx:min(n_steps, idx + sz_window)]

        if np.sum((est_idx <= LEVEL3_RATIO_RESP) & (
                (est_vent == 0.0) | (est_vent == 1.0) & (est_peep_dense == 0.0) | (est_vent == 1.0) & (
                est_peep_dense == 1.0) & (est_peep_threshold == 1.0))) >= FRACTION_TSH_RESP * len(est_idx):
            new_event_status_arr[idx] = "event_3"
        elif np.sum((est_idx <= LEVEL2_RATIO_RESP) & (
                (est_vent == 0.0) | (est_vent == 1.0) & (est_peep_dense == 0.0) | (est_vent == 1.0) & (
                est_peep_dense == 1.0) & (est_peep_threshold == 1.0))) >= FRACTION_TSH_RESP * len(est_idx):
            new_event_status_arr[idx] = "event_2"
        elif np.sum((est_idx <= LEVEL1_RATIO_RESP) & (
                (est_vent == 0.0) | (est_vent == 1.0) & (est_peep_dense == 0.0) | (est_vent == 1.0) & (
                est_peep_dense == 1.0) & (est_peep_threshold == 1.0))) >= FRACTION_TSH_RESP * len(est_idx):
            new_event_status_arr[idx] = "event_1"
        elif np.sum(np.isnan(est_idx)) < FRACTION_TSH_RESP * len(est_idx):
            new_event_status_arr[idx] = "event_0"

    return new_event_status_arr


def correct_right_edge_l0(event_status_arr=None, pf_event_est_arr=None,
                          offset_back_windows=None):
    """Correct right edges of event 0 (correct level to level 0)

    INPUTS:
    event_status_arr: Estimate resp event level at each time-point
    pf_event_est_arr: 1D time series of estimated P/F ratio at each time-step
    offset_back_windows: Do not process edge windows at the end of the stay

    RETURNS: Event status array with right edge of L0 zones corrected
    """
    on_right_edge = False
    in_event = False
    corrected_event_status_arr = np.copy(event_status_arr)
    for idx in range(0, len(corrected_event_status_arr) - offset_back_windows):
        cur_state = corrected_event_status_arr[idx].decode()
        if cur_state in ["event_0"] and not in_event:
            in_event = True
        elif in_event and cur_state not in ["event_0"]:
            in_event = False
            on_right_edge = True
        if on_right_edge:
            if pf_event_est_arr[idx] < LEVEL1_RATIO_RESP:
                on_right_edge = False
            else:
                corrected_event_status_arr[idx] = "event_0"

    return corrected_event_status_arr


def correct_right_edge_l1(event_status_arr=None, pf_event_est_arr=None,
                          offset_back_windows=None):
    """Correct right edges of event 1 (correct to level 1)

    INPUTS:
    event_status_arr: Estimate resp event level at each time-point
    pf_event_est_arr: 1D time series of estimated P/F ratio at each time-step
    offset_back_windows: Do not process edge windows at the end of the stay

    RETURNS: Event status array with right edge of L1 zones corrected
    """
    on_right_edge = False
    in_event = False
    corrected_event_status_arr = np.copy(event_status_arr)
    for idx in range(0, len(corrected_event_status_arr) - offset_back_windows):
        cur_state = corrected_event_status_arr[idx].decode()
        if cur_state in ["event_1"] and not in_event:
            in_event = True
        elif in_event and cur_state not in ["event_1"]:
            in_event = False
            on_right_edge = True
        if on_right_edge:
            if pf_event_est_arr[idx] < LEVEL2_RATIO_RESP or pf_event_est_arr[idx] >= LEVEL1_RATIO_RESP:
                on_right_edge = False
            else:
                corrected_event_status_arr[idx] = "event_1"

    return corrected_event_status_arr


def correct_right_edge_l2(event_status_arr=None, pf_event_est_arr=None,
                          offset_back_windows=None):
    """Correct right edges of event 2 (correct to level 2)

    INPUTS:
    event_status_arr: Estimate resp event level at each time-point
    pf_event_est_arr: 1D time series of estimated P/F ratio at each time-step
    offset_back_windows: Do not process edge windows at the end of the stay

    RETURNS: Event status array with right edge of L2 zones corrected
    """
    on_right_edge = False
    in_event = False
    corrected_event_status_arr = np.copy(event_status_arr)
    for idx in range(0, len(corrected_event_status_arr) - offset_back_windows):
        cur_state = corrected_event_status_arr[idx].decode()
        if cur_state in ["event_2"] and not in_event:
            in_event = True
        elif in_event and cur_state not in ["event_2"]:
            in_event = False
            on_right_edge = True
        if on_right_edge:
            if pf_event_est_arr[idx] < LEVEL3_RATIO_RESP or pf_event_est_arr[idx] >= LEVEL2_RATIO_RESP:
                on_right_edge = False
            else:
                corrected_event_status_arr[idx] = "event_2"

    return corrected_event_status_arr


def correct_right_edge_l3(event_status_arr=None, pf_event_est_arr=None,
                          offset_back_windows=None):
    """ Correct right edges of event 3 (correct to level 3)

    INPUTS:
    event_status_arr: Estimate resp event level at each time-point
    pf_event_est_arr: 1D time series of estimated P/F ratio at each time-step
    offset_back_windows: Do not process edge windows at the end of the stay

    RETURNS: Event status array with right edge of L3 zones corrected
    """
    on_right_edge = False
    in_event = False
    corrected_event_status_arr = np.copy(event_status_arr)
    for idx in range(0, len(corrected_event_status_arr) - offset_back_windows):
        cur_state = corrected_event_status_arr[idx].decode()
        if cur_state in ["event_3"] and not in_event:
            in_event = True
        elif in_event and cur_state not in ["event_3"]:
            in_event = False
            on_right_edge = True
        if on_right_edge:
            if pf_event_est_arr[idx] >= LEVEL3_RATIO_RESP:
                on_right_edge = False
            else:
                corrected_event_status_arr[idx] = "event_3"

    return corrected_event_status_arr


def merge_short_vent_gaps(vent_status_arr, short_gap_hours):
    """ Merge short gaps in the ventilation status array

    INPUTS:
    vent_status_arr: Binary ventilation status estmate at each time-step
    short_gap_hours: All gaps which are less than or equal to this threshold will be merged

    RETURNS: Ventilator status array with gaps removed
    """
    in_gap = False
    gap_length = 0
    new_vent_status_arr = np.copy(vent_status_arr)

    for idx in range(len(vent_status_arr)):
        cur_state = vent_status_arr[idx]
        if in_gap and (cur_state == 0.0 or np.isnan(cur_state)):
            gap_length += MINS_PER_STEP
        elif not in_gap and (cur_state == 0.0 or np.isnan(cur_state)):
            in_gap = True
            in_gap_idx = idx
            gap_length = MINS_PER_STEP
        elif in_gap and cur_state == 1.0:
            in_gap = False
            if gap_length / 60. <= short_gap_hours:
                new_vent_status_arr[in_gap_idx:idx] = 1.0

    return new_vent_status_arr


def delete_short_vent_events(vent_status_arr, short_event_hours):
    """ Delete short events in the ventilation status array
    
    INPUTS:
    vent_status_arr: Original ventilation annotation with small events not yet removed
    short_event_hours: Threshold in hours when an event is considered "small"

    RETURNS: Ventilation status array with small events removed
    """
    new_vent_status_arr = np.copy(vent_status_arr)

    in_event = False
    event_length = 0
    for idx in range(len(vent_status_arr)):
        cur_state = vent_status_arr[idx]
        if in_event and cur_state == 1.0:
            event_length += MINS_PER_STEP
        if not in_event and cur_state == 1.0:
            in_event = True
            event_length = MINS_PER_STEP
            event_start_idx = idx
        if in_event and (cur_state == 0.0 or np.isnan(cur_state)):
            in_event = False
            if event_length / 60. < short_event_hours:
                new_vent_status_arr[event_start_idx:idx] = 0.0
    return new_vent_status_arr


def delete_low_density_hr_gap(vent_status_arr, hr_status_arr):
    """ Deletes gaps in ventilation which are caused by likely sensor dis-connections.
    If HR is not available during FRACTION_VENT_HR_GAP of the steps we set ventilation to 1.

    INPUTS:
    vent_status_arr: Pre-estimated ventilation status at a time-point
    hr_status_arr: Binary indicator of whether HR sensor was connected at a time-point

    RETURNS: Corrected ventilation status array
    """
    in_event = False
    in_gap = False
    gap_idx = -1

    vent_new_arr = np.copy(vent_status_arr)

    for idx in range(len(vent_status_arr)):

        # Beginning of new event, not from inside gap
        if not in_event and not in_gap and vent_new_arr[idx] == 1.0:
            in_event = True

        # Beginning of potential gap that needs to be closed
        elif in_event and vent_new_arr[idx] == 0.0:
            in_gap = True
            gap_idx = idx
            in_event = False

        # The gap is over, re-assign the status of ventilation to merge the gap, enter new event
        if in_gap and vent_new_arr[idx] == 1.0:

            hr_sub_arr = hr_status_arr[gap_idx:idx]

            # Close the gap if the density of HR is too low in between
            if np.sum(hr_sub_arr) / hr_sub_arr.size <= FRACTION_VENT_HR_GAP:
                vent_new_arr[gap_idx:idx] = 1.0

            in_gap = False
            in_event = True

    return vent_new_arr


def ellis(x_orig):
    """ ELLIS model converting SpO2 in 100 % units into a PaO2 ABGA
        estimate
    
    INPUTS: 
    x_orig: SpO2 values from which to estimate

    RETURNS: Estimated PaO2 values for each time-step
    """
    x_new = np.copy(x_orig)
    x_new[np.isnan(x_new)] = SPO2_NORMAL_VALUE  # Normal value assumption
    x = x_new / 100
    x[x == 1] = 0.999
    exp_base = (11700 / ((1 / x) - 1))
    exp_sqrbracket = np.sqrt(pow(50, 3) + (exp_base ** 2))
    exp_first = np.cbrt(exp_base + exp_sqrbracket)
    exp_second = np.cbrt(exp_base - exp_sqrbracket)
    exp_full = exp_first + exp_second
    return exp_full


def delete_small_continuous_blocks(event_arr, block_threshold=None):
    """ Given an event array, deletes small contiguous blocks that are sandwiched between two other blocks, one of which
        is longer, they both have the same label. For the moment we delete blocks smaller than 30 minutes. Note this
        requires only a linear pass over the array

    INPUTS:
    event_arr: Event array (0 no event, 1 event, or discrete for the case of respiratory event labels)
    block_threshold: Blocks smaller than this length should be removed

    RETURNS: Event array after small continuous blocks are removed
    """

    block_list = []
    active_block = None

    # Build a block list
    for jdx in range(event_arr.size):
        new_block = event_arr[jdx]

        # Start a new block at the beginning
        if active_block is None:
            active_block = new_block
            left_block_idx = jdx

        # Change to a new block
        elif not active_block == new_block:
            block_list.append((active_block, left_block_idx, jdx - 1))
            left_block_idx = jdx
            active_block = new_block

        # Same last block unconditionally
        if jdx == event_arr.size - 1:
            block_list.append((new_block, left_block_idx, jdx))

    # Merge blocks
    while True:
        all_clean = True
        for bidx, block in enumerate(block_list):
            block_label, lidx, ridx = block
            block_len = ridx - lidx + 1

            # Candidate for merging
            if block_len <= block_threshold:

                if len(block_list) == 1:
                    all_clean = True
                    break

                # Only right block
                elif bidx == 0:
                    next_block = block_list[bidx + 1]
                    nb_label, nb_lidx, nb_ridx = next_block
                    nb_len = nb_ridx - nb_lidx + 1

                    # Merge blocks
                    if nb_len > block_len and nb_len > block_threshold:
                        block_list[bidx] = (nb_label, lidx, nb_ridx)
                        block_list.remove(next_block)
                        all_clean = False
                        break

                # Only left block
                elif bidx == len(block_list) - 1:
                    prev_block = block_list[bidx - 1]
                    pb_label, pb_lidx, pb_ridx = prev_block
                    pb_len = pb_ridx - pb_lidx + 1

                    if pb_len > block_len and pb_len > block_threshold:
                        block_list[bidx] = (pb_label, pb_lidx, ridx)
                        block_list.remove(prev_block)
                        all_clean = False
                        break

                # Interior block
                else:
                    prev_block = block_list[bidx - 1]
                    next_block = block_list[bidx + 1]
                    pb_label, pb_lidx, pb_ridx = prev_block
                    nb_label, nb_lidx, nb_ridx = next_block
                    pb_len = pb_ridx - pb_lidx + 1
                    nb_len = nb_ridx - nb_lidx + 1

                    if pb_label == nb_label and (pb_len > block_threshold or nb_len > block_threshold):
                        block_list[bidx] = (pb_label, pb_lidx, nb_ridx)
                        block_list.remove(prev_block)
                        block_list.remove(next_block)
                        all_clean = False
                        break

        # Traversed block list with no actions required
        if all_clean:
            break

    # Now back-translate the block list to the list
    out_arr = np.copy(event_arr)

    for blabel, lidx, ridx in block_list:
        out_arr[lidx:ridx + 1] = blabel

    # Additionally build an array where the two arrays are different
    diff_arr = (out_arr != event_arr).astype(bool)

    return (out_arr, diff_arr)


def gen_circ_failure_ep(map_col=None, lactate_col=None, milri_col=None, dobut_col=None,
                        levosi_col=None, theo_col=None, noreph_col=None, epineph_col=None, vaso_col=None):
    """ Circulatory failure endpoint definition
    
    INPUTS:
    event_status_arr: Event status array to be filled with circ. failure level
    map_col: Mean arterial pressure imputed values at each time-point
    lactate_col: Lactate imputed values at each time-point
    milri_col: Milrinone drug imputed values at each time-point
    dobut_col: Dobutamine drug imputed values at each time-point
    levosi_col: Levosimendan drug imputed values at each time-point
    theo_col: Theophyllin drug imputed values at each time-point
    noreph_col: Norepinephrine drug imputed values at each time-point
    epineph_col: Epinephrine drug imputed values at each time-point
    vaso_col: Vasopressin drug imputed values at each time-point

    RETURNS: Binary indicator array if the patient was in circulatory failure (=1) or not (=0)
    """
    circ_status_arr = np.zeros_like(map_col)
    n_steps = len(circ_status_arr)
    # Computation of the circulatory failure toy version of the endpoint
    for jdx in range(0, n_steps):
        sym_window = EVENT_SEARCH_WINDOW // 2  # We center window search for circ
        low_idx = max(0, jdx - sym_window)
        high_idx = min(jdx + sym_window, n_steps)
        map_subarr = map_col[low_idx:high_idx]
        lact_subarr = lactate_col[low_idx:high_idx]
        milri_subarr = milri_col[low_idx:high_idx]
        dobut_subarr = dobut_col[low_idx:high_idx]
        levosi_subarr = levosi_col[low_idx:high_idx]
        theo_subarr = theo_col[low_idx:high_idx]
        noreph_subarr = noreph_col[low_idx:high_idx]
        epineph_subarr = epineph_col[low_idx:high_idx]
        vaso_subarr = vaso_col[low_idx:high_idx]
        map_crit = (map_subarr < 65)
        drug_crit = (milri_subarr > 0) | (dobut_subarr > 0) | (levosi_subarr > 0) | (theo_subarr > 0) | (
                noreph_subarr > 0) | (epineph_subarr > 0) | (vaso_subarr > 0)
        map_crit_arr = (map_crit | drug_crit)
        lact_crit_arr = (lact_subarr > 2)
        map_condition = np.sum(map_crit_arr) >= FRACTION_TSH_CIRC * len(map_crit_arr)
        lact_condition = np.sum(lact_crit_arr) >= FRACTION_TSH_CIRC * len(map_crit_arr)
        if map_condition and lact_condition:
            circ_status_arr[jdx] = 1.0

    return circ_status_arr


def compute_pao2(current_idx, pao2_col, pao2_meas_cnt, spo2_col, spo2_meas_cnt, search_window):
    ''' Estimate the current PaO2 value

    INPUTS:
    current_idx: Index of array at which PaO2 value is estimated
    pao2_col: PaO2 pre-imputed column
    pao2_meas_cnt: Cumulative measurement counts of PaO2 at each grid-point
    spo2_col: SpO2 pre-imputed column
    spo2_meas_cnt: Cumulative measurement counts of SpO2 at each grid-point
    search_window: Backward search window to find real PaO2 measurements in

    RETURNS: PaO2 value estimated at <current_idx> and a boolean indicating whether the
             estimated from a real PaO2 measurement close in time.
    '''

    # Estimate the current PaO2 value
    bw_pao2_meas = pao2_meas_cnt[max(0, current_idx - search_window):current_idx + 1]
    bw_pao2 = pao2_col[max(0, current_idx - search_window):current_idx + 1]
    pao2_meas = bw_pao2_meas[-1] - bw_pao2_meas[0] >= 1
    pao2_avail = 0

    # PaO2 was just measured, just use the value
    if pao2_meas:
        pao2_estimate = bw_pao2[-1]
        pao2_avail = 1

    # Have to forecast PaO2 from a previous SpO2
    else:
        bw_spo2 = spo2_col[max(0, current_idx - ABGA_WINDOW):current_idx + 1]
        bw_spo2_meas = spo2_meas_cnt[max(0, current_idx - ABGA_WINDOW):current_idx + 1]
        spo2_meas = bw_spo2_meas[-1] - bw_spo2_meas[0] >= 1

        # Standard case, take the last SpO2 measurement
        if spo2_meas:
            spo2_val = bw_spo2[-1]
            pao2_estimate = ellis(np.array([spo2_val]))[0]

        # Extreme edge case, there was no SpO2 measurement in the last 24 hours
        else:
            spo2_val = SPO2_NORMAL_VALUE
            pao2_estimate = ellis(np.array([spo2_val]))[0]

    return pao2_estimate, pao2_avail


def compute_fio2(current_idx, current_time, suppox_idx, suppox_time, fio2_col, fio2_meas_cnt,
                 vent_mode_col, vent_status_col, suppox_col, search_window):
    ''' Estimate the current FiO2 value at a grid-point

    INPUTS:
    current_idx: Time-grid index at which FiO2 should be estimated
    current_time: Absolute time corresponding to grid-point
    suppox_idx: Current measurement index in supplementary oxygen meas. array synced to <current_time>
    suppox_time: Absolute time corresponding to current SuppOx measurement
    fio2_col: Pre-imputed column of FiO2 values
    fio2_meas_cnt: Cumulative measurement count of FiO2 measurements
    vent_mode_col: Pre-imputed ventilation mode column
    vent_status_col: Estimated ventilation state at each grid-point
    suppox_col: Pre-imputed column of supplementary oxygen measurements
    search_window: Length of search window in which to look for recent FiO2 measurements

    RETURNS: FiO2 value estimated at <current_idx>, and 3 status indicators which are mutually exclusive
             and indicate which estimation mode was used to produce the estimate
    '''

    # Estimate the current FiO2 value
    bw_fio2 = fio2_col[max(0, current_idx - search_window):current_idx + 1]
    bw_fio2_meas = fio2_meas_cnt[max(0, current_idx - search_window):current_idx + 1]
    fio2_meas = bw_fio2_meas[-1] - bw_fio2_meas[0] > 0

    mode_group_est = vent_mode_col[current_idx]
    fio2_avail = 0
    fio2_ambient = 0
    fio2_suppox = 0
    # FiO2 is measured since beginning of stay and EtCO2 was measured, we use FiO2 (indefinite forward filling)
    # if ventilation is active or the current estimate of ventilation mode group is NIV.
    if fio2_meas and (vent_status_col[current_idx] == 1.0 or mode_group_est == NIV_VENT_MODE):
        fio2_val = bw_fio2[-1] / 100
        fio2_avail = 1

    # No real measurements up to now, or the last real measurement
    # was more than 8 hours away.
    # Use supplemental oxygen or ambient air oxygen
    else:

        # No suppox measurment in the max_ffil period or before current timestep because it the first timestep
        if suppox_idx == -1 or (current_time - suppox_time) > np.timedelta64(SUPPOX_MAX_FFILL, 'h'):
            fio2_val = AMBIENT_FIO2
            fio2_ambient = 1

        # Find the most recent source variable of SuppOx
        else:
            suppox = suppox_col[suppox_idx]

            # SuppOx information from main source
            if np.isfinite(suppox):
                fio2_val = suppox_to_fio2(int(suppox)) / 100
                fio2_suppox = 1
            else:
                raise Exception("SuppOx has to be finite")

    return fio2_val, fio2_avail, fio2_ambient, fio2_suppox


# NEEDS TEST
def compute_vent_status(etco2_col, etco2_meas_cnt, peep_col, peep_meas_cnt,
                        hr_meas_cnt, vent_mode_col, tv_col, airway_col, peep_search_window, hr_search_window,
                        vent_vote_threshold, peep_threshold):
    ''' Compute the ventilation status (patient was ventilated or not at a grid point) based on
        a ventilation voting mechanism taking different factors into account

    INPUTS:
    etco2_col: Pre-imputed EtCO2 column
    etco2_meas_cnt: Cumulative measurement count for the EtCo2 measurements
    peep_col: Pre-imputed PEEP column
    peep_meas_cnt: Cumulative measurement count for the PEEP measurements
    hr_meas_cnt: Cumulative measurement count for the HR measurements
    vent_mode_col: Pre-imputed ventilation mode column
    tv_col: Pre-imputed TV column
    airway_col: Pre-imputed airway presence columns
    peep_search_window: Window size in which to search for PEEP measurements
    hr_search_window: Window size in which to search for HR measurements
    vent_vote_threshold: Threshold to decide whether ventilation presence should be estimated
    peep_threshold: PEEP threshold to use for the point-wise PEEP condition

    RETURNS: A binary array indicating ventilation status at each time-point, and auxiliary arrays giving the HR
             measurement status, and PEEP measurent status as well as indication whether the PEEP threshold
             was crossed at a time-point
    '''

    n_steps = len(etco2_col)
    vent_status = np.zeros(n_steps)
    peep_status = np.zeros(n_steps)
    peep_threshold_status = np.zeros(n_steps)
    hr_status = np.zeros(n_steps)

    for jdx in range(0, n_steps):
        low_vent_idx = max(0, jdx - peep_search_window)
        high_vent_idx = min(n_steps, jdx + peep_search_window)
        low_peep_idx = max(0, jdx - peep_search_window)
        high_peep_idx = min(n_steps, jdx + peep_search_window)
        low_hr_idx = max(0, jdx - hr_search_window)
        high_hr_idx = min(n_steps, jdx + hr_search_window)

        win_etco2 = etco2_col[low_vent_idx:high_vent_idx]
        win_etco2_meas = etco2_meas_cnt[low_vent_idx:high_vent_idx]
        win_peep = peep_col[low_peep_idx:high_peep_idx]
        win_peep_meas = peep_meas_cnt[low_peep_idx:high_peep_idx]
        win_hr_meas = hr_meas_cnt[low_hr_idx:high_hr_idx]

        etco2_meas_win = win_etco2_meas[-1] - win_etco2_meas[0] > 0
        peep_meas_win = win_peep_meas[-1] - win_peep_meas[0] > 0
        hr_meas_win = win_hr_meas[-1] - win_hr_meas[0] > 0
        current_vent_group = vent_mode_col[jdx]
        current_tv = tv_col[jdx]
        current_airway = airway_col[jdx]
        vote_score = 0

        # EtCO2 requirement (still needed)
        if etco2_meas_win and (win_etco2 > 0.5).any():
            vote_score += 2

        # Ventilation group requirement (still needed)
        if current_vent_group in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
            vote_score += 1
        elif current_vent_group in [1.0]:
            vote_score -= 1
        elif current_vent_group in [11.0, 12.0, 13.0, 15.0, 17.0]:
            vote_score -= 2

        # TV presence requirement (still needed)
        if current_tv > 0:
            vote_score += 1

        # Airway requirement (still needed)
        if current_airway in [1, 2]:
            vote_score += 2

        # No airway (still needed)
        if current_airway in [3, 4, 5, 6]:
            vote_score -= 1

        if vote_score >= vent_vote_threshold:
            vent_status[jdx] = 1

        if peep_meas_win:
            peep_status[jdx] = 1
        if np.any(win_peep >= peep_threshold):
            peep_threshold_status[jdx] = 1
        if hr_meas_win:
            hr_status[jdx] = 1

    return vent_status, peep_status, peep_threshold_status, hr_status


# NEEDS TEST
def compute_pao2_fio2_estimates(abs_dtime_arr=None, suppox_dtime_arr=None, fio2_col=None, fio2_meas_cnt=None,
                                pao2_col=None, pao2_meas_cnt=None, spo2_col=None, spo2_meas_cnt=None, suppox_col=None,
                                vent_mode_col=None, vent_status_col=None, sz_fio2_window=None,
                                sz_pao2_window=None):
    """ Compute the PaO2 and FiO2 estimates at a particular time-point in the stay

    INPUTS: 
    ratio_arr: For length computation
    abs_dtime_arr: Absolute date time in stay
    suppox_async_red_ts: Time stamps of where asynchronuous supplementary oxygen measurements become available
    fio2_col: FiO2 imputed values
    fio2_meas_cnt: Cumulative number of real FiO2 measurements since beginning of the stay
    pao2_meas_cnt: Cumulative number of real PaO2 measurements since beginning ot the stay
    pao2_col: PaO2 imputed values
    spo2_col: SpO2 imputed values
    spo2_meas_cnt: Cumulative number of real SpO2 measurements since beginning of the stay
    pao2_est_arr: Array to be filled with PaO2 estimates (OUTPUT)
    fio2_est_arr: Array to be filled with FiO2 estimates (OUTPUT)
    vent_mode_col: Ventilation mode codes at each timepoint
    vent_status_arr: Ventilation status at each time-point
    event_count: Event count dictionary for debugging purposes
    fio2_avail_arr: Binary indicator of whether available FiO2 was used for estimation (OUTPUT)
    suppox_max_ffill: Maximum forward filling time of Supplementary Oxygen measurement 
    ambient_fio2: FiO2 value to default to in an ambient air assumption
    fio2_ambient_arr: Binary indicator of whether ambient air FiO2 was used for estimation (OUTPUT)
    suppox_val: Supplementary oxygen raw values 
    fio2_suppox_arr: Binary indicator of whether Suppox was used for FiO2 estimation (OUTPUT)
    sz_fio2_window: Size of the FiO2 window to use for searching measurements
    sz_pao2_window: Size of the PaO2 window to use for searching measurements
    pao2_avail_col: Availability of a close PaO2 measurement at a time-point

    RETURNS: Estimated PaO2 / FiO2 values at a time-point, and the 3 status arrays of the way FiO2 
             was estimated at a particular time-point.
    """

    # Array pointers tracking the current active value of each type
    suppox_async_red_ptr = -1
    fio2_avail_arr = np.zeros_like(fio2_col)
    fio2_ambient_arr = np.zeros_like(fio2_col)
    fio2_suppox_arr = np.zeros_like(fio2_col)
    pao2_avail_arr = np.zeros_like(fio2_col)
    pao2_est_arr = np.zeros_like(fio2_col)
    fio2_est_arr = np.zeros_like(fio2_col)

    # Label each point in the 30 minute window (except ventilation)
    for jdx in range(0, len(abs_dtime_arr)):
        cur_time = abs_dtime_arr[jdx]

        while True:
            suppox_async_red_ptr = suppox_async_red_ptr + 1
            if suppox_async_red_ptr >= len(suppox_dtime_arr) or suppox_dtime_arr[suppox_async_red_ptr] > cur_time:
                suppox_async_red_ptr = suppox_async_red_ptr - 1
                break

        if suppox_async_red_ptr >= 0:
            suppox_time = suppox_dtime_arr[suppox_async_red_ptr]
        else:
            suppox_time = None

        # Estimate the current FiO2 value
        fio2_val, fio2_avail, fio2_ambient, fio2_suppox = compute_fio2(jdx, cur_time, suppox_async_red_ptr, suppox_time,
                                                                       fio2_col, fio2_meas_cnt, vent_mode_col,
                                                                       vent_status_col, suppox_col, sz_fio2_window)
        fio2_avail_arr[jdx] = fio2_avail
        fio2_ambient_arr[jdx] = fio2_ambient
        fio2_suppox_arr[jdx] = fio2_suppox

        pao2_val, pao2_avail = compute_pao2(jdx, pao2_col, pao2_meas_cnt, spo2_col, spo2_meas_cnt, sz_pao2_window)
        pao2_avail_arr[jdx] = pao2_avail

        pao2_est_arr[jdx] = pao2_val
        fio2_est_arr[jdx] = fio2_val

    out_dict = {}
    out_dict["pao2_est"] = pao2_est_arr
    out_dict["fio2_est"] = fio2_est_arr
    out_dict["fio2_avail"] = fio2_avail_arr
    out_dict["fio2_suppox"] = fio2_suppox_arr
    out_dict["fio2_ambient"] = fio2_ambient_arr
    out_dict["pao2_avail"] = pao2_avail_arr

    return out_dict


def load_relevant_columns(df_pid, var_map):
    """ Loads the relevant columns from the patient stay
    
    INPUTS: 
    df_pid: Pandas data-frame of the imputed values for a patient
    var_map: Map of the meta-variable IDs of particular channels

    RETURNS: Dictionary with relevant channel columns
    """
    pat_cols = {}

    pat_cols["fio2"] = np.array(df_pid[var_map["FiO2"]])
    pat_cols["pao2"] = np.array(df_pid[var_map["PaO2"]])
    pat_cols["etco2"] = np.array(df_pid[var_map["etCO2"]])

    pat_cols["noreph"] = np.array(df_pid[var_map["Norephenephrine"][0]])
    pat_cols["epineph"] = np.array(df_pid[var_map["Epinephrine"][0]])
    pat_cols["vaso"] = np.array(df_pid[var_map["Vasopressin"][0]])

    pat_cols["milri"] = np.array(df_pid[var_map["Milrinone"][0]])
    pat_cols["dobut"] = np.array(df_pid[var_map["Dobutamine"][0]])
    pat_cols["levosi"] = np.array(df_pid[var_map["Levosimendan"][0]])
    pat_cols["theo"] = np.array(df_pid[var_map["Theophyllin"][0]])

    pat_cols["lactate"] = np.array(df_pid[var_map["Lactate"][0]])
    pat_cols["peep"] = np.array(df_pid[var_map["PEEP"]])

    # Heartrate
    pat_cols["hr_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["HR"])])

    pat_cols["tv"] = np.array(df_pid[var_map["TV"]])
    pat_cols["map"] = np.array(df_pid[var_map["MAP"][0]])
    pat_cols["airway"] = np.array(df_pid[var_map["Airway"]])

    # Ventilator mode group columns
    pat_cols["vent_mode"] = np.array(df_pid[var_map["vent_mode"]])

    pat_cols["spo2"] = np.array(df_pid[var_map["SpO2"]])

    pat_cols["fio2_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["FiO2"])])
    pat_cols["pao2_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["PaO2"])])
    pat_cols["etco2_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["etCO2"])])
    pat_cols["peep_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["PEEP"])])
    pat_cols["hr_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["HR"])])
    pat_cols["spo2_meas_cnt"] = np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["SpO2"])])

    # Absolute time
    pat_cols["abs_dtime"] = np.array(df_pid[DATETIME])

    return pat_cols


def suppox_to_fio2(suppox_val):
    """ Conversion of supplemental oxygen to FiO2 estimated value

    INPUTS: 
    suppox_val: Supplementary oxygen values

    RETURNS: Estimated FiO2 values at time-points
    """
    if suppox_val > MAX_SUPPOX_KEY:
        return MAX_SUPPOX_TO_FIO2_VAL
    else:
        return SUPPOX_TO_FIO2[suppox_val]


def assemble_out_df(time_col=None, rel_time_col=None, pid_col=None, event_status_arr=None,
                    relabel_arr=None, fio2_avail_arr=None, fio2_suppox_arr=None,
                    fio2_ambient_arr=None, fio2_est_arr=None, pao2_est_arr=None,
                    pao2_avail_arr=None, ratio_arr=None, vent_status_arr=None, circ_status_arr=None):
    """ Assembles the complete data-frame from the constructed endpoint and status arrays

    INPUTS:
    time_col: Absolute time column
    rel_time_col: Relative time column
    pid_col: Column with patient ID
    event_status_arr: Respiratory event levels estimated at each time-point
    relabel_arr: Binary indicator if a position was relabeled with a post-processing step, for the resp
                 status estimation.
    fio2_avail_arr: Indicator whether FiO2 measurement was available at a time-point
    fio2_suppox_arr: Indicator whether FiO2 was estimated from supplementary oxygen at a time-point
    fio2_ambient_arr: Indicator whether ambient air assumption was used at a time-point
    fio2_est_arr: FiO2 value estimated at each time-point
    pao2_est_arr: PaO2 value estimated at each time-point
    pao2_sur_est: TBD
    pao2_avail_arr: Binary indicator whether a real PaO2 measurement was available close to time-point
    pao2_sao2_model_arr: Binary indicator whether SaO2 model was used to estimate PaO2 at a time-point
    pao2_full_model_arr: Binary indicator whether full model was used to estimate PaO2 at a time-point
    ratio_arr: P/F ratios estimated at each time-point
    sur_ratio_arr: TBD
    vent_status_arr: Presence of ventilation at each time-point
    vent_period_arr: More precise post-processed version of ventilation annotation
    vent_votes_arr: Ventilation detection is based on a voting algorithm based on several factors, number
                    of points associated with each time-point
    vent_votes_etco2_arr: Ventilation votes for the ETCO2 condition
    vent_votes_ventgroup_arr: Ventilation votes for the ventilation code group condition
    vent_votes_tv_arr: Ventilation votes for the TV condition
    vent_votes_airway_arr: Ventilation votes for the airway condition
    circ_status_arr: Circulatory failure level annotated at each time-point

    RETURNS: Complete endpoint data-frame for a patient
    """
    df_out_dict = {}

    df_out_dict[DATETIME] = time_col
    df_out_dict[REL_DATETIME] = rel_time_col
    df_out_dict[PID] = pid_col
    status_list = list(map(lambda raw_str: raw_str.decode("unicode_escape"), event_status_arr.tolist()))
    df_out_dict["resp_failure_status"] = status_list
    df_out_dict["resp_failure_status_relabel"] = relabel_arr

    # Status columns
    df_out_dict["fio2_available"] = fio2_avail_arr
    df_out_dict["fio2_suppox"] = fio2_suppox_arr
    df_out_dict["fio2_ambient"] = fio2_ambient_arr
    df_out_dict["fio2_estimated"] = fio2_est_arr
    df_out_dict["pao2_estimated"] = pao2_est_arr

    df_out_dict["pao2_available"] = pao2_avail_arr
    df_out_dict["estimated_ratio"] = ratio_arr
    df_out_dict["vent_state"] = vent_status_arr

    # Circulatory failure related
    df_out_dict["circ_failure_status"] = circ_status_arr
    df_out = pd.DataFrame(df_out_dict)

    return df_out


def endpoint_gen_benchmark(batch_id, endpoint_path, imputed_path, merged_path):
    """ Endpoint generation function for one batch of patients

    INPUTS:
    configs: Configuration file dictionary
    """

    var_map = VAR_IDS_EP

    if not os.path.exists(endpoint_path):
        os.mkdir(endpoint_path)

    logging.info("Generating endpoints for batch {}".format(batch_id))
    batch_fpath = os.path.join(imputed_path, "batch_{}.parquet".format(batch_id))

    if not os.path.exists(batch_fpath):
        logging.info("WARNING: Input file does not exist, exiting...")
        sys.exit(1)

    df_batch = pd.read_parquet(os.path.join(imputed_path, "batch_{}.parquet".format(batch_id)))

    logging.info("Loaded imputed data done...")

    cand_raw_batch = glob.glob(os.path.join(merged_path, "part-{}.parquet".format(batch_id)))
    assert (len(cand_raw_batch) == 1)
    pids = list(df_batch[PID].unique())

    logging.info("Number of patients in batch: {}".format(len(df_batch[PID].unique())))
    out_fp = os.path.join(endpoint_path, "batch_{}.parquet".format(batch_id))

    out_dfs = []

    for pidx, pid in enumerate(pids):

        print("Patient {}/{}".format(pidx + 1, len(pids)))

        df_pid = df_batch[df_batch[PID] == pid]

        if df_pid.shape[0] == 0:
            logging.info("WARNING: No input data for PID: {}".format(pid))
            continue

        df_merged_pid = pd.read_parquet(cand_raw_batch[0], filters=[(PID, "=", pid)])
        df_merged_pid.sort_values(by=DATETIME, inplace=True)

        # Main route of SuppOx
        df_suppox_red_async = df_merged_pid[[var_map["SuppOx"], DATETIME]]
        df_suppox_red_async = df_suppox_red_async.dropna(how="all", thresh=2)
        suppox_async_red_ts = np.array(df_suppox_red_async[DATETIME])

        suppox_col = np.array(df_suppox_red_async[var_map["SuppOx"]])

        # Strategy is to create an imputed SuppOx column based on the spec using
        # forward filling heuristics

        # Load patient columns from data-frame
        pat_cols = load_relevant_columns(df_pid, var_map)

        spo2_col = percentile_smooth(pat_cols["spo2"], SPO2_PERCENTILE,
                                     SPO2_MIN_WINDOW)

        # Label each point in the 30 minute window with ventilation
        vent_status_arr, peep_status, peep_threshold_status, hr_status = compute_vent_status(pat_cols['etco2'],
                                                                                             pat_cols['etco2_meas_cnt'],
                                                                                             pat_cols['peep'],
                                                                                             pat_cols['peep_meas_cnt'],
                                                                                             pat_cols['hr_meas_cnt'],
                                                                                             pat_cols['vent_mode'],
                                                                                             pat_cols['tv'],
                                                                                             pat_cols['airway'],
                                                                                             PEEP_SEARCH_WINDOW,
                                                                                             HR_SEARCH_WINDOW,
                                                                                             VENT_VOTE_TSH,
                                                                                             PEEP_TSH)

        vent_status_arr = delete_low_density_hr_gap(vent_status_arr, hr_status)

        vent_status_arr = merge_short_vent_gaps(vent_status_arr, SHORT_GAP_TSH)

        vent_status_arr = delete_short_vent_events(vent_status_arr, SHORT_EVENT_TSH)

        # Estimate the FiO2/PaO2 values at indiviual time points
        est_out_dict = compute_pao2_fio2_estimates(abs_dtime_arr=pat_cols["abs_dtime"],
                                                   suppox_dtime_arr=suppox_async_red_ts,
                                                   fio2_col=pat_cols["fio2"], pao2_col=pat_cols["pao2"],
                                                   spo2_col=spo2_col,
                                                   fio2_meas_cnt=pat_cols["fio2_meas_cnt"],
                                                   pao2_meas_cnt=pat_cols["pao2_meas_cnt"],
                                                   spo2_meas_cnt=pat_cols["spo2_meas_cnt"],
                                                   vent_mode_col=pat_cols["vent_mode"], vent_status_col=vent_status_arr,
                                                   suppox_col=suppox_col,
                                                   sz_fio2_window=FI02_SEARCH_WINDOW,
                                                   sz_pao2_window=PA02_SEARCH_WINDOW)

        pao2_est_arr = est_out_dict["pao2_est"]
        fio2_est_arr = est_out_dict["fio2_est"]
        fio2_avail_arr = est_out_dict["fio2_avail"]
        pao2_avail_arr = est_out_dict["pao2_avail"]
        fio2_suppox_arr = est_out_dict["fio2_suppox"]
        fio2_ambient_arr = est_out_dict["fio2_ambient"]

        # Smooth individual components of the P/F ratio estimate
        pao2_est_arr = kernel_smooth_arr(pao2_est_arr, bandwidth=PAO2_BW)
        # Convex combination of the estimate
        pao2_est_arr = mix_real_est_pao2(pat_cols["pao2"], pat_cols["pao2_meas_cnt"], pao2_est_arr)

        ratio_arr = np.divide(pao2_est_arr, fio2_est_arr)

        resp_status_arr = assign_resp_levels(pf_event_est_arr=ratio_arr,
                                             vent_status_arr=vent_status_arr,
                                             peep_status_arr=peep_status,
                                             sz_window=EVENT_SEARCH_WINDOW,
                                             peep_threshold_arr=peep_threshold_status,
                                             offset_back_windows=OFFSET_RESP)
        # Re-traverse the array and correct the right edges of events
        resp_status_arr = correct_right_edge_l0(event_status_arr=resp_status_arr, pf_event_est_arr=ratio_arr,
                                                offset_back_windows=OFFSET_RESP)
        resp_status_arr = correct_right_edge_l1(event_status_arr=resp_status_arr, pf_event_est_arr=ratio_arr,
                                                offset_back_windows=OFFSET_RESP)
        resp_status_arr = correct_right_edge_l2(event_status_arr=resp_status_arr, pf_event_est_arr=ratio_arr,
                                                offset_back_windows=OFFSET_RESP)
        resp_status_arr = correct_right_edge_l3(event_status_arr=resp_status_arr, pf_event_est_arr=ratio_arr,
                                                offset_back_windows=OFFSET_RESP)

        # Traverse the array and delete short gap
        resp_status_arr, relabel_arr = delete_small_continuous_blocks(resp_status_arr,
                                                                      block_threshold=PF_MERGE_THRESHOLD)

        circ_status_arr = gen_circ_failure_ep(map_col=pat_cols["map"], lactate_col=pat_cols["lactate"],
                                              milri_col=pat_cols["milri"], dobut_col=pat_cols["dobut"],
                                              levosi_col=pat_cols["levosi"], theo_col=pat_cols["theo"],
                                              noreph_col=pat_cols["noreph"], epineph_col=pat_cols["epineph"],
                                              vaso_col=pat_cols["vaso"])

        time_col = np.array(df_pid[DATETIME])
        rel_time_col = np.array(df_pid[REL_DATETIME])
        pid_col = np.array(df_pid[PID])

        df_out = assemble_out_df(time_col=time_col, rel_time_col=rel_time_col, pid_col=pid_col,
                                 event_status_arr=resp_status_arr, relabel_arr=relabel_arr,
                                 fio2_avail_arr=fio2_avail_arr, fio2_suppox_arr=fio2_suppox_arr,
                                 fio2_ambient_arr=fio2_ambient_arr, fio2_est_arr=fio2_est_arr,
                                 pao2_est_arr=pao2_est_arr, pao2_avail_arr=pao2_avail_arr, ratio_arr=ratio_arr,
                                 vent_status_arr=vent_status_arr,
                                 circ_status_arr=circ_status_arr)
        out_dfs.append(df_out)

    all_df = pd.concat(out_dfs, axis=0)
    all_df.to_parquet(out_fp)
