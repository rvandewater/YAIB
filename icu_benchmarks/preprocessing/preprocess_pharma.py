import gc

import numpy as np
import pandas as pd

from icu_benchmarks.common.constants import PID, DATETIME, INSTANTANEOUS_STATE, START_STATE, STOP_STATE, \
    HR_VARID, PHARMAID, INFID, PHARMA_DATETIME, PHARMA_RATE, PHARMA_VAL, VARID, VALUE, SHORT_GAP, PHARMA_STATUS

def drop_duplicates_pharma(df):
    """
    df: long-format dataframe of a patient
    varref: variable reference table that contain the mean and standard deviation of values for a subset of variables
    """
    df_dup = df[df.duplicated([PHARMA_DATETIME, PHARMAID, INFID], keep=False)]
    for pharmaid in df_dup[PHARMAID].unique():
        for infusionid in df_dup[df_dup[PHARMAID] == pharmaid][INFID].unique():
            tmp = df_dup[(df_dup[PHARMAID] == pharmaid) & (df_dup[INFID] == infusionid)]
            if len(tmp[PHARMA_STATUS].unique()) == 1 and tmp[PHARMA_STATUS].unique()[0] == INSTANTANEOUS_STATE:
                for i in range(len(tmp)):
                    df.loc[tmp.index[i], INFID] = "%s_%s" % (int(df.loc[tmp.index[i], INFID]), i)
                # tmp = df[(df[PHARMAID] == pharmaid) & (
                #     df[INFID].apply(lambda x: "%s_" % (infusionid) in x if type(x) == str else False))]
            elif len(tmp[PHARMA_STATUS].unique()) == 1 and tmp[PHARMA_STATUS].unique()[0] == STOP_STATE:
                if (tmp[PHARMA_VAL] != 0).sum() == 1:
                    df.drop(tmp.index[tmp[PHARMA_VAL] == 0], inplace=True)
                else:
                    df.drop(tmp.index[:-1], inplace=True)
            elif len(tmp[PHARMA_STATUS].unique()) == 2 and STOP_STATE in tmp[PHARMA_STATUS].unique():
                df.drop(tmp.index[tmp[PHARMA_STATUS] != STOP_STATE], inplace=True)
            else:
                raise Exception("Debug needed")
    return df


def process_instantaneous_state(df, acting_period):
    '''
    Convert the infusion channel with status injection/tablet to "infusion-like" channel.
    '''
    infusionid = int(df.iloc[0][INFID])

    df.set_index(PHARMA_DATETIME, inplace=True)
    drug_giventime_INSTANTANEOUS_STATE = df.index.tolist()

    df_new = []
    for i, dt in enumerate(drug_giventime_INSTANTANEOUS_STATE):
        tmp = df.loc[[dt]].copy()

        endtime_instantaneous_drug = dt + np.timedelta64(acting_period, "m")
        tmp.loc[endtime_instantaneous_drug, PHARMA_VAL] = tmp.loc[dt, PHARMA_VAL]
        tmp.loc[endtime_instantaneous_drug, PHARMA_STATUS] = STOP_STATE
        tmp.loc[endtime_instantaneous_drug, INFID] = "%d_%d" % (infusionid, i)

        tmp.loc[dt, PHARMA_VAL] = 0
        tmp.loc[dt, PHARMA_STATUS] = START_STATE
        tmp.loc[dt, INFID] = "%d_%d" % (infusionid, i)

        df_new.append(tmp.reset_index())
    df_new = pd.concat(df_new).sort_values(PHARMA_DATETIME)
    return df_new


def process_single_infusion(df, acting_period):
    '''
    Convert given dose from a single infusion channel to rate
    '''
    infusionid = int(df.iloc[0][INFID])
    if len(df[PHARMA_STATUS].unique()) == 1 and df[PHARMA_STATUS].unique()[0] == INSTANTANEOUS_STATE:
        df = process_instantaneous_state(df, acting_period)

    df_rate = []
    for sub_infusionid in df[INFID].unique():
        tmp = df[df[INFID] == sub_infusionid].copy()
        try:
            assert ((tmp[PHARMA_STATUS] == START_STATE).sum() == 1)
        except AssertionError:
            tmp.set_index(PHARMA_DATETIME, inplace=True)
            beg_time = tmp.index[0] - np.timedelta64(acting_period, "m")
            tmp.loc[beg_time, PHARMA_VAL] = 0
            tmp.loc[beg_time, PHARMA_STATUS] = START_STATE
            tmp.loc[beg_time, INFID] = sub_infusionid
            tmp.sort_index(inplace=True)
            tmp.reset_index(inplace=True)
        try:
            assert ((tmp[PHARMA_STATUS] == STOP_STATE).sum() == 1)
        except AssertionError:
            pass
        tmp.loc[:, PHARMA_RATE] = 0
        tmp.loc[tmp.index[:-1], PHARMA_RATE] = tmp[PHARMA_VAL].values[1:] / (tmp[PHARMA_DATETIME].diff() / np.timedelta64(1,
                                                                                                          "m")).values[
                                                                     1:]
        tmp.rename(columns={PHARMA_RATE: str(sub_infusionid)}, inplace=True)
        df_rate.append(tmp[[PHARMA_DATETIME, str(sub_infusionid)]].set_index(PHARMA_DATETIME))
    df_rate = pd.concat(df_rate, axis=1).sum(axis=1).to_frame(name=str(infusionid))
    return df_rate


def convert_cumul_value_to_rate(df, cumul_urine_id_lst, general_table):
    pid = df.iloc[0][PID]

    rec_adm_time = general_table.loc[pid].admissiontime
    # if the first HR measuremet time is earlier than recorded admission time, then we estimated
    # the "true" admission time to be the earlier of these two time points.
    if df[df[VARID] == HR_VARID][VALUE].notnull().sum() > 0:
        hr_first_meas_time = df.loc[df[df[VARID] == HR_VARID][VALUE].notnull().index[0], DATETIME]
        esti_adm_time = min(rec_adm_time, hr_first_meas_time)
    else:
        esti_adm_time = rec_adm_time

    df_urine = df[df[VARID].isin(cumul_urine_id_lst)]

    if len(df_urine) == 0:
        return df
    else:
        for vid in df_urine[VARID].unique():
            df_tmp = df_urine[df_urine[VARID] == vid]  # table of a single urine variable

            index_pre_general_table = df_tmp.index[df_tmp[DATETIME] < esti_adm_time - np.timedelta64(15 * 60 + 30,
                                                                                                     "s")]
            # number of records before general_admission time
            if len(index_pre_general_table) == 0:
                pass
            elif len(index_pre_general_table) == 1:
                # if there's one record before general_admission,
                # reset datetime from system reset time 12pm to the general_admission time
                index_pre_general_table = df_tmp.index[df_tmp[DATETIME] < esti_adm_time]
                df.loc[index_pre_general_table[0], DATETIME] = esti_adm_time
            else:
                index_pre_general_table = df_tmp.index[df_tmp[DATETIME] < esti_adm_time]
                df.drop(index_pre_general_table[:-1], inplace=True)
                df.loc[index_pre_general_table[-1], DATETIME] = esti_adm_time

            df_tmp = df[df[VARID] == vid]
            if df_tmp.duplicated([DATETIME]).sum() == 0:
                pass
            else:
                df.drop(df_tmp.index[df_tmp.duplicated([DATETIME])], inplace=True)

            # delete urine record if therre's only one left
            if (df[VARID] == vid).sum() < 2:
                df.drop(df.index[df[VARID] == vid], inplace=True)
                continue

            # compute the cumulative values over the entire icu stay
            df_tmp = df[df[VARID] == vid]
            t_reset = df_tmp[(df_tmp[VALUE].diff() < 0) | (
                    df_tmp.index == df_tmp.index[0])][DATETIME]  # reset time for the cumulative counting
            for i in np.arange(1, len(t_reset)):
                tmp = df_tmp[df_tmp[DATETIME] >= t_reset.iloc[i]]
                if i < len(t_reset) - 1:
                    tmp = tmp[tmp[DATETIME] < t_reset.iloc[i + 1]]
                df.loc[tmp.index, VALUE] += df.loc[df_tmp.index[df_tmp[DATETIME] < t_reset.iloc[i]][-1], VALUE]

            # drop the time point with time difference from the previous time point that is smaller than 5 minute
            df_tmp = df[df[VARID] == vid]
            tdiff = (df_tmp[DATETIME].diff().iloc[1:] / np.timedelta64(1,'h'))
            if (tdiff < SHORT_GAP).sum() > 0:
                df.drop(df_tmp.index[1:][tdiff.values < SHORT_GAP], inplace=True)

            if (df[VARID] == vid).sum() < 2:
                df.drop(df.index[df[VARID] == vid], inplace=True)
                continue

            # debug if the cumulative value is not strictly increasing
            df_tmp = df[df[VARID] == vid]
            vdiff = df_tmp[VALUE].diff()
            try:
                assert ((vdiff < 0).sum() == 0)
            except AssertionError:
                import ipdb
                ipdb.set_trace()
        gc.collect()

        for vid in df_urine[VARID].unique():
            df_tmp = df[df[VARID] == vid]
            if len(df_tmp) == 0:
                continue
            elif len(df_tmp) == 1:
                continue
            else:
                tdiff = (df_tmp[DATETIME].diff() / np.timedelta64(1,'h'))
                df.loc[df_tmp.index[1:], VALUE] = (df_tmp[VALUE].diff().iloc[1:] / tdiff.iloc[1:]).values
                df.loc[df_tmp.index[0], VALUE] = 0

        return df
