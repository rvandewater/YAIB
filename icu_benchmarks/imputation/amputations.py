"""This file implements amputation mechanisms (MCAR, MAR (logisitc) and MNAR (logistic)) for missing data generation.
It was inspired from: https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values
Original code: https://github.com/BorisMuzellec/MissingDataOT/blob/master/utils.py
"""

import gin
import torch
import logging
import numpy as np
from scipy import optimize


def MCAR_mask(X, p):
    """
    Missing completely at random mechanism.

    Parameters
    ----------
    X : torch.FloatTensor, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    mask : torch.BoolTensor
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape
    mask = np.zeros((n, d))

    ber = torch.rand(n, d)
    mask = ber < p

    return mask


def BO_mask(X, p):
    """
    Black out missing mechanism. Removes values across dimensions.

    Parameters
    ----------
    X : torch.FloatTensor, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    mask : torch.BoolTensor
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    indices = torch.randperm(n)[: int(n * p)]
    mask = torch.zeros(n, d).bool()
    mask[indices, :] = True

    return mask


def MAR_logistic_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Parameters
    ----------
    X : torch.FloatTensor, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns
    -------
    mask : torch.BoolTensor
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape
    mask = torch.zeros(n, d).bool()

    # number of variables that will have no missing values (at least one variable)
    d_obs = max(int(p_obs * d), 1)
    # number of variables that will have missing values
    d_na = d - d_obs

    # Sample variables that will all be observed, and those with missing values
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    # Other variables will have NA proportions that depend on those observed variables, through a logistic model
    # The parameters of this logistic model are random

    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    # Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


def MNAR_logistic_mask(X, p, p_params=0.3, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.

    Parameters
    ----------
    X : torch.FloatTensor, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).
    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)

    Returns
    -------
    mask : torch.BoolTensor
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape
    mask = torch.zeros(n, d).bool()

    # number of variables used as inputs (at least 1)
    d_params = max(int(p_params * d), 1) if exclude_inputs else d
    # number of variables masked with the logistic model
    d_na = d - d_params if exclude_inputs else d

    # Sample variables that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    # Other variables will have NA proportions selected by a logistic model
    # The parameters of this logistic model are random.

    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    # Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    # If the inputs of the logistic model are excluded from MNAR missingness, mask some
    # values used in the logistic model at random
    # This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p

    return mask


def pick_coeffs(X, idxs_obs=None, idxs_nas=None):
    d_obs = len(idxs_obs)
    d_na = len(idxs_nas)
    coeffs = torch.randn(d_obs, d_na)
    Wx = X[:, idxs_obs].mm(coeffs)
    coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p):
    d_obs, d_na = coeffs.shape
    intercepts = torch.zeros(d_na)
    for j in range(d_na):

        def f(x):
            return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p

        intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts


@gin.configurable("amputation")
def ampute_data(data, mechanism, p_miss, p_obs=0.3):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values.

    Parameters
    ----------
    data : DataFrame
        Data for which missing values will be simulated.
    mechanism : str,
            Indicates the missing-data mechanism to be used. ("MCAR", "MAR" or "MNAR")
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
            If mecha = "MAR" or "MNAR", proportion of variables with *no* missing values
            that will be used for the logistic masking model.

    Returns
    ----------
    imputed_data: DataFrame
        The data with the generated missing values.
    """
    logging.info(f"Applying {mechanism} amputation.")
    X = torch.tensor(data.values.astype(np.float32))

    if mechanism == "MAR":
        mask = MAR_logistic_mask(X, p_miss, p_obs)
    elif mechanism == "MNAR":
        mask = MNAR_logistic_mask(X, p_miss, p_obs)
    elif mechanism == "MCAR":
        mask = MCAR_mask(X, p_miss)
    elif mechanism == "BO":
        mask = BO_mask(X, p_miss)
    else:
        logging.error("Not a valid amputation mechanism. Missing-data mechanisms to be used are MCAR, MAR or MNAR.")

    amputed_data = data.mask(mask)
    return amputed_data, mask
