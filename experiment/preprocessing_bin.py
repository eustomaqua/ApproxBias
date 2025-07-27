# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds
#

import numpy as np
import pandas as pd
from copy import deepcopy

# from experiment.datasets import (Ricci, German, Adult,
#                                  PropublicaRecidivism,
#                                  PropublicaViolentRecidivism)
# from experiment.datasets import preprocess, adversarial
# from experiment.datasets import (AVAILABLE_FAIR_DATASET,
#                                  DATASETS, DATASET_NAMES)


# ===============================
# Data preprocessing


# -------------------------------
# cf. manf_data.py
# -------------------------------


def binarized_data_set(processed_binsensitive):
    binarized_binsens = deepcopy(processed_binsensitive)
    for u in binarized_binsens.columns:
        if binarized_binsens[u].dtype == bool:
            binarized_binsens[u] = binarized_binsens[u].astype('int')
    return binarized_binsens


def transform_X_A_and_y(dataset, processed_binsensitive):
    y = processed_binsensitive[dataset.label_name]

    sensitive_attrs = deepcopy(dataset.sensitive_attrs)
    if len(sensitive_attrs) > 1:
        new_attr_name = '-'.join(sensitive_attrs)
        # sensitive_attrs += [new_attr_name]
        sensitive_attrs.append(new_attr_name)
    else:
        new_attr_name = None
    A = processed_binsensitive[sensitive_attrs]

    X = processed_binsensitive.drop(columns=dataset.label_name)
    X = X.drop(columns=sensitive_attrs)
    return X, A, y, new_attr_name


def transform_disturb_prime(X, A, y, index, belongs_priv,
                            belongs_priv_with_joint):
    """ X, y: np.ndarray or pd.DataFrame
    belongs_priv           : list, element is a np.ndarray
    belongs_priv_with_joint: list, element is index
    """
    if isinstance(X, pd.DataFrame):
        X_idx = X.iloc[index]
        A_idx = A if (A is None) else A.iloc[index]
        y_idx = y.iloc[index]
    elif isinstance(X, np.ndarray):
        X_idx = X[index]
        A_idx = A if (A is None) else A[index]
        y_idx = y[index]

    not_unpriv = [t[index] for t in belongs_priv]
    if not belongs_priv_with_joint:
        joint_ = belongs_priv_with_joint
    elif len(belongs_priv_with_joint) == 2:  # cannot only > 1
        joint_ = [t[index] for t in belongs_priv_with_joint]
    else:  # if isinstance(belongs_priv_with_joint, list):
        joint_ = np.array(belongs_priv_with_joint)[index].tolist()
    # return X_idx_org, X_idx_qtb, y_idx, not_unpriv, joint_
    return X_idx, A_idx, y_idx, not_unpriv, joint_


# normalisation, via scaler

# def normalise_helper_aux(X_trn, A_trn):
#     nb_inst, nb_feat = X_trn.shape
#     _, nb_attr = A_trn.shape  # nb_sens
#     return nb_inst, nb_feat, nb_attr
#
# def normalise_helper_in_(X_trn, A_trn):
#     XA_trn = np.concatenate([X_trn, A_trn], axis=1)
#     return XA_trn
#
# def normalise_helper_out(XA_trn, nb_feat):
#     X_trn = XA_trn[:, :nb_feat]
#     A_trn = XA_trn[:, nb_feat:]
#     return X_trn, A_trn


def normalise_disturb_prime(scaler, X_trn, A_trn,
                            X_val, A_val, X_tst, A_tst):
    # all are np.ndarray (not pd.DataFrame)
    _, nb_feat = X_trn.shape

    XA_trn = np.concatenate([X_trn, A_trn], axis=1)
    XA_tst = np.concatenate([X_tst, A_tst], axis=1)
    XA_val = []
    # XA_trn = normalise_helper_in_(X_trn, A_trn)
    # XA_tst = normalise_helper_in_(X_tst, A_tst)
    # _, nb_feat, _ = normalise_helper_aux(X_trn, A_trn)

    scaler = scaler.fit(XA_trn)
    XA_trn = scaler.transform(XA_trn)
    XA_tst = scaler.transform(XA_tst)
    X_trn, A_trn = XA_trn[:, :nb_feat], XA_trn[:, nb_feat:]
    X_tst, A_tst = XA_tst[:, :nb_feat], XA_tst[:, nb_feat:]
    # X_trn, A_trn = normalise_helper_out(XA_trn, nb_feat)
    # X_tst, A_tst = normalise_helper_out(XA_tst, nb_feat)

    if len(X_val) > 0:
        XA_val = np.concatenate([X_val, A_val], axis=1)
        # XA_val = normalise_helper_in_(X_val, A_val)
        XA_val = scaler.transform(XA_val)
        # X_val, A_val = normalise_helper_out(XA_val, nb_feat)
        X_val = XA_val[:, :nb_feat]
        A_val = XA_val[:, nb_feat:]

    return scaler, X_trn, A_trn, X_val, A_val, X_tst, A_tst


def normalise_disturb_whole(scaler, X_trn, A_trn):
    _, nb_feat = X_trn.shape
    X_and_A = np.concatenate([X_trn, A_trn], axis=1)
    scaler = scaler.fit(X_and_A)
    X_and_A = scaler.transform(X_and_A)
    X_trn, A_trn = X_and_A[:, :nb_feat], X_and_A[:, nb_feat:]
    return scaler, X_trn, A_trn


# -------------------------------
# cf. mext_data.py
# -------------------------------


# -------------------------------
# -------------------------------
