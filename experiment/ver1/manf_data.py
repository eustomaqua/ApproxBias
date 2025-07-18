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

from hfm.utils.verifiers import DTY_BOL


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


def transform_unpriv_tag(dataset, processed_original,
                         joint=('and', 'or', 'both')):
    assert joint in ['and', 'or', 'both'], "Improper joint-parameter"
    belongs_priv = dataset.find_where_belongs(processed_original)

    if len(belongs_priv) > 1 and joint == 'and':
        belongs_priv_with_joint = np.logical_and(
            belongs_priv[0], belongs_priv[1]).astype(DTY_BOL).tolist()
    elif len(belongs_priv) > 1 and joint == 'or':
        belongs_priv_with_joint = np.logical_or(
            belongs_priv[0], belongs_priv[1]).astype(DTY_BOL).tolist()
    elif len(belongs_priv) > 1 and joint == 'both':
        belongs_priv_with_joint = [
            np.logical_and(belongs_priv[0],
                           belongs_priv[1]).astype(DTY_BOL),
            np.logical_or(belongs_priv[0],
                          belongs_priv[1]).astype(DTY_BOL),
        ]
    else:
        belongs_priv_with_joint = []

    return belongs_priv, belongs_priv_with_joint


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


# Seperate/divide datasets
#
# with only one sensitive attribute
#      multiple sensitive attributes


def sens_attr_divided_set(A, new_attr_name=None):
    # tmp_A = A if new_attr_name is None else A[new_attr_name]
    # `new_attr_name` could be `non-joint-sens-attr-name`
    """
    A (sens_attr): pd.DataFrame
    new_attr_name: str
    """
    tmp_A = A[new_attr_name] if new_attr_name is not None else A
    tmp_A = tmp_A.values.reshape(-1)  # np.ndarray of np.int64
    ele_A = np.unique(tmp_A)          # np.ndarray of np.int64

    idx_A = {}
    for i in ele_A:
        idx_A[i] = tmp_A == i  # np.ndarray of np.bool_
    return ele_A, idx_A


def group_of_disjoint_set(X, A, y, ele_i, idx_i):
    # for ele_i in ele_A:
    #   idx_i = idx_A[ele_i]
    Si_X = X[idx_i]
    Si_A = A[idx_i]
    Si_y = y[idx_i]
    return Si_X, Si_A, Si_y


def group_of_formulated(Si_X, Si_A, Si_y_fx):
    Ti_Xy = deepcopy(Si_X)
    Ti_Xy['label_name'] = Si_y_fx
    return Ti_Xy.values


# -------------------------------
# cf. mext_data.py
# -------------------------------


# -------------------------------
# -------------------------------
