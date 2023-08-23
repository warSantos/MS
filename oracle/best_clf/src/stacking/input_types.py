from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from src.files import read_train_test_meta, load_x_y

def fix_nan_inf_values(a):

    a[np.isnan(a) | np.isinf(a)] = 0

def read_proba_tmk(
        file_train_cls: str,
        file_test_cls: str,
        dir_meta_input: str,
        dataset: str,
        n_folds: int,
        fold_id: int,
        models: List[str]
):
    X_train_cls, _ = load_x_y(file_train_cls, "train")
    X_test_cls, _ = load_x_y(file_test_cls, "test")

    # Reading meta-layer input (classification probabilities)
    X_train_meta, X_test_meta = read_train_test_meta(dir_meta_input, dataset, n_folds, fold_id, models)

    # Concat proba + TFIDF MetaFeat
    X_train = sparse.hstack([X_train_cls, X_train_meta]).tocsr()
    X_test = sparse.hstack([X_test_cls, X_test_meta]).tocsr()

    return X_train, X_test


def read_proba_extra_features(
        dir_meta_input: str,
        dir_mf: str,
        dataset: str,
        n_folds: int,
        fold_id: int,
        models: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    # Reading meta-layer input (classification probabilities)
    X_train_meta, X_test_meta = read_train_test_meta(
        dir_meta_input, dataset, n_folds, fold_id, models)

    # Reading extra features
    X_train_extra = pd.read_pickle(f"{dir_mf}/fold_{fold_id}/{dataset}/train.csv")
    X_test_extra = pd.read_pickle(f"{dir_mf}/fold_{fold_id}/{dataset}/test.csv")

    # Concat proba + Extra Fatures
    X_train = np.hstack([X_train_meta, X_train_extra])
    y_train = np.hstack([X_test_meta, X_test_extra])

    return X_train, y_train

def fwls(X_mf, X_meta):

    R_list = []
    for row in np.arange(X_meta.shape[0]):
        M = X_mf[row].reshape(X_mf.shape[1], 1)
        P = X_meta[row].reshape(X_meta.shape[1], 1)
        F = np.multiply(M, P.T)
        r = F.shape[0]
        c = F.shape[1]
        f = F.reshape(r * c)
        R_list.append(f)
    return np.vstack(R_list)

def read_mfs(
    dir_meta_input: str,
    dir_mf: str,
    dataset: str,
    n_folds: int,
    fold_id: int,
    models: List[str],
    mf_combination: str,
    load_proba: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    # Reading meta-layer input (classification probabilities)

    # Reading extra features
    X_train_mf = np.load(f"{dir_mf}/{fold_id}/{dataset}/train.npz", allow_pickle=True)["X_train"]
    X_test_mf = np.load(f"{dir_mf}/{fold_id}/{dataset}/test.npz", allow_pickle=True)["X_test"]

    # Bebugging Meta Features with FWLS (matrix of 1's).
    #X_train_mf = np.zeros((X_train_mf.shape[0], X_train_mf.shape[1])) + 1
    #X_test_mf = np.zeros((X_test_mf.shape[0], X_test_mf.shape[1])) + 1

    if load_proba:
        X_train_meta, X_test_meta = read_train_test_meta(
            dir_meta_input, dataset, n_folds, fold_id, models)
    
        if mf_combination == "concat":
            # Concat proba + mf Fatures
            X_train = np.hstack([X_train_meta, X_train_mf])
            X_test = np.hstack([X_test_meta, X_test_mf])
        elif mf_combination == "fwls":
            X_train = fwls(X_train_mf, X_train_meta)
            X_test = fwls(X_test_mf, X_test_meta)
    else:
        X_train = X_train_mf
        X_test = X_test_mf

    return X_train, X_test