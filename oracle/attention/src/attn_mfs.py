"""
from sys import path

path.append("../error_estimator")
path.append("../utils")

from local_utils import load_clfs_probas
from mfs import make_mfs
"""

import numpy as np

def apply_attention_format(X: np.ndarray, n_times: int):

    n_docs, dim = X.shape
    X_att = np.tile(X, n_times)
    return X_att.reshape(n_docs, -1, dim)

def load_embeddings(data_source: str,
                    dataset: str,
                    fold: int,
                    n_folds: int,
                    att_mode: bool,
                    n_clfs: int):
    
    base = f"{data_source}/representations/{dataset}/{n_folds}_folds/bert/{fold}"
    train = np.load(f"{base}/train.npz")[f"X_train"]
    test = np.load(f"{base}/test.npz")[f"X_test"]
    
    if att_mode:
        att_train = apply_attention_format(train, n_clfs)
        att_test = apply_attention_format(test, n_clfs)
        return att_train, att_test

    return train, test

def to_attention(arrays: list):

    n_docs = len(arrays[0])
    dim = arrays[0].shape[1]
    a = np.hstack(arrays)
    return a.reshape(n_docs, -1, dim)

def load_mfs(data_source: str,
             dataset: str,
             clfs: list,
             fold: int):
    
    train, test = [], []
    mf_type = "meta_features/error_estimation/probas-based"
    clf_sufix = '/'.join(sorted([ f"{c[0]}_{c[1]}" for c in clfs ]))
    for clf, _ in clfs:
        mf_path = f"{data_source}/{mf_type}/{dataset}/{fold}/{clf_sufix}/{clf}/feats.npz"
        loader = np.load(mf_path)
        train.append(loader["train"])
        test.append(loader["test"])
    train = to_attention(train)
    test = to_attention(test)
    return train, test


    


def get_error_est_mfs(data_source: str,
                      dataset: str,
                      clfs: list,
                      fold: int,
                      n_folds: int):
    
    X_train = load_mfs(data_source, dataset, clfs, fold, n_folds, "train")
    X_test = load_mfs(data_source, dataset, clfs, fold, n_folds, "test")
    return X_train, X_test