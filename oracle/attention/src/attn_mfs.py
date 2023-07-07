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
    return X_att.reshape(n_docs, dim, -1)

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



def load_mfs(data_source: str,
             dataset: str,
             clfs: list,
             fold: int,
             n_folds: int,
             train_test: str):
    
    base = f"{data_source}/representations/{dataset}/{n_folds}_folds/bert/{fold}"
    attn_feats = np.load(f"{base}/{train_test}.npz")[f"X_{train_test}"]
    n_docs, dim = attn_feats.shape
    attn_feats = np.tile(attn_feats, 7)
    return attn_feats.reshape(n_docs, dim, -1)


def get_error_est_mfs(data_source: str,
                      dataset: str,
                      clfs: list,
                      fold: int,
                      n_folds: int):
    
    X_train = load_mfs(data_source, dataset, clfs, fold, n_folds, "train")
    X_test = load_mfs(data_source, dataset, clfs, fold, n_folds, "test")
    return X_train, X_test