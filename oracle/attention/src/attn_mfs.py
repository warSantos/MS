"""
from sys import path

path.append("../error_estimator")
path.append("../utils")

from local_utils import load_clfs_probas
from mfs import make_mfs
"""

import numpy as np


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
    
    """
    mf_set_sufix = "centroids-ratios_dist_neigborhood_probas_probas-based"
    features = []
    clf_sufix_set = '/'.join(
        sorted([f"{clf}_{proba_type}" for clf, proba_type in clfs]))
    for clf, _ in clfs:
        mf_input_dir = f"{data_source}/{dataset}/{n_folds}_folds/{clf_sufix_set}/{mf_set_sufix}/{clf}/{fold}"
        features.append(
            np.load(f"{mf_input_dir}/x_{train_test}.npz")[f"X_{train_test}"][:, :18])

    attn_feats = []
    for doc_idx in np.arange(features[0].shape[0]):
        attn_feats.append(
            np.vstack([features[idx_clf][doc_idx]
                       for idx_clf in np.arange(len(clfs))])
        )
    #print(len(attn_feats), attn_feats[0].shape)
    return attn_feats
    """



def get_error_est_mfs(data_source: str,
                      dataset: str,
                      clfs: list,
                      fold: int,
                      n_folds: int):
    
    X_train = load_mfs(data_source, dataset, clfs, fold, n_folds, "train")
    X_test = load_mfs(data_source, dataset, clfs, fold, n_folds, "test")
    return X_train, X_test