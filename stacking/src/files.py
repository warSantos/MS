#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" TODO: Short description.
TODO: Longer description.
"""

from typing import Tuple, List
import numpy as np

def load_x_y(
        file: str,
        test_train: str
) -> Tuple[np.ndarray, np.ndarray]:
    loaded = np.load(file, allow_pickle=True)

    X = loaded[f"X_{test_train}"]
    y = loaded[f"y_{test_train}"]

    if X.size == 1:
        X = X.item()

    return X, y


def read_train_test_meta(
        dir_meta_input: str,
        dataset: str,
        n_folds: int,
        fold_id: int,
        algorithms: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    Xs_train, Xs_test = [], []

    for alg in algorithms:
        file_train_meta = f"{dir_meta_input}/{dataset}/{n_folds}_folds/{alg}/{fold_id}/train.npz"
        file_test_meta = f"{dir_meta_input}/{dataset}/{n_folds}_folds/{alg}/{fold_id}/test.npz"

        X_train_meta, _ = load_x_y(file_train_meta, 'train')
        X_test_meta, _ = load_x_y(file_test_meta, 'test')

        Xs_train.append(X_train_meta)
        Xs_test.append(X_test_meta)

    X_train_meta = np.hstack(Xs_train)
    X_test_meta = np.hstack(Xs_test)

    return X_train_meta, X_test_meta

def load_y(
        data_source: str,
        dataset: str,
        fold: int,
        sp_settings: str
) -> Tuple[np.ndarray, np.ndarray]:

    labels_dir = f"{data_source}/datasets/labels/{sp_settings}/{dataset}/{fold}"
    y_train = np.load(f"{labels_dir}/train.npy")
    y_test = np.load(f"{labels_dir}/test.npy")

    return y_train, y_test

def read_train_test_bert(
        data_source: str,
        dataset: str,
        algorithms: List[str],
        n_folds: int,
        fold_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    
    Xs_train, Xs_test = [], []

    for clf in algorithms:
        
        probs_dir = f"{data_source}/clfs_output/split_10_with_val/{dataset}/{n_folds}_folds/{clf}/{fold_id}"

        X_train_meta = np.load(f"{probs_dir}/train_probas.npy")
        X_test_meta = np.load(f"{probs_dir}/probas.npy")

        Xs_train.append(X_train_meta)
        Xs_test.append(X_test_meta)

    X_train_meta = np.hstack(Xs_train)
    X_test_meta = np.hstack(Xs_test)

    return X_train_meta, X_test_meta
