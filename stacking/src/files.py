#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" TODO: Short description.
TODO: Longer description.
"""

import json
import numpy as np
from typing import Tuple, List

def check_f1(scoring_path: str):

    with open(scoring_path, 'r') as fd:
        scores = json.load(fd)
        if scores["precision"] > 65:
            return True
        return False


def load_x_y(
        file: str,
        test_train: str
) -> Tuple[np.ndarray, np.ndarray]:
    loaded = np.load(file, allow_pickle=True)

    X = loaded[f"X_{test_train}"]
    
    if f"y_{test_train}" not in loaded:
        return X, None

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

    labels_dir = f"{data_source}/clfs_output/{sp_settings}/{dataset}/10_folds/ltr/{fold}"
    y_train = np.load(f"{labels_dir}/train.npz")["y_train"]
    y_test = np.load(f"{labels_dir}/test.npz")["y_test"]

    return (y_train, y_test)

def read_train_test_meta_oracle(
        data_source: str,
        dataset: str,
        n_folds: int,
        fold_id: int,
        clf_set: List[List],
        oracle_strategy: str,
        confidence: float,
        zero_train: bool
) -> Tuple[np.ndarray, np.ndarray]:
    
    Xs_train, Xs_test = [], []

    for clf, proba_type, oracle_feats in clf_set:
        
        probas_dir = f"{data_source}/{proba_type}/split_{n_folds}/{dataset}/{n_folds}_folds/{clf}/{fold_id}"
        X_train_meta, _ = load_x_y(f"{probas_dir}/train.npz", 'train')
        X_test_meta, _ = load_x_y(f"{probas_dir}/test.npz", 'test')
        
        oracle_base_dir = f"{data_source}/oracle/{oracle_strategy}/{proba_type}/{dataset}/{n_folds}_folds/{clf}/{oracle_feats}/{fold_id}"

        oracle_train = np.load(f"{oracle_base_dir}/train.npz")['y']
        if not zero_train:
            oracle_train[oracle_train == 0] = 1
        
        oracle_test = np.load(f"{oracle_base_dir}/test.npz")['y']

        if check_f1(f"{oracle_base_dir}/scoring.json"):
            
            probs_class_0 = 1 - oracle_test
            oracle_test = np.array([ 0 if prob > confidence else 1 for prob in probs_class_0 ])
        else:
            oracle_test = np.zeros(oracle_test.shape[0]) + 1

        Xs_train.append(X_train_meta * oracle_train[:, None])
        Xs_test.append(X_test_meta * oracle_test[:, None])

    X_train_meta = np.hstack(Xs_train)
    X_test_meta = np.hstack(Xs_test)

    return X_train_meta, X_test_meta