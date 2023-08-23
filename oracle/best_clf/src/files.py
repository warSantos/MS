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
        if scores["precision"] > 75:
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
        strategy: str,
        clf_set_sufix: str,
        confidence: float,
        zero_train: bool
) -> Tuple[np.ndarray, np.ndarray]:
    
    Xs_train, Xs_test = [], []

    for clf, proba_type, oracle_feats in clf_set:
        
        probas_dir = f"{data_source}/{proba_type}/split_{n_folds}/{dataset}/{n_folds}_folds/{clf}/{fold_id}"
        X_train_meta, _ = load_x_y(f"{probas_dir}/train.npz", 'train')
        X_test_meta, _ = load_x_y(f"{probas_dir}/test.npz", 'test')
        
        if strategy == "upper_bound":
            error_estimation_dir = f"{data_source}/oracle/{strategy}/{proba_type}/{dataset}/{n_folds}_folds/{clf}/{oracle_feats}/{fold_id}"
        else:
            error_estimation_dir = f"{data_source}/oracle/{strategy}/{proba_type}/{dataset}/{n_folds}_folds/{clf}/{clf_set_sufix}/{oracle_feats}/{fold_id}"

        upper_train = np.load(f"{error_estimation_dir}/train.npz")['y']
        if not zero_train:
            upper_train[upper_train == 0] = 1
        
        test_error_estimation = np.load(f"{error_estimation_dir}/test.npz")['y']
        if strategy not in ["upper_bound"]:
            
            if check_f1(f"{error_estimation_dir}/scoring.json"):
            
                probs_class_0 = 1 - test_error_estimation
                test_error_estimation = np.array([ 0 if prob > confidence else 1 for prob in probs_class_0 ])
            else:
                test_error_estimation = np.zeros(test_error_estimation.shape[0]) + 1

        Xs_train.append(X_train_meta * upper_train[:, None])
        Xs_test.append(X_test_meta * test_error_estimation[:, None])

    X_train_meta = np.hstack(Xs_train)
    X_test_meta = np.hstack(Xs_test)

    return X_train_meta, X_test_meta


def read_cost_error_estimation(
        data_source: str,
        dataset: str,
        n_folds: int,
        fold_id: int,
        clf_set: List[List],
        strat_tuple: Tuple[str, str],
        confidence: float,
        zero_train: bool
) -> Tuple[np.ndarray, np.ndarray]:
    
    Xs_train, Xs_test = [], []

    cost_measure, strategy = strat_tuple

    for clf, proba_type, oracle_feats in clf_set:
        
        probas_dir = f"{data_source}/{proba_type}/split_{n_folds}/{dataset}/{n_folds}_folds/{clf}/{fold_id}"
        X_train_meta, _ = load_x_y(f"{probas_dir}/train.npz", 'train')
        X_test_meta, _ = load_x_y(f"{probas_dir}/test.npz", 'test')
        
        error_estimation_dir = f"{data_source}/oracle/{strategy}/{proba_type}/{dataset}/{n_folds}_folds/{clf}/{oracle_feats}/{fold_id}"
        
        upper_train = np.load(f"{error_estimation_dir}/train.npz")['y']
        
        if not zero_train:
            upper_train[upper_train == 0] = 1
        
        test_error_estimation = np.load(f"{error_estimation_dir}/test.npz")['y']
        # Verifying if the error estimation macro is above 65.
        if check_f1(f"{error_estimation_dir}/scoring.json"):
            
            probs_class_0 = 1 - test_error_estimation
            test_error_estimation = np.array([ 0 if prob > confidence else 1 for prob in probs_class_0  ])
            upper_dir = f"{data_source}/oracle/upper_bound/{proba_type}/{dataset}/{n_folds}_folds/{clf}/{fold_id}/"
            upper_test = np.load(f"{upper_dir}/test.npz")['y']

            if cost_measure == "zero_error_cost":
                
                test_error_estimation[(upper_test == 0) & (test_error_estimation != upper_test)] = 0
            
            elif cost_measure == "keep_error_cost":
            
                test_error_estimation[(upper_test == 1) & (test_error_estimation != upper_test)] = 1

        else:
            test_error_estimation = np.zeros(test_error_estimation.shape[0]) + 1

        Xs_train.append(X_train_meta * upper_train[:, None])
        Xs_test.append(X_test_meta * test_error_estimation[:, None])

    X_train_meta = np.hstack(Xs_train)
    X_test_meta = np.hstack(Xs_test)

    return X_train_meta, X_test_meta