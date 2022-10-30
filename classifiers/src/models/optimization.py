#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" TODO: Short description.
TODO: Longer description.
"""

import os

import numpy as np
from joblib import load, dump
from optuna.integration import OptunaSearchCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.models import get_clf

def execute_optimization(
        classifier_name: str,
        file_model: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        opt_cv: int = 5,
        opt_n_iter: int = 30,
        opt_scoring: str = "f1_macro",
        opt_n_jobs: int = -1,
        clf_n_jobs: int = 1,
        seed: int = 42,
        load_model: bool = False
) -> BaseEstimator:
    classifier, hyperparameters = get_clf(classifier_name, n_jobs=clf_n_jobs)
    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("classifier", classifier)
    ])
    hyperparameters = {f"classifier__{k}": v for k, v in hyperparameters.items()}

    optuna_search = OptunaSearchCV(
        pipeline,
        hyperparameters,
        cv=StratifiedKFold(opt_cv, shuffle=True, random_state=seed),
        error_score="raise",
        n_trials=opt_n_iter,
        random_state=seed,
        scoring=opt_scoring,
        n_jobs=opt_n_jobs,
        refit=True
    )

    if load_model and os.path.exists(file_model):
        print("\tModel already executed! Loading model...", end="")
        optuna_search = load(file_model)
    else:
        print("\tExecuting model...", end="")
        optuna_search.fit(X_train, y_train)
        dump(optuna_search, file_model)

    return optuna_search
