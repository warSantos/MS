#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" TODO: Short description.
TODO: Longer description.
"""

import os
import json
import numpy as np
from joblib import load, dump
from optuna.integration import OptunaSearchCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.models import get_classifier

def load_params(params_path: str):

    with open(params_path, 'r') as fd:
        params = json.load(fd)
        return { key.replace("classifier__",''):params[key] for key in params if key.find("classifier__") > -1 }

def save_params(optuna_search: OptunaSearchCV, params_path: str):

    with open(params_path, 'w') as fd:
        json.dump(optuna_search.best_params_, fd)

def run_model(classifier_name: str,
              params_path: str,
              X_train: np.ndarray,
              y_train: np.ndarray,
              n_jobs: int):
    
    classifier, _ = get_classifier(classifier_name, n_jobs=n_jobs)
    best_params = load_params(params_path)
    classifier.set_params(**best_params)
    classifier.fit(X_train, y_train)
    return classifier

def execute_optimization(
        classifier_name: str,
        output_dir: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        opt_cv: int = 5,
        opt_n_iter: int = 30,
        opt_scoring: str = "f1_macro",
        opt_n_jobs: int = 1,
        clf_n_jobs: int = -1,
        seed: int = 42,
        load_model: bool = False
) -> BaseEstimator:
    
    params_path = f"{output_dir}/hyper_params.json"
    if load_model and os.path.exists(params_path):
        
        print("\tModel already trained! Loading best params set...")
        classifier = run_model(classifier_name, 
                               params_path, 
                               X_train, 
                               y_train, 
                               clf_n_jobs)
        return classifier
        
    else:

        classifier, hyperparameters = get_classifier(classifier_name, n_jobs=clf_n_jobs)
        pipeline = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", classifier)
        ])
        hyperparameters = {f"classifier__{k}": v for k,
                        v in hyperparameters.items()}

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
        print("\tOptimizing model...")
        optuna_search.fit(X_train, y_train)
        save_params(optuna_search, params_path)
        """
        classifier = run_model(classifier_name, 
                               params_path, 
                               X_train, 
                               y_train, 
                               clf_n_jobs)
        """
        classifier = optuna_search.best_estimator_
        return classifier