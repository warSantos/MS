import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from optuna.integration import OptunaSearchCV
from optuna.distributions import (IntDistribution,
                                  FloatDistribution,
                                  CategoricalDistribution)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from xgboost import XGBClassifier

from joblib import load, dump

def get_clf(clf="xgboost", n_jobs=25):

    if clf == "xgboost":
        # , tree_method='gpu_hist')
        CLF_XGB = XGBClassifier(random_state=42, verbosity=0, tree_method='gpu_hist')
        HYP_XGB = {
            "n_estimators": IntDistribution(low=100, high=500, step=100),
            "learning_rate": FloatDistribution(low=.01, high=.5),
            "max_depth": IntDistribution(low=3, high=14),
            "subsample": FloatDistribution(low=.5, high=1.),
            "booster": CategoricalDistribution(["gbtree", "gblinear", "dart"]),
            "colsample_bytree": FloatDistribution(low=.5, high=1.)
        }
        return CLF_XGB, HYP_XGB, {}
    elif clf == "logistic_regression":
        CLF_LOGISTIC = LogisticRegression(random_state=42, solver="sag", n_jobs=n_jobs, max_iter=150)
        HYP_LOGISTIC = {
            "C": FloatDistribution(low=1e-8, high=20.),
            "penalty": CategoricalDistribution(["none", "l2"]),
            "class_weight": CategoricalDistribution([None, "balanced"])
        }
        return CLF_LOGISTIC, HYP_LOGISTIC, {}
    elif clf == "rf":
        CLF_RF = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
        HYP_RF = {
            "n_estimators": IntDistribution(low=100, high=500, step=50),
            "max_depth": IntDistribution(low=3, high=14),
            "min_samples_leaf": CategoricalDistribution([1, 2, 4]),
            "min_samples_split": CategoricalDistribution([2, 5, 10]),
            "bootstrap": CategoricalDistribution([True, False])
        }
        DEFAULT_HYP = {
            "n_estimators": 300,
            "max_depth": 5,
            "min_samples_leaf": 2
        }
        return CLF_RF, HYP_RF, DEFAULT_HYP
    else:
        HYP_GBM = {}
        return GradientBoostingClassifier(), HYP_GBM, {}


def execute_optimization(
        classifier_name: str,
        file_model: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        opt_cv: int = 4,
        opt_n_iter: int = 30,
        opt_scoring: str = "f1_macro",
        opt_n_jobs: int = 5,
        clf_n_jobs: int = 10,
        seed: int = 42,
        load_model: bool = False
) -> BaseEstimator:

    classifier, hyperparameters, _ = get_clf(classifier_name, n_jobs=clf_n_jobs)
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
        n_jobs=opt_n_jobs
    )

    if load_model and os.path.exists(file_model):
        print("\tModel already executed! Loading model...")
        optuna_search = load(file_model)
    else:
        print("\tExecuting model...")
        optuna_search.fit(X_train, y_train)
        dump(optuna_search, file_model)

    return optuna_search


