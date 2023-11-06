#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Classification models.
Classification models to run the text classification tasks.
"""
import warnings

from optuna.distributions import UniformDistribution, CategoricalDistribution, IntUniformDistribution, LogUniformDistribution
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
AVAILABLE_CLASSIFIER = ["linear_svm", "logistic_regression", "knn", "xgboost"]


class LinearSVC(svm.LinearSVC):
    def predict_proba(self, X):
        return self._predict_proba_lr(X)

class SVC(svm.SVC):
    def predict_proba(self, X):
        return self._predict_proba_lr(X)

# LinearSVM
CLF_SVM = LinearSVC(random_state=42, dual=False)
HYP_SVM = {
    "C": UniformDistribution(low=1e-8, high=20.)
}

CLF_SVM_RBF = SVC(random_state=42, kernel='rbf')
HYP_SVM_RBF = {
    "C": UniformDistribution(low=1e-8, high=20.)
}

# Logistic Regression
CLF_LOGISTIC = LogisticRegression(random_state=42, solver="sag")
HYP_LOGISTIC = {
    "C": UniformDistribution(low=1e-8, high=20.),
    "penalty": CategoricalDistribution(["none", "l2"]),
    "class_weight": CategoricalDistribution([None, "balanced"])
}

# kNN
CLF_KNN = KNeighborsClassifier()
HYP_KNN = {
    "n_neighbors": IntUniformDistribution(low=3, high=100),
    "metrics": CategoricalDistribution(["cosine", "l1", "l2", "minkowski", "euclidean"]),
    "weights": CategoricalDistribution(["uniform", "distance"])
}

# XGBoost
CLF_XGB = XGBClassifier(random_state=42, verbosity=0)
HYP_XGB = {
    "n_estimators": IntUniformDistribution(low=100, high=1000, step=50),
    "learning_rate": LogUniformDistribution(low=.01, high=.5),
    "eta": LogUniformDistribution(low=.025, high=.5),
    "max_depth": IntUniformDistribution(low=1, high=14),
    "subsample": LogUniformDistribution(low=.5, high=1.),
    "gamma": LogUniformDistribution(low=1e-8, high=1.),
    "colsample_bytree": LogUniformDistribution(low=.5, high=1.)
}

# Random Forest
CLF_RF = RandomForestClassifier(random_state=42, verbose=0)
HYP_RF = {
    "n_estimators": IntUniformDistribution(low=100, high=300, step=50),
    "criterion": CategoricalDistribution(["gini"]),
    "max_depth": IntUniformDistribution(low=1, high=14)
}


def get_classifier(
        classifier_name: str,
        *,
        n_jobs: int = -1
):
    if classifier_name == "linear_svm":
        classifier, hyperparameters = CLF_SVM, HYP_SVM
    if classifier_name == "svm_rbf":
        classifier, hyperparameters = CLF_SVM_RBF, HYP_SVM_RBF
    elif classifier_name == "logistic_regression":
        classifier, hyperparameters = CLF_LOGISTIC, HYP_LOGISTIC
        classifier.set_params(**{"n_jobs": n_jobs})
    elif classifier_name == "knn":
        classifier, hyperparameters = CLF_KNN, HYP_KNN
        classifier.set_params(**{"n_jobs": n_jobs})
    elif classifier_name == "xgboost":
        classifier, hyperparameters = CLF_XGB, HYP_XGB
        classifier.set_params(**{"n_jobs": n_jobs})
    elif classifier_name == "random_forest":
        classifier, hyperparameters = CLF_RF, HYP_RF
    else:
        raise ValueError(f"Classifier {classifier_name} does not exits. Possible values: {AVAILABLE_CLASSIFIER}")

    return classifier, hyperparameters
