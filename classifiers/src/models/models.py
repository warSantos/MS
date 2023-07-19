import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
#from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    IntDistribution)


class LinearSVC(LinearSVC):
    def predict_proba(self, X):
        return self._predict_proba_lr(X)


class SVC(SVC):
    def predict_proba(self, X):
        return self._predict_proba_lr(X)


# LinearSVM
CLF_SVM = LinearSVC(random_state=42, dual=False)
HYP_SVM = {
    "C": FloatDistribution(low=1e-8, high=20.)
}

# RBF SVM
CLF_SVM_RBF = SVC(random_state=42, kernel='rbf')
HYP_SVM_RBF = {
    "C": FloatDistribution(low=1e-8, high=20.)
}

# Centroid
CLF_NC = NearestCentroid()
HYP_NC = {
    "metric": CategoricalDistribution(["cosine", "euclidean"]),
    "shrink_threshold": FloatDistribution(low=0, high=1, step=0.2)
}

# Logistic Regression
CLF_LOGISTIC = LogisticRegression(random_state=42, solver="sag", max_iter=200)
HYP_LOGISTIC = {
    "C": FloatDistribution(low=1e-8, high=20.),
    "penalty": CategoricalDistribution(["none", "l2"]),
    "class_weight": CategoricalDistribution([None, "balanced"])
}

# kNN
CLF_KNN = KNeighborsClassifier()
HYP_KNN = {
    "n_neighbors": IntDistribution(low=3, high=100),
    "metric": CategoricalDistribution(["cosine", "l1", "l2", "minkowski", "euclidean"]),
    "weights": CategoricalDistribution(["uniform", "distance"])
}

# XGBoost
CLF_XGB = XGBClassifier(random_state=42, tree_method='gpu_hist', verbosity=0)
#CLF_XGB = XGBClassifier(random_state=42, tree_method='hist', verbosity=0)
HYP_XGB = {
    "n_estimators": IntDistribution(low=100, high=1000, step=50),
    "learning_rate": FloatDistribution(low=.01, high=.5),
    "eta": FloatDistribution(low=.025, high=.5),
    "max_depth": IntDistribution(low=1, high=14),
    "subsample": FloatDistribution(low=.5, high=1.),
    "gamma": FloatDistribution(low=1e-8, high=1.),
    "colsample_bytree": FloatDistribution(low=.5, high=1.)
}

# Random Forest
CLF_RF = RandomForestClassifier(random_state=42, verbose=0)
HYP_RF = {
    "n_estimators": IntDistribution(low=100, high=500, step=50),
    "max_depth": IntDistribution(low=1, high=14)
}

#CLF_GBM = LGBMClassifier(objective='multiclass', random_state=42)
#HYP_GBM = {
#    "n_estimators": IntDistribution(low=100, high=500, step=50),
#    "num_leaves": IntDistribution(low=16, high=56, step=5),
#    "max_depth": IntDistribution(low=3, high=13, step=2)
#}

ALIAS = {
    "knn/pr": "kpr",
    "knn/tr": "ktr",
    "logistic_regression/pr": "lpr",
    "logistic_regression/tr": "ltr",
    "svm/fr": "sfr",
    "svm/tmk": "stmk",
    "xgboost/fr": "xfr",
    "xgboost/tmk": "xtmk",
    "knn/fr": "kfr",
    "knn/tmk": "ktmk",
    "logistic_regression/fr": "lfr",
    "logistic_regression/tmk": "ltmk",
    "svm/pr": "spr",
    "svm/tr": "str",
    "linear_svm/tmk": "lstmk",
    "linear_svm/tr": "lstr",
    "linear_svm/fr": "lsfr",
    "xgboost/pr": "xpr",
    "xgboost/tr": "xtr"
}

def get_classifier(
        classifier_name: str,
        *,
        n_jobs: int = -1
):
    if classifier_name == "linear_svm":
        classifier, hyperparameters = CLF_SVM, HYP_SVM
    elif classifier_name == "svm":
        classifier, hyperparameters = CLF_SVM_RBF, HYP_SVM_RBF
    elif classifier_name == "logistic_regression":
        classifier, hyperparameters = CLF_LOGISTIC, HYP_LOGISTIC
        classifier.set_params(**{"n_jobs": n_jobs})
    elif classifier_name == "knn":
        classifier, hyperparameters = CLF_KNN, HYP_KNN
        classifier.set_params(**{"n_jobs": n_jobs})
    elif classifier_name == "xgboost":
        classifier, hyperparameters = CLF_XGB, HYP_XGB
        #classifier.set_params(**{"n_jobs": n_jobs})
    elif classifier_name == "random_forest":
        classifier, hyperparameters = CLF_RF, HYP_RF
    elif classifier_name == "centroid":
        classifier, hyperparameters = CLF_NC, HYP_NC
    else:
        raise ValueError(f"Classifier {classifier_name} does not exits.")

    return classifier, hyperparameters

def fix_labels(y: np.ndarray):

    if np.min(y) > 0:
        return y - 1
    return y