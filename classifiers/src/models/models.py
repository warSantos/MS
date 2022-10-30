from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier

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

CLF_SVM_RBF = SVC(random_state=42, kernel='rbf')
HYP_SVM_RBF = {
    "C": FloatDistribution(low=1e-8, high=20.)
}


CLF_NC = NearestCentroid()
HYP_NC = {
    "metric": CategoricalDistribution(["cosine", "euclidean"]),
    "shrink_threshold": FloatDistribution(low=0, high=1, step=0.2)
}

# Logistic Regression
CLF_LOGISTIC = LogisticRegression(random_state=42, solver="sag")
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
    "weights": CategoricalDistribution(["uniform", "distance"]),
    "n_jobs": 1
}

# Random Forest
CLF_RF = RandomForestClassifier(random_state=42, verbose=0)
HYP_RF = {
    "n_estimators": IntDistribution(low=100, high=500, step=50),
    "max_depth": IntDistribution(low=1, high=14)
}

CLF_GBM = LGBMClassifier(objective='multiclass', random_state=42)
HYP_GBM = {
    "n_estimators": IntDistribution(low=100, high=500, step=50),
    "num_leaves": IntDistribution(low=16, high=56, step=5),
    "max_depth": IntDistribution(low=3, high=13, step=2)
}

def get_clf(clf_name: str, *, n_jobs: int = 10):

    if clf_name == "centroid":
        clf, hyper_ps = CLF_NC, HYP_NC
    if clf_name == "gbm":
        clf, hyper_ps = CLF_GBM, HYP_GBM
    if clf_name == "svm":
        clf, hyper_ps = CLF_SVM, HYP_SVM
    if clf_name == "rf":
        clf, hyper_ps = CLF_RF, HYP_RF
    if clf_name == "lr":
        clf, hyper_ps = CLF_LOGISTIC, HYP_LOGISTIC
    if clf_name == "knn":
        clf, hyper_ps = CLF_KNN, HYP_KNN

    return clf, hyper_ps
