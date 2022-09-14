from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier

class LinearSVC(LinearSVC):
    def predict_proba(self, X):
        return self._predict_proba_lr(X)

class SVC(SVC):
    def predict_proba(self, X):
        return self._predict_proba_lr(X)

def get_clf(clf_name: str, *, n_jobs: int = -1):

    if clf_name == "centroid":
        estimator = NearestCentroid()
    if clf_name == "gbm":
        estimator = LGBMClassifier()
    if clf_name == "svm":
        estimator = LinearSVC(random_state=42, max_iter=1000)
    if clf_name == "rf":
        estimator = RandomForestClassifier(n_estimators=200, n_jobs=n_jobs)
    if clf_name == "lr":
        estimator = LogisticRegression(
            random_state=42, solver="liblinear")
    if clf_name == "knn":
        estimator = KNeighborsClassifier(n_neighbors=3,  n_jobs=n_jobs)
    
    return estimator
