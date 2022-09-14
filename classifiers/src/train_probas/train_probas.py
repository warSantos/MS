import os
import json
from termios import N_SLIP
import numpy as np
from src.models.models import get_clf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from joblib import load, dump

def build_train_probas(
        X: np.ndarray,
        y: np.ndarray,
        clf: str,
        n_splits: int = 10,
        n_jobs: int = 10,
        log_dir: str = None
) -> np.ndarray:

    sfk = StratifiedKFold(n_splits=n_splits)

    sfk.get_n_splits(X, y)
    
    alig_idx = np.arange(y.shape[0])
    idx_list = []
    probas = []
    
    for fold, (train_index, test_index) in enumerate(sfk.split(X, y)):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        idx_list.append(alig_idx[test_index])

        # Applying oversampling when it is needed.
        for c in set(y_test) - set(y_train):
            
            #perturbation = np.random.rand(1, X_train.shape[1]) / 100
            #sintetic = np.mean(X_train[y_train == c], axis=0) + perturbation
            sintetic = np.zeros(X_train.shape[1])
            X_train = np.vstack([X_train, sintetic])
            y_train = np.hstack([y_train, [c]])
        
        estimator = get_clf(clf, n_jobs=n_jobs)
        estimator.fit(X_train, y_train)
        probas.append(estimator.predict_proba(X_test))

        if log_dir is not None:
            dst = f"{log_dir}/{fold}"
            os.makedirs(dst, exist_ok=True)
            # Saving the model.
            dump(estimator, f"{dst}/model.joblib")
            # Saving model's f1 and accuracy.
            with open(f"{dst}/scoring.json", "w") as fd:
                scoring = {}
                y_pred = estimator.predict(X_test)
                scoring["macro"] = f1_score(y_test, y_pred, average="macro")
                scoring["micro"] = f1_score(y_test, y_pred, average="micro")
                scoring["accuracy"] = accuracy_score(y_test, y_pred)
                json.dump(scoring, fd)

    probas = np.vstack(probas)
    sorted_idxs = np.hstack(idx_list).argsort()
    return probas[sorted_idxs]