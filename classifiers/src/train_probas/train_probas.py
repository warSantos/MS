import os
import json
from termios import N_SLIP
import numpy as np
from src.models.models import get_classifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from joblib import load, dump

from src.models.optimization import execute_optimization

def build_train_probas(
        X: np.ndarray,
        y: np.ndarray,
        clf: str,
        n_splits: int = 3,
        n_jobs: int = 30,
        output_dir: str = None,
        load_model: bool = False,
        calibration: bool = False
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
                
        output = f"{output_dir}/{fold}"
        os.makedirs(output, exist_ok=True)
        model_path = f"{output}/model.joblib"
        
        # Hyper tuning
        optuna_search = execute_optimization(
            classifier_name=clf,
            file_model=model_path,
            X_train=X_train,
            y_train=y_train,
            opt_n_jobs=1,
            clf_n_jobs=n_jobs,
            load_model=load_model,
            calibration=calibration)
        
        probas.append(optuna_search.predict_proba(X_test))
        scoring = {}
        y_pred = optuna_search.predict(X_test)
        scoring["macro"] = f1_score(y_test, y_pred, average="macro")
        scoring["micro"] = f1_score(y_test, y_pred, average="micro")
        scoring["accuracy"] = accuracy_score(y_test, y_pred)
        print(f"\t\tFOLD {fold} - Macro: {scoring['macro']}")#, end='')

        # Saving model's f1 and accuracy.
        with open(f"{output}/scoring.json", "w") as fd:
            json.dump(scoring, fd)

    probas = np.vstack(probas)
    sorted_idxs = np.hstack(idx_list).argsort()
    return probas[sorted_idxs]