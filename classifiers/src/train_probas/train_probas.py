import os
import json
import numpy as np
from src.models.models import get_classifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score

from src.models.optimization import execute_optimization
from src.models.calibration import probabilities_calibration


def report_scoring(y_test: np.ndarray,
                   y_pred: np.ndarray,
                   output_dir: str,
                   fold: int):

    scoring = {}
    scoring["macro"] = f1_score(y_test, y_pred, average="macro")
    scoring["micro"] = f1_score(y_test, y_pred, average="micro")
    scoring["accuracy"] = accuracy_score(y_test, y_pred)
    print(f"\t\tFOLD {fold} - Macro: {scoring['macro']}")  # , end='')

    # Saving model's f1 and accuracy.
    with open(f"{output_dir}/scoring.json", "w") as fd:
        json.dump(scoring, fd)


def build_train_probas(
        clf: str,
        base_output_dir: str,
        X: np.ndarray,
        y: np.ndarray,
        save_probas_calib: bool,
        do_calib: bool,
        calib_method: str,
        n_splits: int = 4,
        n_jobs: int = 5,
        opt_n_jobs: int = 1,
        load_model: bool = False
) -> dict:

    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(X, y)

    align_idx = np.arange(y.shape[0])
    idx_list = []
    probas = []
    calib_probas = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        align = align_idx[test_index]
        idx_list.append(align)

        # Applying oversampling when it is needed.
        for c in set(y_test) - set(y_train):

            sintetic = np.zeros(X_train.shape[1])
            X_train = np.vstack([X_train, sintetic])
            y_train = np.hstack([y_train, [c]])

        if do_calib:
            X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                              y_train,
                                                              random_state=42,
                                                              stratify=y_train,
                                                              test_size=0.10)
        else:
            X_train, y_train = X_train, y_train

        output_dir = f"{base_output_dir}/sub_fold/{fold}"
        os.makedirs(output_dir, exist_ok=True)

        # Hyper tuning
        estimator = execute_optimization(
            classifier_name=clf,
            output_dir=output_dir,
            X_train=X_train,
            y_train=y_train,
            opt_n_jobs=opt_n_jobs,
            clf_n_jobs=n_jobs,
            load_model=load_model)

        test_probas = estimator.predict_proba(X_test)
        probas.append(test_probas)
        
        print("\tNormal.")
        
        report_scoring(y_test, test_probas.argmax(axis=1), output_dir, fold)
        
        if save_probas_calib:
            eval_probas = estimator.predict_proba(X_val)
            np.savez(f"{output_dir}/test.npz",
                     X_test=test_probas, y_test=y_test)
            np.savez(f"{output_dir}/eval.npz",
                     X_eval=eval_probas, y_eval=y_val)
            np.savez(f"{output_dir}/align.npz", align=align)
        
        if do_calib:
            print("\tCalibrated.")
            calib_output_dir = output_dir.replace(
                "normal_probas", calib_method)
            os.makedirs(calib_output_dir, exist_ok=True)
            calib_est = probabilities_calibration(
                estimator, X_val, y_val, calib_method)
            c_probas = calib_est.predict_proba(X_test)
            np.savez(f"{calib_output_dir}/test",
                     X_test=c_probas, y_test=y_test)
            
            report_scoring(y_test, c_probas.argmax(
                axis=1), calib_output_dir, fold)
            calib_probas.append(c_probas)


    sorted_idxs = np.hstack(idx_list).argsort()
    probas = np.vstack(probas)[sorted_idxs]

    if do_calib:
        calib_probas = np.vstack(calib_probas)[sorted_idxs]
        return {"probas": probas,
                "calib_probas": calib_probas}

    return {"probas": probas,
            "calib_probas": None}
