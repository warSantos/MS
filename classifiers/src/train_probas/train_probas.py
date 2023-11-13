import os
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.models.optimization import execute_optimization
from src.models.calibration import probabilities_calibration
from src.models.models import fix_labels
from src.aws.awsio import store_nparrays_in_aws, load_reps_from_aws
from src.utils.utils import report_scoring, replace_nan

def build_train_probas(
        clf: str,
        base_input_dir: str,
        base_output_dir: str,
        save_probas_calib: bool,
        do_calib: bool,
        calib_method: str,
        rep: str,
        n_splits: int = 4,
        n_jobs: int = 5,
        opt_n_jobs: int = 1,
        load_model: bool = False,
        do_optimization: bool = True
) -> dict:

    probas = []
    calib_probas = []
    indexes = []

    for fold in range(n_splits):

        input_dir = f"{base_input_dir}/sub_folds/{fold}"
        train_loader = load_reps_from_aws(f"{input_dir}/train.npz", "train")
        test_loader = load_reps_from_aws(f"{input_dir}/test.npz", "test")
        
        if rep == "tr":
            X_train, X_test = train_loader["X_train"].tolist(), test_loader["X_test"].tolist()
        else:
            X_train, X_test = replace_nan(train_loader["X_train"], rep), replace_nan(test_loader["X_test"], rep)
        
        y_train, y_test = fix_labels(train_loader["y_train"]), fix_labels(test_loader["y_test"])
        
        indexes.append(test_loader["fold_indexes"])

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
            load_model=load_model,
            do_optimization=do_optimization)

        test_probas = estimator.predict_proba(X_test)
        probas.append(test_probas)
        
        print("\tNormal.")
    
        report_scoring(y_test, test_probas.argmax(axis=1), output_dir, fold)
        
        if save_probas_calib:
            eval_probas = estimator.predict_proba(X_val)
            
            store_nparrays_in_aws(f"{output_dir}/test.npz",
                     {"X_test": test_probas, "y_test": y_test})
            store_nparrays_in_aws(f"{output_dir}/eval.npz",
                     {"X_eval": eval_probas, "y_eval": y_val})
        
        if do_calib:
            print("\tCalibrated.")
            calib_output_dir = output_dir.replace(
                "normal_probas", calib_method)
            os.makedirs(calib_output_dir, exist_ok=True)
            calib_est = probabilities_calibration(
                estimator, X_val, y_val, calib_method)
            c_probas = calib_est.predict_proba(X_test)
            
            store_nparrays_in_aws(f"{calib_output_dir}/test.npz",
                     {"X_test": c_probas, "y_test": y_test})
            
            report_scoring(y_test, c_probas.argmax(
                axis=1), calib_output_dir, fold)
            calib_probas.append(c_probas)

    indexes = np.hstack(indexes)
    sort = indexes.argsort()
    probas = np.vstack(probas)[sort]

    if do_calib:
        calib_probas = np.vstack(calib_probas)[sort]
        return {"probas": probas,
                "calib_probas": calib_probas}

    return {"probas": probas,
            "calib_probas": None}
