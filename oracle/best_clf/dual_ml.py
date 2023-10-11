# Python libraries
import os
import json
import warnings
import itertools
import traceback

# 3rd party libraries
import pandas as pd
import numpy as np
from joblib import load
from optuna.logging import set_verbosity, WARNING
from sklearn.metrics import f1_score

# Local libraries
from utils import load_meta_features
from src.optimization import execute_optimization
from src.loader.loader import read_meta_data
from src.files import (
    load_y
)

# Loading basic settings of the envirioment.
try:
    fd = open("data/settings.json", 'r')
    settings = json.load(fd)
except:
    traceback.print_exc()
    print("Was not possible to read basic enviorioment settings.")
    exit(1)

DATA_SOURCE = settings["DATA_SOURCE"]

# Lib configs
set_verbosity(WARNING)
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect sub-processes

# Directories and files
DIR_OUTPUT = f"{DATA_SOURCE}/stacking/stacking_output"

# Loading basic settings of the envirioment.
try:
    fd = open("data/dual_mp.json", 'r')
    execution = json.load(fd)
except:
    traceback.print_exc()
    print("Was not possible to read executions settings.")
    exit(1)

DATASETS = execution["DATASETS"]
META_LAYERS = execution["META_LAYERS"]
N_JOBS = execution["N_JOBS"]
SEED = execution["SEED"]
np.random.seed(SEED)

SPLIT_SETTINGS = execution["SPLIT_SETTINGS"]
LOAD_MODEL = execution["LOAD_MODEL"]
FEATURES_SET = execution["FEATURES_SET"]
FEATURES_SET.sort()
CUT_SCORE = execution["CUT_SCORE"]
cheating = execution["CHEATING"]
insert_mf = execution["INSERT_MF"]

mf_sufix = '_'.join(FEATURES_SET)


if __name__ == "__main__":

    iterations = itertools.product(DATASETS, META_LAYERS, execution["SETS"])
    for ((dataset, n_classes, n_folds), meta_layer, clf_set) in iterations:

        CLFS = execution[clf_set]
        CLFS.sort(key=lambda c: c[0])
        sufix = '/'.join([f"{tup[0]}_{tup[1]}" for tup in CLFS])

        for fold_id in np.arange(n_folds):

            print(
                f"[{meta_layer.upper()} / {clf_set.upper()}] - {dataset.upper():10s} : {CUT_SCORE} - fold_{fold_id}")

            # Reading classification labels.
            y_train, y_test = load_y(
                DATA_SOURCE, dataset, fold_id, SPLIT_SETTINGS)

            # Setting output dir.
            dir_dest = f"{DIR_OUTPUT}/{dataset}/{n_folds}_folds/{meta_layer}/hard_docs{'_upper_bound' if cheating else ''}"
            dir_dest = f"{dir_dest}{'/insert_mf' if insert_mf else ''}"
            dir_dest = f"{dir_dest}/{sufix}/{mf_sufix}"
            dir_dest = f"{dir_dest}/fold_{fold_id}"
            print(f"[OUTPUT: {dir_dest}]")

            # Reading models probabilities.
            X_train, X_test = read_meta_data(
                DATA_SOURCE,
                dataset,
                CLFS,
                n_folds,
                fold_id
            )

            # Loading hard file detector and train settings.
            detector_dir = f"{DATA_SOURCE}/oracle/hard_docs/{dataset}/{sufix}/{mf_sufix}/fold_{fold_id}"
            hd_loader = np.load(
                f"{detector_dir}/output.pkl", allow_pickle=True)
            train_counts = np.array(hd_loader["train_counts"])
            test_counts = np.array(hd_loader["test_counts"])
            detector = load(f"{detector_dir}/model.joblib")

            # Loading hard file detector meta-features.
            detect_test = []
            for clf, _ in CLFS:
                detect_test.append(np.load(
                    f"{DATA_SOURCE}/meta_features/error_estimation/probas-based/{dataset}/{fold_id}/{sufix}/{clf}/feats.npz")["test"])
            detect_test = np.hstack(detect_test)

            if cheating:
                test_hard_docs = np.zeros(X_test.shape[0])
                test_hard_docs[test_counts < 4] = 1
                detect_probs = test_hard_docs
            else:
                #test_hard_docs = detector.predict(detect_test)
                detect_probs = detector.predict_proba(detect_test)[:, 1]
                test_hard_docs = detect_probs.copy()
                test_hard_docs[test_hard_docs > CUT_SCORE] = 1
                test_hard_docs[test_hard_docs < CUT_SCORE] = 0                


            # Training/Optmizing Hard Docs Specialist ML.
            hard_docs_ids = train_counts < 4
            hard_X_train, hard_y_train = X_train[hard_docs_ids], y_train[hard_docs_ids]
            soft_docs_ids = train_counts > 3
            soft_X_train, soft_y_train = X_train[soft_docs_ids], y_train[soft_docs_ids]
            n_softs = np.sum(soft_docs_ids)

            random_idxs = np.random.randint(0, n_softs, size=(1000,))
            # Joining soft and hard docs.
            hard_X_train = np.vstack([hard_X_train, soft_X_train[random_idxs]])
            hard_y_train = np.hstack([hard_y_train, soft_y_train[random_idxs]])
            
            if insert_mf:
                mf_train, mf_test = load_meta_features(DATA_SOURCE,
                                                       dataset,
                                                       n_classes,
                                                       CLFS,
                                                       sufix,
                                                       fold_id,
                                                       FEATURES_SET,
                                                       mf_sufix,
                                                       n_folds)
                hard_mfs = mf_train[hard_docs_ids]
                soft_mfs = mf_train[soft_docs_ids]
                soft_mfs = soft_mfs[random_idxs]
                spec_mfs = np.vstack([hard_mfs, soft_mfs])
                hard_X_train = np.hstack([hard_X_train, spec_mfs])
                spec_X_test = np.hstack([X_test, mf_test])
            else:
                spec_X_test = X_test

            # Training/Optmizing normal ML.
            print("\tTraining Specialist Model...")
            file_model = f"{dir_dest}/hard_docs_model.joblib"
            os.makedirs(dir_dest, exist_ok=True)
            specialist_model = execute_optimization(
                meta_layer,
                file_model,
                hard_X_train,
                hard_y_train,
                seed=SEED,
                clf_n_jobs=1,
                opt_n_jobs=-1,
                load_model=LOAD_MODEL
            )

            print("\n\tTraining Normal Model...")
            # Training/Optmizing normal ML.
            file_model = f"{dir_dest}/model.joblib"
            os.makedirs(dir_dest, exist_ok=True)
            optuna_search = execute_optimization(
                meta_layer,
                file_model,
                X_train,
                y_train,
                seed=SEED,
                clf_n_jobs=1,
                opt_n_jobs=-1,
                load_model=LOAD_MODEL
            )

            spec_preds = specialist_model.predict(spec_X_test)
            normal_preds = optuna_search.predict(X_test)

            final_preds = []
            for idx in np.arange(test_hard_docs.shape[0]):
                if test_hard_docs[idx] == 1:
                    final_preds.append(spec_preds[idx])
                else:
                    final_preds.append(normal_preds[idx])

            # Prediction
            f1_macro = f1_score(y_test, final_preds, average="macro")
            f1_micro = f1_score(y_test, final_preds, average="micro")

            msg = f"\n\tF1-Macro: {f1_macro:.4f}\n\tF1-Micro: {f1_micro:.4f}"
            print(msg)

            # Save prediction
            file_scoring = f"{dir_dest}/scoring.json"
            scoring = {
                "f1_macro": f1_macro,
                "f1_micro": f1_micro
            }
            with open(file_scoring, "w+", encoding="utf-8") as fp:
                json.dump(scoring, fp, ensure_ascii=False, indent=4)

            np.savez(f"{dir_dest}/output",
                     test_hard_docs=test_hard_docs,
                     test_counts=test_counts,
                     train_counts=train_counts,
                     spec_preds=spec_preds,
                     detect_probs=detect_probs,
                     normal_preds=normal_preds,
                     y_test=y_test)
