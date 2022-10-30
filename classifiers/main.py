from cProfile import label
from sys import exit
import os
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from itertools import product
from joblib import load, dump

# Loading local libraries.
from src.models.models import get_clf
from src.models.optimization import execute_optimization
from src.train_probas.train_probas import build_train_probas

# Warning Control.
import warnings
from optuna.exceptions import ExperimentalWarning
from optuna.logging import set_verbosity, WARNING

set_verbosity(WARNING)
warnings.filterwarnings("ignore")


try:
    with open("data/settings.json", 'r') as fd:
        settings = json.load(fd)
except:
    raise("file data/setup/settings.json not found.")


DATASETS = settings["DATASETS"]
CLFS = settings["CLFS"]
REP = settings["REP"]
DATA_SOURCE = settings["DATA_SOURCE"]
SPLIT_SETTINGS = settings["SPLIT_SETTINGS"]
N_JOBS = settings["N_JOBS"]
N_FOLDS = settings["N_FOLDS"]
SUB_FOLDS_JOBS = settings["SUB_FOLDS_JOBS"]
BUILD_TRAIN_PROBAS = settings["BUILD_TRAIN_PROBAS"]

iterations = product(
    DATASETS,
    CLFS,
    REP,
    SPLIT_SETTINGS,
    range(N_FOLDS))


for dset, clf, rep, sp_setting, fold in iterations:

    print(f" {dset.upper()} - [{clf.upper()} / {rep.upper()}] - FOLD: {fold}")

    reps_dir = f"{DATA_SOURCE}/representations/{dset}/{N_FOLDS}_folds/{sp_setting}/{rep}/{fold}/"
    
    X_train = np.load(f"{reps_dir}/train.npy", allow_pickle=True)
    X_test = np.load(f"{reps_dir}/test.npy", allow_pickle=True)
    
    labels_dir = f"{DATA_SOURCE}/datasets/labels/{sp_setting}/{dset}/{fold}/"
    
    y_train = np.load(f"{labels_dir}/train.npy", allow_pickle=True)
    y_test = np.load(f"{labels_dir}/test.npy", allow_pickle=True)
    
    log_dir = f"{DATA_SOURCE}/clfs_output/{sp_setting}/{dset}/{N_FOLDS}_folds/{rep}/{clf}/{fold}"
    
    model_path = f"{log_dir}/model.joblib"
    os.makedirs(log_dir, exist_ok=True)
    
    optuna_search = execute_optimization(
        classifier_name=clf,
        file_model=model_path,
        X_train=X_train,
        y_train=y_train,
        opt_n_jobs=N_JOBS,
        load_model=True)
    
    probas = optuna_search.best_estimator_.predict_proba(X_test)
    y_pred = probas.argmax(axis=1)
    

    np.save(f"{log_dir}/probas", probas)
    np.save(f"{log_dir}/y_pred", y_pred)
    
    scoring = {}

    scoring["macro"] = f1_score(y_test, y_pred, average="macro")
    scoring["micro"] = f1_score(y_test, y_pred, average="micro")
    scoring["accuracy"] = accuracy_score(y_test, y_pred)

    print(f"\n\tMacro: {scoring['macro'] * 100}")
    print(f"\tMicro: {scoring['micro'] * 100}\n")

    with open(f"{log_dir}/scoring.json", "w") as fd:
        json.dump(scoring, fd)

    if BUILD_TRAIN_PROBAS:
        print("\tBuilding Train Probas.")
        sub_splits_path = f"{log_dir}/sub_splits"
        X_train_probas = build_train_probas(X_train, 
            y_train, 
            clf, 
            n_splits=10, 
            n_jobs=SUB_FOLDS_JOBS, 
            log_dir=sub_splits_path,
            best_params=optuna_search.best_params_)
        np.save(f"{log_dir}/train_probas", X_train_probas)