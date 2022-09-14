from cProfile import label
from sys import exit
import os
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from itertools import product
from joblib import load, dump

try:
    with open("data/setup/settings.json", 'r') as fd:
        settings = json.load(fd)
except:
    raise("file data/setup/settings.json not found.")

# Loading local libraries.
from src.models.models import get_clf
from src.train_probas.train_probas import build_train_probas


DATASETS = settings["DATASETS"]
CLFS = settings["CLFS"]
REP = settings["REP"]
DATA_SOURCE = settings["DATA_SOURCE"]
SPLIT_SETTINGS = settings["SPLIT_SETTINGS"]
n_jobs = settings["n_jobs"]
N_FOLDS = settings["N_FOLDS"]
BUILD_TRAIN_PROBAS = settings["BUILD_TRAIN_PROBAS"]

iterations = product(
    DATASETS,
    CLFS,
    REP,
    SPLIT_SETTINGS,
    range(N_FOLDS))


for dset, clf, rep, sp_setting, fold in iterations:

    print(f" {dset.upper()} - [{clf.upper()} / {rep.upper()}] - FOLD: {fold}")

    reps_dir = f"{DATA_SOURCE}/reps/{sp_setting}/{rep}/{dset}/{fold}/"
    
    X_train = np.load(f"{reps_dir}/train.npy", allow_pickle=True)
    X_test = np.load(f"{reps_dir}/test.npy", allow_pickle=True)
    
    labels_dir = f"{DATA_SOURCE}/labels/{sp_setting}/{dset}/{fold}/"
    
    y_train = np.load(f"{labels_dir}/train.npy", allow_pickle=True)
    y_test = np.load(f"{labels_dir}/test.npy", allow_pickle=True)
    
    log_dir = f"{DATA_SOURCE}/clfs_output/{sp_setting}/{dset}/{N_FOLDS}_folds/{clf}/{fold}"
    
    model_path = f"{log_dir}/model.joblib"
    if not os.path.exists(model_path):
        estimator = get_clf(clf, n_jobs=n_jobs)
        estimator.fit(X_train, y_train)
        dump(estimator, model_path)
    else:
        estimator = load(model_path)

    probas = estimator.predict_proba(X_test)
    y_pred = probas.argmax(axis=1)

    
    os.makedirs(log_dir, exist_ok=True)

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
        X_train_probas = build_train_probas(X_train, y_train, clf, n_splits=10, n_jobs=n_jobs, log_dir=sub_splits_path)
        np.save(f"{log_dir}/train_probas", X_train_probas)