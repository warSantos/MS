from cProfile import label
from sys import exit
import os
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from itertools import product

# Loading local libraries.
from src.models.models import ALIAS
from src.models.optimization import execute_optimization
from src.train_probas.train_probas import build_train_probas
from src.models.calibration import probabilities_calibration
from joblib import load, dump



# Warning Control.
import warnings
from optuna.exceptions import ExperimentalWarning
from optuna.logging import set_verbosity, WARNING

set_verbosity(WARNING)
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect sub-processes


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
NESTED_FOLDS = settings["NESTED_FOLDS"]
CALIBRATION = settings["CALIBRATION"]
if CALIBRATION:
    normal_or_calib = "calibrated_probabilities"
else:
    normal_or_calib = "normal_probabilities"
BUILD_TRAIN_PROBAS = settings["BUILD_TRAIN_PROBAS"]
LOAD_MODEL = settings["LOAD_MODEL"]

iterations = product(
    DATASETS,
    CLFS,
    REP,
    SPLIT_SETTINGS,
    range(N_FOLDS))


for dset, clf, rep, sp_setting, fold in iterations:

    print(f" {dset.upper()} - [{clf.upper()} / {rep.upper()}] - FOLD: {fold}")

    # Loading representations.
    reps_dir = f"{DATA_SOURCE}/representations/{dset}/{N_FOLDS}_folds/{rep}/{fold}"
    train_load = np.load(f"{reps_dir}/train.npz", allow_pickle=True)
    test_load = np.load(f"{reps_dir}/test.npz", allow_pickle=True)
    
    X_train = train_load["X_train"].tolist().toarray()
    X_test = test_load["X_test"].tolist().toarray()
    
    y_train = train_load["y_train"]
    y_test = test_load["y_test"]
    
    output_dir = f"{DATA_SOURCE}/{normal_or_calib}/{sp_setting}/{dset}/{N_FOLDS}_folds/{ALIAS[f'{clf}/{rep}']}/{fold}"
    model_path = f"{output_dir}/model.joblib"
    os.makedirs(output_dir, exist_ok=True)
    
    X_test_path = f"{output_dir}/test"
    # If test probas weren't computed yet.
    if not os.path.exists(f"{X_test_path}.npz"):
        optuna_search = execute_optimization(
            classifier_name=clf,
            file_model=model_path,
            X_train=X_train,
            y_train=y_train,
            opt_n_jobs=1,
            clf_n_jobs=N_JOBS,
            load_model=LOAD_MODEL,
            calibration=CALIBRATION)
        
        probas = optuna_search.predict_proba(X_test)
        y_pred = probas.argmax(axis=1)
        
        np.savez(X_test_path, X_test=probas)
        
        scoring = {}

        scoring["macro"] = f1_score(y_test, y_pred, average="macro")
        scoring["micro"] = f1_score(y_test, y_pred, average="micro")
        scoring["accuracy"] = accuracy_score(y_test, y_pred)

        print(f"\n\tMacro: {scoring['macro'] * 100}")
        print(f"\tMicro: {scoring['micro'] * 100}\n")

        with open(f"{output_dir}/scoring.json", "w") as fd:
            json.dump(scoring, fd)

    if BUILD_TRAIN_PROBAS:
        print("\tBuilding Train Probas.")
        sub_splits_path = f"{output_dir}/sub_splits"
        train_probas_path = f"{output_dir}/train"
        # If train probs weren't computed yet.
        if not os.path.exists(f"{train_probas_path}.npz"):
            X_train_probas = build_train_probas(X_train, 
                y_train, 
                clf, 
                n_splits=NESTED_FOLDS, 
                n_jobs=N_JOBS, 
                output_dir=sub_splits_path,
                load_model=LOAD_MODEL,
                calibration=CALIBRATION)
            np.savez(train_probas_path, X_train=X_train_probas)

# Loading Labels.
#labels_dir = f"{DATA_SOURCE}/datasets/labels/{sp_setting}/{dset}/{fold}/"
#y_train = np.load(f"{labels_dir}/train.npy", allow_pickle=True)
#y_test = np.load(f"{labels_dir}/test.npy", allow_pickle=True)
