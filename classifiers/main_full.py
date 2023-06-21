from cProfile import label
from sys import exit
import os
import json
import numpy as np
from scipy import sparse
from itertools import product
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# Loading local libraries.
from src.models.models import ALIAS
from src.models.calibration import probabilities_calibration
from src.models.optimization import execute_optimization
from src.train_probas.train_probas import build_train_probas

# Warning Control.
import warnings
from optuna.exceptions import ExperimentalWarning
from optuna.logging import set_verbosity, WARNING

set_verbosity(WARNING)
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect sub-processes


def report_scoring(y_test, y_pred, output_dir):

    scoring = {}
    scoring["macro"] = f1_score(y_test, y_pred, average="macro")
    scoring["micro"] = f1_score(y_test, y_pred, average="micro")
    scoring["accuracy"] = accuracy_score(y_test, y_pred)

    print(f"\n\tMacro: {scoring['macro'] * 100}")
    print(f"\tMicro: {scoring['micro'] * 100}\n")

    with open(f"{output_dir}/scoring.json", "w") as fd:
        json.dump(scoring, fd)


try:
    with open("data/settings.json", 'r') as fd:
        settings = json.load(fd)
except:
    raise ("file data/setup/settings.json not found.")


SEED = 42
DATASETS = settings["DATASETS"]
CLFS = settings["CLFS"]
REP = settings["REP"]
DATA_SOURCE = settings["DATA_SOURCE"]
SPLIT_SETTINGS = settings["SPLIT_SETTINGS"]
SAVE_PROBAS_CALIB = settings["SAVE_PROBAS_CALIB"]
N_FOLDS = settings["N_FOLDS"]
NESTED_FOLDS = settings["NESTED_FOLDS"]
DO_TEST = settings["DO_TEST"]
DO_TRAIN = settings["DO_TRAIN"]
DO_TEST_CALIB = settings["DO_TEST_CALIB"]
DO_TRAIN_CALIB = settings["DO_TRAIN_CALIB"]
LOAD_MODEL = settings["LOAD_MODEL"]
CALIB_METHOD = settings["CALIB_METHOD"]
CLFS_SETUP = settings["CLFS_SETUP"]

probas_dir = "normal_probas"

iterations = product(
    DATASETS,
    CLFS,
    REP,
    SPLIT_SETTINGS,
    range(N_FOLDS))


for dset, clf, rep, sp_setting, fold in iterations:

    N_JOBS = CLFS_SETUP[clf]["n_jobs"]
    OPT_N_JOBS = CLFS_SETUP[clf]["opt_n_jobs"]

    print(f" {dset.upper()} - [{clf.upper()} / {rep.upper()}] - FOLD: {fold}")

    # Loading representations.
    reps_dir = f"{DATA_SOURCE}/representations/{dset}/{N_FOLDS}_folds/{rep}/{fold}"
    train_load = np.load(f"{reps_dir}/train.npz", allow_pickle=True)
    test_load = np.load(f"{reps_dir}/test.npz", allow_pickle=True)

    try:
        if rep == "tr":
            print("Loading Sparse TF-IDF")
            X_train = train_load["X_train"]#.tolist()
            X_test = test_load["X_test"]#.tolist()
        else:
            X_train = train_load["X_train"].tolist().toarray()
            X_test = test_load["X_test"].tolist().toarray()
            print(f"Loading nd.array {rep}")
    except:
        X_train = sparse.csr_matrix(train_load["X_train"])
        X_test = sparse.csr_matrix(test_load["X_test"])

    y_train = train_load["y_train"]
    y_test = test_load["y_test"]

    output_dir = f"{DATA_SOURCE}/{probas_dir}/{sp_setting}/{dset}/{N_FOLDS}_folds/{ALIAS[f'{clf}/{rep}']}/{fold}"
    os.makedirs(output_dir, exist_ok=True)

    # If test probas weren't computed yet.
    if DO_TEST:

        X_test_path = f"{output_dir}/test"
        estimator = execute_optimization(
            classifier_name=clf,
            output_dir=output_dir,
            X_train=X_train,
            y_train=y_train,
            opt_n_jobs=OPT_N_JOBS,
            clf_n_jobs=N_JOBS,
            load_model=LOAD_MODEL)

        probas = estimator.predict_proba(X_test)
        y_pred = probas.argmax(axis=1)
        np.savez(X_test_path, X_test=probas)
        print("Normal.")
        report_scoring(y_test, y_pred, output_dir)

        if DO_TEST_CALIB:
            calib_output_dir = f"{DATA_SOURCE}/{CALIB_METHOD}/{sp_setting}/{dset}/{N_FOLDS}_folds/{ALIAS[f'{clf}/{rep}']}/{fold}"
            os.makedirs(calib_output_dir, exist_ok=True)
            
            calib_est = probabilities_calibration(clf,
                                                  estimator.get_params(), 
                                                  X_train, 
                                                  y_train, 
                                                  CALIB_METHOD,
                                                  10)
            
            c_probas = calib_est.predict_proba(X_test)
            np.savez(f"{calib_output_dir}/test",
                     X_test=c_probas, y_test=y_test)
            print("Calibrated.")
            report_scoring(y_test, c_probas.argmax(axis=1), calib_output_dir)

    if DO_TRAIN:

        print("\tBuilding Train Probas.")
        train_probas_path = f"{output_dir}/train"
        X_train_probas = build_train_probas(clf,
                                            output_dir,
                                            X_train,
                                            y_train,
                                            DO_TRAIN_CALIB,
                                            CALIB_METHOD,
                                            n_splits=NESTED_FOLDS,
                                            n_jobs=N_JOBS,
                                            opt_n_jobs=OPT_N_JOBS,
                                            load_model=LOAD_MODEL)
        
        np.savez(train_probas_path,
                 X_train=X_train_probas["probas"], y_train=y_train)

        if X_train_probas["calib_probas"] is not None:
            calib_output_dir = f"{DATA_SOURCE}/{CALIB_METHOD}/{sp_setting}/{dset}/{N_FOLDS}_folds/{ALIAS[f'{clf}/{rep}']}/{fold}"
            np.savez(f"{calib_output_dir}/train",
                     X_train=X_train_probas["calib_probas"], y_train=y_train)