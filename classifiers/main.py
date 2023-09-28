from cProfile import label
from sys import exit
import os
import json
import boto3
import numpy as np
from itertools import product
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# Loading local libraries.
from src.models.models import ALIAS, fix_labels
from src.models.calibration import probabilities_calibration
from src.models.optimization import execute_optimization
from src.train_probas.train_probas import build_train_probas
from src.aws.awsio import (load_reps_from_aws,
                           store_nparrays_in_aws,
                           store_json_in_aws,
                           aws_path_exists,
                           aws_stop_instance)

from src.utils.utils import report_scoring, replace_nan

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
    raise ("file data/setup/settings.json not found.")


SEED = 42
DATASETS = settings["DATASETS"]
WITH_VAL = settings["WITH_VAL"]
CLFS = settings["CLFS"]
REP = settings["REP"]
DATA_SOURCE = settings["DATA_SOURCE"]
SAVE_PROBAS_CALIB = settings["SAVE_PROBAS_CALIB"]
NESTED_FOLDS = settings["NESTED_FOLDS"]
DO_TEST = settings["DO_TEST"]
DO_TRAIN = settings["DO_TRAIN"]
DO_TEST_CALIB = settings["DO_TEST_CALIB"]
DO_TRAIN_CALIB = settings["DO_TRAIN_CALIB"]
LOAD_MODEL = settings["LOAD_MODEL"]
DO_OPT = settings["DO_OPT"]
CALIB_METHOD = settings["CALIB_METHOD"]
CLFS_SETUP = settings["CLFS_SETUP"]
AWS_BUCKET = settings["AWS_BUCKET"]
AWS_PROFILE = settings["AWS_PROFILE"]
os.environ["DATA_SOURCE"] = DATA_SOURCE
os.environ["AWS_BUCKET"] = settings["AWS_BUCKET"]
os.environ["AWS_PROFILE"] = settings["AWS_PROFILE"]

probas_dir = "normal_probas"

iterations = product(
    DATASETS,
    CLFS,
    REP)

for dataset_setup, clf, rep in iterations:
    dataset, n_folds, fold_list = dataset_setup
    start, end = fold_list
    for fold in range(start, end):

        sp_setting = f"split_{n_folds}{WITH_VAL}"

        N_JOBS = CLFS_SETUP[clf]["n_jobs"]
        OPT_N_JOBS = CLFS_SETUP[clf]["opt_n_jobs"]

        print(f"{dataset.upper()} - [{clf.upper()} / {rep.upper()}] - FOLD: {fold}")

        # Loading representations.
        #reps_dir = f"{DATA_SOURCE}/representations/{dataset}/{n_folds}_folds/{rep}/{fold}"
        #train_load = np.load(f"{reps_dir}/train.npz", allow_pickle=True)
        #test_load = np.load(f"{reps_dir}/test.npz", allow_pickle=True)

        reps_dir = f"representations/{dataset}/{n_folds}_folds/{rep}/{fold}"
        train_load = load_reps_from_aws(f"{reps_dir}/train.npz", "train")
        test_load = load_reps_from_aws(f"{reps_dir}/test.npz", "test")

        if dataset == "mini_20ng":
            full_X_train = replace_nan(train_load["X_train"])
            X_test = replace_nan(test_load["X_test"])
        else:
            full_X_train = replace_nan(train_load["X_train"].tolist())
            X_test = replace_nan(test_load["X_test"].tolist())
        

        full_y_train = fix_labels(train_load["y_train"])
        y_test = fix_labels(test_load["y_test"])

        """
        if X_test.shape[1] < 2000:
            X_test = X_test.toarray()
            full_X_train = full_X_train.toarray()
        """
                    
        if DO_TEST_CALIB:
            X_train, X_val, y_train, y_val = train_test_split(full_X_train,
                                                            full_y_train,
                                                            random_state=SEED,
                                                            stratify=full_y_train,
                                                            test_size=0.10)
        else:
            X_train, y_train = full_X_train, full_y_train

        output_dir = f"{DATA_SOURCE}/{probas_dir}/{sp_setting}/{dataset}/{n_folds}_folds/{ALIAS[f'{clf}/{rep}']}/{fold}"
        os.makedirs(output_dir, exist_ok=True)

        # If test probas weren't computed yet.
        #X_test_path = f"{output_dir}/test"
        X_test_path = f"{output_dir}/test.npz"
        #if DO_TEST and not os.path.exists(f"{X_test_path}.npz"):
        if DO_TEST and not aws_path_exists(X_test_path):

            estimator = execute_optimization(
                classifier_name=clf,
                output_dir=output_dir,
                X_train=X_train,
                y_train=y_train,
                opt_n_jobs=OPT_N_JOBS,
                clf_n_jobs=N_JOBS,
                load_model=LOAD_MODEL,
                do_optimization=DO_OPT)

            probas = estimator.predict_proba(X_test)
            y_pred = probas.argmax(axis=1)
            #np.savez(X_test_path, X_test=probas)
            store_nparrays_in_aws(X_test_path,
                                  {"X_test": probas})
            print("Normal.")
            report_scoring(y_test, y_pred, output_dir)

            if SAVE_PROBAS_CALIB:
                probas_val = estimator.predict_proba(X_test)
                #np.savez(X_test_path.replace('test', 'eval'), X_eval=probas_val)
                store_nparrays_in_aws(X_test_path.replace('test', 'eval'),
                                      {"X_eval": probas_val})

            if DO_TEST_CALIB:
                calib_output_dir = f"{DATA_SOURCE}/{CALIB_METHOD}/{sp_setting}/{dataset}/{n_folds}_folds/{ALIAS[f'{clf}/{rep}']}/{fold}"
                os.makedirs(calib_output_dir, exist_ok=True)
                calib_est = probabilities_calibration(
                    estimator, X_val, y_val, CALIB_METHOD)
                c_probas = calib_est.predict_proba(X_test)
                #np.savez(f"{calib_output_dir}/test",
                #        X_test=c_probas, y_test=y_test)
                store_nparrays_in_aws(f"{calib_output_dir}/test.npz",
                                      {"X_test":c_probas, "y_test": y_test})
                
                print("Calibrated.")
                report_scoring(y_test, c_probas.argmax(axis=1), calib_output_dir)

        #train_probas_path = f"{output_dir}/train"
        train_probas_path = f"{output_dir}/train.npz"
        #if DO_TRAIN and not os.path.exists(f"{train_probas_path}.npz"):
        if DO_TRAIN and not aws_path_exists(train_probas_path):

            print("\tBuilding Train Probas.")
            X_train_probas = build_train_probas(
                clf,
                reps_dir,
                output_dir,
                SAVE_PROBAS_CALIB,
                DO_TRAIN_CALIB,
                CALIB_METHOD,
                n_splits=NESTED_FOLDS,
                n_jobs=N_JOBS,
                opt_n_jobs=OPT_N_JOBS,
                load_model=LOAD_MODEL,
                do_optimization=DO_OPT)
            
            #np.savez(train_probas_path,
            #        X_train=X_train_probas["probas"], y_train=full_y_train)
            store_nparrays_in_aws(train_probas_path,
                                  {"X_train": X_train_probas["probas"], "y_train": full_y_train})

            if X_train_probas["calib_probas"] is not None:
                calib_output_dir = f"{DATA_SOURCE}/{CALIB_METHOD}/{sp_setting}/{dataset}/{n_folds}_folds/{ALIAS[f'{clf}/{rep}']}/{fold}"
                #np.savez(f"{calib_output_dir}/train",
                #        X_train=X_train_probas["calib_probas"], y_train=full_y_train)
                store_nparrays_in_aws(f"{calib_output_dir}/train.npz",
                                  {"X_train": X_train_probas["calib_probas"], "y_train": full_y_train})
                

# Stoping instance.
aws_stop_instance()