# Python libraries
import os
import json
import warnings
from itertools import product

# 3rd party libraries
import pandas as pd
import numpy as np
from optuna.logging import set_verbosity, WARNING
from sklearn.metrics import f1_score

# Local libraries
from src.optimization import execute_optimization
from src.loader.loader import read_meta_data
from src.files import (
    load_y,
    read_train_test_meta_oracle,
    read_cost_error_estimation
)

# Lib configs
set_verbosity(WARNING)
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect sub-processes

# Loading basic settings of the envirioment.
try:
    fd = open("data/new_settings.json", 'r')
    settings = json.load(fd)
except:
    print("Was not possible to read the experiments's settings.")
    exit(1)

# Directories and files
DATA_SOURCE = settings["DATA_SOURCE"]
DIR_OUTPUT = f"{DATA_SOURCE}/stacking/stacking_output"

# Experiments's parameters.
DATASETS = settings["DATASETS"]
N_JOBS = settings["N_JOBS"]
SEED = settings["SEED"]
SPLIT_SETTINGS = settings["SPLIT_SETTINGS"]
LOAD_MODEL = settings["LOAD_MODEL"]
meta_layer = settings["META_LAYERS"]

for experiment, dataset_setup in product(settings["EXPERIMENTS"], DATASETS):
    
    dataset, n_folds = dataset_setup
    C_SETS = settings["CLFS_SETS"][settings["EXPERIMENT_OPTIONS"][experiment]["CLFS_SETS"]]
    for clf_set, fold in product(C_SETS, np.arange(n_folds)):

        print(
            f"[{meta_layer.upper()} / {experiment.upper()}] - {dataset.upper():10s} - fold_{fold}")

        # Reading classification labels.
        y_train, y_test = load_y(DATA_SOURCE, dataset, fold, SPLIT_SETTINGS)

        # Setting output dir.
        dir_dest = f"{DIR_OUTPUT}/{dataset}/{n_folds}_folds/{meta_layer}/{experiment}"



        # TODO: Falta arrumar o código de leitura do upperbound e error_detection.


        if experiment in ["calibrated", "normal_probas"]:
        
            X_train, X_test = read_meta_data(
                DATA_SOURCE,
                dataset,
                settings[experiment],
                n_folds,
                fold
            )
            sufix = '/'.join(sorted([ f"{tup[0]}_{tup[1]}" for tup in settings[experiment] ]))
            dir_dest = f"{dir_dest}/{sufix}"

        elif experiment == "error_detection":
            
            oracle_set = settings[experiment][clf_set]
            zero_train = settings["ORACLE_ZERO_TRAIN"]
            sufix = '/'.join(sorted([ f"{tup[0]}_{tup[1]}" for tup in oracle_set ]))
            X_train, X_test = read_train_test_meta_oracle(DATA_SOURCE,
                                                        dataset,
                                                        N_FOLDS,
                                                        fold,
                                                        oracle_set,
                                                        experiment,
                                                        sufix,
                                                        confidence,
                                                        zero_train)

            sufix = f"{sufix}/{oracle_set[0][2]}"
            dir_dest = f"{dir_dest}/{sufix}"
        
        elif experiment == "upper_bound":

            sufix = '/'.join(sorted([ f"{tup[0]}_{tup[1]}" for tup in oracle_set ]))
            X_train, X_test = read_train_test_meta_oracle(DATA_SOURCE,
                                                        dataset,
                                                        N_FOLDS,
                                                        fold,
                                                        oracle_set,
                                                        experiment,
                                                        sufix,
                                                        confidence,
                                                        zero_train)

            sufix = f"{sufix}/{oracle_set[0][2]}"
            dir_dest = f"{dir_dest}/{sufix}"

        elif experiment in ["zero_error_cost", "keep_error_cost"]:

            oracle_set = settings["error_detection"]
            zero_train = settings["ORACLE_ZERO_TRAIN"]
            X_train, X_test = read_cost_error_estimation(DATA_SOURCE,
                                                        dataset,
                                                        N_FOLDS,
                                                        fold,
                                                        oracle_set,
                                                        (experiment, "error_detection"),
                                                        confidence,
                                                        zero_train)

            sufix = '/'.join(sorted([ f"{tup[0]}_{tup[1]}" for tup in oracle_set ]))
            sufix = f"zero_train_{zero_train}/{confidence}/{sufix}/{oracle_set[0][2]}"
            dir_dest = f"{dir_dest}/{sufix}"
        
        else:
            raise ValueError(f"Invalid value ({experiment}) for type_input.")

        print(f"[OUTPUT: {dir_dest}]")

        dir_dest = f"{dir_dest}/fold_{fold}"
        # Optimization/Training.

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

        # Prediction
        probas = optuna_search.predict_proba(X_test)
        y_pred = probas.argmax(axis=1)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_micro = f1_score(y_test, y_pred, average="micro")

        msg = f"""
        \tF1-Macro: {f1_macro:.4f}
        \tF1-Micro: {f1_micro:.4f}
        """
        print(msg)

        # Save prediction
        file_scoring = f"{dir_dest}/scoring.json"
        scoring = {
            "f1_macro": f1_macro,
            "f1_micro": f1_micro
        }
        with open(file_scoring, "w+", encoding="utf-8") as fp:
            json.dump(scoring, fp, ensure_ascii=False, indent=4)

        # Saving probabilities.
        probas_path = f"{dir_dest}/probas"
        np.savez(probas_path, X_test=probas, allow_pickle=True)
