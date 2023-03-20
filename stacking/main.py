# Python libraries
import os
import json
import warnings
import itertools
import traceback

# 3rd party libraries
import pandas as pd
import numpy as np
from optuna.logging import set_verbosity, WARNING
from sklearn.metrics import f1_score

# Local libraries
from src.optimization import execute_optimization
from src.loader.loader import read_meta_data
from src.files import (
    read_train_test_meta,
    load_y,
    read_train_test_meta_oracle
)
from src.constants import IDS_MODELS

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

# Execution configs
MODELS = list(IDS_MODELS.values())  # All 18 models

# Loading basic settings of the envirioment.
try:
    fd = open("data/execution.json", 'r')
    execution = json.load(fd)
except:
    traceback.print_exc()
    print("Was not possible to read executions settings.")
    exit(1)

DATASETS = execution["DATASETS"]
N_FOLDS = execution["N_FOLDS"]
META_LAYERS = execution["META_LAYERS"]
META_FEATURES = execution["META_FEATURES"]
N_JOBS = execution["N_JOBS"]
SEED = execution["SEED"]

SPLIT_SETTINGS = execution["SPLIT_SETTINGS"]
LOAD_MODEL = execution["LOAD_MODEL"]
CONFIDENCE_STEPS = execution["CONFIDENCE_STEPS"]

DIR_META_INPUT = f"{DATA_SOURCE}/clfs_output/{SPLIT_SETTINGS}"

if __name__ == "__main__":
    iterations = itertools.product(
        META_FEATURES, META_LAYERS, DATASETS, CONFIDENCE_STEPS, range(N_FOLDS))
    for (meta_feature, meta_layer, dataset, confidence, fold_id) in iterations:
        print(
            f"[{meta_layer.upper()} / {meta_feature.upper()}] - {dataset.upper():10s} - fold_{fold_id}")

        # Reading classification labels.
        y_train, y_test = load_y(DATA_SOURCE, dataset, fold_id, SPLIT_SETTINGS)

        # Setting output dir.
        dir_dest = f"{DIR_OUTPUT}/{dataset}/{N_FOLDS}_folds/{meta_layer}/{meta_feature}"

        # Reading meta-layer input
        if meta_feature == "proba":

            X_train, X_test = read_train_test_meta(
                DIR_META_INPUT, dataset, N_FOLDS, fold_id, ["bert", "xlnet", "ktmk"])

        elif meta_feature in ["calibrated", "normal_probas"]:
            X_train, X_test = read_meta_data(
                DATA_SOURCE,
                dataset,
                execution["CLF_SET"],
                N_FOLDS,
                fold_id
            )

        elif meta_feature.find("local_") > -1 or meta_feature == "upper_bound":
            oracle_set = execution["ORACLE_CLF_SET"]
            zero_train = execution["ORACLE_ZERO_TRAIN"]
            X_train, X_test = read_train_test_meta_oracle(DATA_SOURCE,
                                                          dataset,
                                                          N_FOLDS,
                                                          fold_id,
                                                          oracle_set,
                                                          meta_feature,
                                                          confidence,
                                                          zero_train)

            sufix = '/'.join(sorted([ f"{tup[0]}_{tup[1]}" for tup in oracle_set ]))
            sufix = f"zero_train_{zero_train}/{confidence}/{sufix}/{oracle_set[0][2]}"
            dir_dest = f"{dir_dest}/{sufix}"
        else:
            raise ValueError(f"Invalid value ({meta_feature}) for type_input.")

        print(f"[OUTPUT: {dir_dest}]")

        dir_dest = f"{dir_dest}/fold_{fold_id}"
        # Optimization/Training.

        file_model = f"{dir_dest}/model.joblib"
        os.makedirs(dir_dest, exist_ok=True)
        optuna_search = execute_optimization(
            meta_layer,
            file_model,
            X_train,
            y_train,
            seed=SEED,
            opt_n_jobs=N_JOBS,
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
