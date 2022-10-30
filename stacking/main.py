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

# Loading basic settings of the envirioment.
try:
    fd = open("data/settings.json", 'r')
    settings = json.load(fd)
except:
    traceback.print_exc()
    print("Was not possible to read basic enviorioment settings.")
    exit(1)

DATA_SOURCE = settings["DATA_SOURCE"]

from src.constants import IDS_MODELS, REP_CLFS, MIX_REPS_MINUS_BERT
from src.files import (
                        read_train_test_meta,
                        read_train_test_bert,
                        load_y,
                        read_train_test_meta_oracle
                    )
from src.stacking.input_types import read_mfs
from src.optimization import execute_optimization
from src.feature_selection.feature_importance import FeatureSelector

# Lib configs
set_verbosity(WARNING)
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect sub-processes

# Directories and files
MFS_DIR = f"{DATA_SOURCE}/meta_features"
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
WITH_PROBA = execution["WITH_PROBA"]
MF_COMBINATION = execution["MF_COMBINATION"]
NUM_FEATS = execution["NUM_FEATS"]
BERT_STACKING = execution["BERT_STACKING"]
SPLIT_SETTINGS = execution["SPLIT_SETTINGS"]
DIR_META_INPUT = f"{DATA_SOURCE}/clfs_output/{SPLIT_SETTINGS}"

if __name__ == "__main__":
    iterations = itertools.product(META_FEATURES, META_LAYERS, DATASETS, NUM_FEATS, range(N_FOLDS))
    for (meta_feature, meta_layer, dataset, num_feats, fold_id) in iterations:
        print(f"[{meta_layer.upper()} / {meta_feature.upper()}] - {dataset.upper():10s} - fold_{fold_id}")
    
        # Reading classification labels.
        y_train, y_test = load_y(DATA_SOURCE, dataset, fold_id, SPLIT_SETTINGS)
        
        dir_dest = f"{DIR_OUTPUT}/{dataset}/{N_FOLDS}_folds/"

        # Reading meta-layer input
        if meta_feature == "proba":
            X_train, X_test = read_train_test_meta(DIR_META_INPUT, dataset, N_FOLDS, fold_id, MODELS)
            dir_dest += f"{meta_layer}/{meta_feature}/fold_{fold_id}"
        elif meta_feature == "mix_reps":
            X_train, X_test = read_train_test_meta(DIR_META_INPUT, dataset, N_FOLDS, fold_id, REP_CLFS)
            dir_dest += f"{meta_layer}/{meta_feature}/fold_{fold_id}"
        elif meta_feature == "mix_reps_without_bert":
            X_train, X_test = read_train_test_meta(DIR_META_INPUT, dataset, N_FOLDS, fold_id, MIX_REPS_MINUS_BERT)
            dir_dest += f"{meta_layer}/{meta_feature}/fold_{fold_id}"
        elif meta_feature == "oracle_upper":
            oracle_path = f"{DATA_SOURCE}/oracle"
            X_train, X_test = read_train_test_meta_oracle(DIR_META_INPUT, 
                                                            dataset,
                                                            N_FOLDS,
                                                            fold_id,
                                                            REP_CLFS,
                                                            oracle_path,
                                                            "upper_bound")

            dir_dest += f"{meta_layer}/{meta_feature}/fold_{fold_id}"
        elif BERT_STACKING:
            X_train, X_test = read_train_test_bert(DATA_SOURCE, dataset, BERT_STACKING, N_FOLDS, fold_id)
            dir_dest += f"/{SPLIT_SETTINGS}/{meta_feature}/{fold_id}"
        # If is there any MF to load.
        elif META_FEATURES:
            meta_feature_path = f"{MFS_DIR}/{meta_feature}"
            X_train, X_test = read_mfs(DIR_META_INPUT, meta_feature_path, dataset, N_FOLDS, fold_id, MODELS, MF_COMBINATION, load_proba=WITH_PROBA)
            dir_dest += f"""
            {dir_dest}
            {meta_layer}/
            num_feats/{nf}/
            with_proba/{WITH_PROBA}/
            combination/{MF_COMBINATION}/
            {meta_feature}/
            fold_{fold_id}""".replace(' ','').replace('\n','')
        else:
            raise ValueError(f"Invalid value ({meta_feature}) for type_input.")

        print(f"[OUTPUT: {dir_dest}]")
        # Verify if feature selection must be applied.
        nf = 18 * len(set(y_train))
        if num_feats > -1:
            fs = FeatureSelector()
            settings_path = f"{dataset}/{meta_feature}/{MF_COMBINATION}/{fold_id}"
            feat_ranking = fs.feature_importance(settings_path, X_train, y_train, n_feats=nf)
            X_train = np.take(X_train, feat_ranking, axis=1)
            X_test = np.take(X_test, feat_ranking, axis=1)
        
        # Optimization/Training.

        file_model = f"{dir_dest}/model.joblib"
        os.makedirs(dir_dest, exist_ok=True)
        optuna_search = execute_optimization(
            meta_layer,
            file_model,
            X_train,
            y_train,
            opt_n_jobs=N_JOBS,
            load_model=False
        )

        # Prediction
        y_pred = optuna_search.predict(X_test)
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
        
        # Saving the prediction values.
        y_pred_file = f"{dir_dest}/y_pred"
        np.save(y_pred_file, y_pred, allow_pickle=True)