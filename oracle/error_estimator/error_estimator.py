import os
import json
import pickle
import warnings
import numpy as np
from joblib import dump
from itertools import product

from optuna.logging import set_verbosity, WARNING
from optuna.exceptions import ExperimentalWarning
from sklearn.exceptions import ConvergenceWarning

from mfs import make_mfs
from scoring import get_scores, get_dict_score, save_scores
from feature_selection import fast_feature_selection, feature_selection

from sys import path
path.append("../")
from local_utils import load_clfs_probas

# Configs Optuna

set_verbosity(WARNING)
warnings.filterwarnings("ignore", category=ExperimentalWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")


def local_error_estimation(data_source: str,
                           dataset: str,
                           probas: dict,
                           name_estimator: str,
                           CLFS: list,
                           meta_features_set: list,
                           speed_feat_selection: str,
                           do_optmization: bool,
                           run_estimator: bool,
                           redo_mfs: bool,
                           n_folds: int):

    oracle_dir = f"{data_source}/oracle"
    mf_set_label = '_'.join(sorted(meta_features_set))
    clf_sufix = '/'.join(sorted([f"{c[0]}_{c[1]}" for c in CLFS]))
    for fold in np.arange(n_folds):

        # Loading labels.
        y_train = np.load(
            f"{data_source}/datasets/labels/split_{n_folds}/{dataset}/{fold}/train.npy")
        y_test = np.load(
            f"{data_source}/datasets/labels/split_{n_folds}/{dataset}/{fold}/test.npy")

        # For each Stacking base model.
        for target_clf, proba_type in CLFS:
            print(f"TARGET CLF: {target_clf}")
            # Buiding logdirs.
            output_dir = f"{oracle_dir}/local_{name_estimator}/{proba_type}/{dataset}/{n_folds}_folds/{target_clf}/{clf_sufix}/{mf_set_label}/{fold}"
            print(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            # Building Meta-Features.
            mf_input_dir = f"{data_source}/meta_features/error_estimation/{dataset}/{n_folds}_folds/{clf_sufix}/{mf_set_label}/{target_clf}/{fold}"
            X_train, X_test, upper_train, upper_test, features_labels = make_mfs(data_source,
                                                                                 dataset,
                                                                                 target_clf,
                                                                                 clf_sufix,
                                                                                 probas,
                                                                                 y_train,
                                                                                 y_test,
                                                                                 fold,
                                                                                 n_folds,
                                                                                 meta_features_set,
                                                                                 redo_mfs)

            # Check if it must run the estimator, otherwise will just build MFs.
            if run_estimator:
                # Featuring selection.
                model_path = f"{output_dir}/{name_estimator}"
                forest_path = f"{output_dir}/forest"

                best_feats, f1, forest, error_estimator = fast_feature_selection(X_train,
                                                                                 X_test,
                                                                                 upper_train,
                                                                                 upper_test,
                                                                                 name_estimator,
                                                                                 forest_path,
                                                                                 do_optmization)

                ranking = (1 - forest.feature_importances_).argsort()
                best_feats_set = ranking[:best_feats]

                # Saving models.
                dump(forest, forest_path)
                dump(error_estimator, model_path)

                # Saving optimal number of features.
                with open(f"{output_dir}/fs.json", 'w') as fd:
                    json.dump({"best_feats": best_feats,
                               "selection_type": speed_feat_selection}, fd)

                # Prediction
                pred_probas = error_estimator.predict_proba(
                    X_test[:, best_feats_set])
                y_pred = pred_probas.argmax(axis=1)
                # Genarating scores.
                pc, rc, f1, acc = get_scores(upper_test, y_pred)

                print(
                    f"DATASET: {dataset.upper()} / CLF: {target_clf} / FOLD - {fold} - Prec: {pc:.2f}, Rec: {rc:.2f}, F1: {f1:.2f}, Acc: {acc:.2f}\n")
                # Saving scores.
                dict_scores = get_dict_score(pc, rc, f1, acc)
                save_scores(output_dir, dict_scores)

                # Saving the error estimation setting.
                np.savez(f"{output_dir}/test", y=pred_probas[:, 1])
                # Saving upper_train.
                np.savez(f"{output_dir}/train", y=upper_train)

                if features_labels:
                    with open(f"{output_dir}/features_labels.pkl", "wb") as fd:
                        pickle.dump(features_labels, fd)


if __name__ == "__main__":

    with open("data/settings.json", 'r') as fd:
        settings = json.load(fd)

    DATA_SOURCE = settings["DATA_SOURCE"]
    DATASETS = settings["DATASETS"]
    ERROR_ESTIMATOR = settings["ERROR_ESTIMATOR"]
    MFS_SET = settings["MFS_SET"]
    SPEED_FS = settings["SPEED_FS"]
    DO_OPTMIZATION = settings["DO_OPTMIZATION"]
    SETS = settings["SETS"]
    RUN_ESTIMATOR = settings["RUN_ESTIMATOR"]
    REDO_MFS = settings["REDO_MFS"]

    for (dataset, n_folds), set_clf in product(DATASETS, SETS):

        CLFS_SET = settings[set_clf]
        probas = load_clfs_probas(DATA_SOURCE,
                                  dataset,
                                  CLFS_SET,
                                  n_folds,
                                  "train_test")

        local_error_estimation(DATA_SOURCE,
                               dataset,
                               probas,
                               ERROR_ESTIMATOR,
                               CLFS_SET,
                               MFS_SET,
                               SPEED_FS,
                               DO_OPTMIZATION,
                               RUN_ESTIMATOR,
                               REDO_MFS,
                               n_folds)
