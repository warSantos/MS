import os
import json
import warnings
import numpy as np
from joblib import dump

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
                           N_FOLDS: int = 10):

    oracle_dir = f"{data_source}/oracle"
    for fold in np.arange(10):

        # Loading labels.
        y_train = np.load(
            f"{data_source}/datasets/labels/split_10/{dataset}/{fold}/train.npy")
        y_test = np.load(
            f"{data_source}/datasets/labels/split_10/{dataset}/{fold}/test.npy")

        mf_set_label = '_'.join(sorted(meta_features_set))
        # For each Stacking base model.
        for target_clf, proba_type in CLFS:
            print(f"TARGET CLF: {target_clf}")
            # Buiding logdirs.
            output_dir = f"{oracle_dir}/local_{name_estimator}/{proba_type}/{dataset}/{N_FOLDS}_folds/{target_clf}/{mf_set_label}/{fold}"
            print(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            # Building Meta-Features.
            X_train, X_test, upper_train, upper_test = make_mfs(data_source,
                                                                dataset,
                                                                target_clf,
                                                                probas,
                                                                y_train,
                                                                y_test,
                                                                fold,
                                                                meta_features_set)

            # Featuring selection.
            model_path = f"{output_dir}/{name_estimator}"
            forest_path = f"{output_dir}/forest"

            if speed_feat_selection == "fast":
                print("Fast Feature Selection")
                best_feats, f1, forest, error_estimator = fast_feature_selection(X_train,
                                                                                 X_test,
                                                                                 upper_train,
                                                                                 upper_test,
                                                                                 name_estimator,
                                                                                 forest_path,
                                                                                 do_optmization)
            else:
                best_feats, forest, error_estimator = feature_selection(
                    X_train, X_test, upper_train, upper_test, name_estimator)

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
            pred_probas = error_estimator.predict_proba(X_test[:, best_feats_set])
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


if __name__ == "__main__":

    with open("data/settings.json", 'r') as fd:
        settings = json.load(fd)

    DATA_SOURCE = settings["DATA_SOURCE"]
    DATASETS = settings["DATASETS"]
    CLFS_SET = settings["CLFS_SET"]
    ERROR_ESTIMATOR = settings["ERROR_ESTIMATOR"]
    MFS_SET = settings["MFS_SET"]
    SPEED_FS = settings["SPEED_FS"]
    DO_OPTMIZATION = settings["DO_OPTMIZATION"]

    for dataset in DATASETS:

        probas = load_clfs_probas(DATA_SOURCE,
                                  dataset,
                                  CLFS_SET,
                                  "train_test")

        local_error_estimation(DATA_SOURCE,
                               dataset,
                               probas,
                               ERROR_ESTIMATOR,
                               CLFS_SET,
                               MFS_SET,
                               SPEED_FS,
                               DO_OPTMIZATION)