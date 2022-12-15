
import os
import json
import numpy as np

from time import time
from collections import Counter
from xgboost import XGBClassifier
from scipy.sparse import csr_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from optuna.integration import OptunaSearchCV
from optuna.distributions import (IntDistribution,
                                  FloatDistribution,
                                  CategoricalDistribution)
from joblib import load, dump

from local_utils import load_stacking_probs, build_clf_beans

# Configs Optuna
import warnings
from optuna.exceptions import ExperimentalWarning
from optuna.logging import set_verbosity, WARNING

set_verbosity(WARNING)
warnings.filterwarnings("ignore", category=ExperimentalWarning)


def agreement_mfs(probas, clf_target, fold, train_test):

    main_preds = probas[clf_target][fold][train_test].argmax(axis=1)
    preds = [probas[clf][fold][train_test].argmax(
        axis=1) for clf in probas if clf != clf_target]
    preds = np.vstack(preds).T

    div = []
    #agree_classes = []
    agree_sizes = []
    for idx in np.arange(main_preds.shape[0]):
        pred_class, agree_size = Counter(preds[idx]).most_common()[0]
        if pred_class == main_preds[idx]:
            div.append(0)
        else:
            div.append(1)
        # agree_classes.append(pred_class)
        agree_sizes.append(agree_size)

    # return div, agree_classes, agree_sizes
    return np.array(div), np.array(agree_sizes)


def confidence_rate(probas, labels):

    conf_hit = []
    conf_freq, hits = build_clf_beans(probas, labels)
    hits_rate = {np.trunc(bean*10)/10: hits[bean] / conf_freq[bean]
                 if bean in hits else 0 for bean in np.arange(0, 1, 0.1)}
    preds = probas.argmax(axis=1)
    for idx, predicted_class in enumerate(preds):
        # Getting the probability of the predicted class
        probability = probas[idx][predicted_class] * 10
        bean = np.trunc(probability) / 10
        bean = 0.9 if bean >= 1 else bean
        conf_hit.append(hits_rate[bean])
    return np.array(conf_hit)


def hits_rate_by_class(probas, labels):

    class_hits_rate = {}
    preds = probas.argmax(axis=1)
    # Vector with hits and misses.
    hits = preds == labels
    # For each label.
    for label in np.unique(labels):
        # Get the docs of the label.
        class_docs = labels == label
        class_hits_rate[label] = np.sum(hits[class_docs]) / np.sum(class_docs)
    return np.array([class_hits_rate[p] for p in preds])


def class_weights(probas, labels):

    cw = {label: np.sum(labels == label) /
          labels.shape[0] for label in np.unique(labels)}
    preds = probas.argmax(axis=1)
    return np.array([cw[p] for p in preds])


def get_clf(clf="xgboost", n_jobs=15):

    if clf == "xgboost":
        # , tree_method='gpu_hist')
        CLF_XGB = XGBClassifier(random_state=42, verbosity=0, n_jobs=n_jobs)
        HYP_XGB = {
            "n_estimators": IntDistribution(low=100, high=500, step=100),
            "learning_rate": FloatDistribution(low=.01, high=.5),
            "max_depth": IntDistribution(low=1, high=14),
            "subsample": FloatDistribution(low=.5, high=1.),
            "booster": CategoricalDistribution(["gbtree", "gblinear", "dart"]),
            "colsample_bytree": FloatDistribution(low=.5, high=1.)
        }
        return CLF_XGB, HYP_XGB
    else:
        HYP_GBM = {}
        return GradientBoostingClassifier(), HYP_GBM


def execute_optimization(
        classifier_name: str,
        file_model: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        opt_cv: int = 4,
        opt_n_iter: int = 30,
        opt_scoring: str = "f1_macro",
        opt_n_jobs: int = 2,
        clf_n_jobs: int = 15,
        seed: int = 42,
        load_model: bool = False
) -> BaseEstimator:

    classifier, hyperparameters = get_clf(classifier_name, n_jobs=clf_n_jobs)
    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("classifier", classifier)
    ])
    hyperparameters = {f"classifier__{k}": v for k,
                       v in hyperparameters.items()}

    optuna_search = OptunaSearchCV(
        pipeline,
        hyperparameters,
        cv=StratifiedKFold(opt_cv, shuffle=True, random_state=seed),
        error_score="raise",
        n_trials=opt_n_iter,
        random_state=seed,
        scoring=opt_scoring,
        n_jobs=opt_n_jobs
    )

    if load_model and os.path.exists(file_model):
        print("\tModel already executed! Loading model...", end="")
        optuna_search = load(file_model)
    else:
        print("\tExecuting model...", end="")
        optuna_search.fit(X_train, y_train)
        dump(optuna_search, file_model)

    return optuna_search


def make_mfs(probas, clf, y_train, y_test, fold, dist_train, dist_test):

    # Building Meta-Features.
    probas_train = probas[clf][fold]["train"]
    probas_test = probas[clf][fold]["test"]
    cw_train = class_weights(probas_train, y_train)
    cw_test = class_weights(probas_test, y_test)
    hrc_train = hits_rate_by_class(probas_train, y_train)
    hrc_test = hits_rate_by_class(probas_test, y_test)
    conf_train = confidence_rate(probas_train, y_train)
    conf_test = confidence_rate(probas_test, y_test)
    div_train, ags_train = agreement_mfs(probas, clf, fold, "train")
    div_test, ags_test = agreement_mfs(probas, clf, fold, "test")
    scaled_ags_train = MinMaxScaler().fit_transform(
        ags_train.reshape(-1, 1)).reshape(-1)
    scaled_ags_test = MinMaxScaler().fit_transform(
        ags_test.reshape(-1, 1)).reshape(-1)

    # Joining Meta-Features.
    X_train = np.vstack([
        cw_train,
        hrc_train,
        conf_train,
        div_train,
        ags_train,
        scaled_ags_train
    ]).T

    X_train = np.hstack([probas_train, dist_train, X_train])

    X_test = np.vstack([
        cw_test,
        hrc_test,
        conf_test,
        div_test,
        ags_test,
        scaled_ags_test
    ]).T
    X_test = np.hstack([probas_test, dist_test, X_test])

    # Making labels (hit or missed)
    preds_train = probas_train.argmax(axis=1)
    upper_train = np.zeros(preds_train.shape[0])
    upper_train[preds_train == y_train] = 1

    preds_test = probas_test.argmax(axis=1)
    upper_test = np.zeros(preds_test.shape[0])
    upper_test[preds_test == y_test] = 1

    return X_train, X_test, upper_train, upper_test


def get_scores(y_test, y_pred):

    pc = precision_score(y_test, y_pred, pos_label=0) * 100
    rc = recall_score(y_test, y_pred, pos_label=0) * 100
    f1 = f1_score(y_test, y_pred, pos_label=0) * 100
    acc = accuracy_score(y_test, y_pred) * 100
    return pc, rc, f1, acc


def get_dict_score(pc, rc, f1, acc):

    d = {
        "f1_macro": f1,
        "precision": pc,
        "recall": rc,
        "accuracy": acc
    }
    return d


def save_scores(output_dir, scores):

    with open(f"{output_dir}/scoring.json", "w") as fd:
        json.dump(scores, fd)


def local_error_estimation(dataset, probas, name_estimator, oracle_dir, CLFS):

    # For each fold.
    for fold in np.arange(10):
        # Loading labels.
        y_train = np.load(
            f"/home/welton/data/datasets/labels/split_10/{dataset}/{fold}/train.npy")
        y_test = np.load(
            f"/home/welton/data/datasets/labels/split_10/{dataset}/{fold}/test.npy")
        # Load fold Meta-Features (Washington).
        dist_train = csr_matrix(np.load(
            f"/home/welton/data/meta_features/features/dist/{fold}/{dataset}/train.npz")["X_train"]).toarray()
        dist_test = csr_matrix(np.load(
            f"/home/welton/data/meta_features/features/dist/{fold}/{dataset}/test.npz")["X_test"]).toarray()
        # For each Stacking base model.
        for target_clf in CLFS:

            # Building Meta-Features.
            X_train, X_test, upper_train, upper_test = make_mfs(
                probas, target_clf, y_train, y_test, fold, dist_train, dist_test)

            output_dir = f"{oracle_dir}/{name_estimator}/{dataset}/{target_clf}/{fold}"
            os.makedirs(output_dir, exist_ok=True)
            file_model = f"{output_dir}/model"

            # Hyperparameter tuning.
            optuna_search = execute_optimization(
                name_estimator,
                file_model,
                X_train[:200],
                upper_train[:200])

            # Prediction
            y_pred = optuna_search.predict(X_test)

            pc = precision_score(upper_test, y_pred, pos_label=0) * 100
            rc = recall_score(upper_test, y_pred, pos_label=0) * 100
            f1 = f1_score(upper_test, y_pred, pos_label=0) * 100
            acc = accuracy_score(upper_test, y_pred) * 100

            print(
                f"\nDATASET: {dataset.upper()} / FOLD - {fold} - Prec: {pc:.2f}, Rec: {rc:.2f}, F1: {f1:.2f}, Acc: {acc:.2f}")


def global_error_estimation(dataset, probas, name_estimator, oracle_dir, CLFS, confidence=75, load_model=True):

    # For each fold.
    for fold in np.arange(10):
        begin = time()
        global_X_train = []
        global_X_test = []
        global_upper_train = []
        global_upper_test = []
        global_y_train = []
        global_y_test = []
        # Loading labels.
        y_train = np.load(
            f"/home/welton/data/datasets/labels/split_10/{dataset}/{fold}/train.npy")
        y_test = np.load(
            f"/home/welton/data/datasets/labels/split_10/{dataset}/{fold}/test.npy")
        # Load fold Meta-Features (Washington).
        dist_train = csr_matrix(np.load(
            f"/home/welton/data/meta_features/features/dist/{fold}/{dataset}/train.npz")["X_train"]).toarray()
        dist_test = csr_matrix(np.load(
            f"/home/welton/data/meta_features/features/dist/{fold}/{dataset}/test.npz")["X_test"]).toarray()
        # Joining all CLFs meta-features.
        for target_clf in CLFS:

            # Building Meta-Features.
            X_train, X_test, upper_train, upper_test = make_mfs(
                probas, target_clf, y_train, y_test, fold, dist_train, dist_test)

            global_X_train.append(X_train)
            global_X_test.append(X_test)

            global_upper_train.append(upper_train)
            global_upper_test.append(upper_test)

            global_y_train.append(y_train)
            global_y_test.append(y_test)

        global_X_train = np.vstack(global_X_train)
        global_upper_train = np.hstack(global_upper_train)
        upper_test = np.hstack(global_upper_test)
        X_test = np.vstack(global_X_test)

        output_dir = f"{oracle_dir}/global_{name_estimator}/error_estimator/{dataset}/{fold}"
        os.makedirs(output_dir, exist_ok=True)
        model_path = f"{output_dir}/model"
        
        """
        # Hyperparameter tuning on GLOBAL Meta-Features (concatenating all CLFs MFs).
        optuna_search = execute_optimization(
            name_estimator,
            model_path,
            global_X_train,
            global_upper_train,
            load_model=True)

        # Gloabl Prediction
        y_pred = optuna_search.predict(X_test)
        """
        if load_model and os.path.exists(model_path):
            error_estimator = load(model_path)
        else:
            error_estimator = XGBClassifier(
                n_estimators=300,
                learning_rate=0.11,
                max_depth=11,
                booster="gbtree",
                colsample_bytree=0.650026359170959,
                random_state=42,
                verbosity=0,
                n_jobs=15)  # , tree_method='gpu_hist')
            error_estimator.fit(global_X_train, global_upper_train)
            dump(error_estimator, model_path)
        
        y_pred = error_estimator.predict(X_test)
        pc, rc, f1, acc = get_scores(upper_test, y_pred)
        dict_scores = get_dict_score(pc, rc, f1, acc)
        save_scores(output_dir, dict_scores)
        print(
            f"\nDATASET: {dataset.upper()} / GLOBAL ESTIMATOR / FOLD - {fold} - Prec: {pc:.2f}, Rec: {rc:.2f}, F1: {f1:.2f}, Acc: {acc:.2f}")

        # Local Prediction.
        for X_test, upper_test, alg in zip(global_X_test, global_upper_test, CLFS):

            output_dir = f"{oracle_dir}/global_{name_estimator}/clfs/{dataset}/{alg}/{fold}"
            os.makedirs(output_dir, exist_ok=True)

            y_pred = error_estimator.predict(np.vstack(X_test))
            pc, rc, f1, acc = get_scores(upper_test, y_pred)
            print(
                f"DATASET: {dataset.upper()} / CLF: {alg} / FOLD - {fold} - Prec: {pc:.2f}, Rec: {rc:.2f}, F1: {f1:.2f}, Acc: {acc:.2f}")

            dict_scores = get_dict_score(pc, rc, f1, acc)
            save_scores(output_dir, dict_scores)

            if f1 < confidence:
                y_pred = np.zeros(y_pred.shape[0]) + 1

            #y_probas = error_estimator.predict_proba(np.vstack(X_test))
            #y_pred = np.array([ 1 - y_probas[idx][0] if y_probas[idx][0] > y_probas[idx][1] else 1 for idx in np.arange(y_probas.shape[0]) ])


            np.savez(f"{output_dir}/test", y=y_pred)
            y_true = np.zeros(y_train.shape[0]) + 1
            np.savez(f"{output_dir}/train", y=y_true)
        end = time()
        print(f"Seconds: {end - begin}s")


if __name__ == "__main__":

    DATASETS = ["webkb"]
    CLFS = ["kpr", "ktr", "lpr", "ltr", "sfr", "stmk", "xfr", "xpr", "xtr", "kfr",
            "ktmk", "lfr", "ltmk", "spr", "str", "xlnet_softmax", "xtmk", "rep_bert"]
    ORACLE_DIR = "/home/welton/data/oracle"
    ESTIMATOR_NAME = "xgboost"

    dataset = DATASETS[0]
    probas = load_stacking_probs(dataset, CLFS, "train_test")
    #local_error_estimation(dataset, probas, "xgboost", ORACLE_DIR, CLFS)
    global_error_estimation(dataset, probas, "xgboost", ORACLE_DIR, CLFS, 75, True)
