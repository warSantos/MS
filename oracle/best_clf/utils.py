
import json
import pickle
import numpy as np
from typing import Any
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

def save_pickle(obj: Any, obj_path: str):

    with open(obj_path, "wb") as fd:
        pickle.dump(obj, fd)

def load_pickle(obj_path: str):
    
    with open(obj_path, "rb") as fd:
        return pickle.load(fd)


def report_scoring(y_test: np.ndarray, y_pred: np.ndarray, output_dir: str):

    scoring = {}
    scoring["macro"] = f1_score(y_test, y_pred, average="macro")
    scoring["micro"] = f1_score(y_test, y_pred, average="micro")
    scoring["accuracy"] = accuracy_score(y_test, y_pred)

    print(f"\n\tMacro: {scoring['macro'] * 100}")
    print(f"\tMicro: {scoring['micro'] * 100}\n")

    with open(f"{output_dir}/scoring.json", "w") as fd:
        json.dump(scoring, fd)

    return scoring["macro"]

def load_labels(source_dir: str,
                dataset: str,
                fold: int,
                n_folds: int) -> tuple:
    
    labels_dir = f"{source_dir}/datasets/labels/split_{n_folds}/{dataset}/{fold}"
    return np.load(f"{labels_dir}/train.npy"), np.load(f"{labels_dir}/test.npy")

def load_full_labels(source_dir: str, datasets: list) -> dict:

    d_labels = {}
    for dataset, n_folds in datasets:
        d_labels[dataset] = {
            "train": [],
            "test": []
        }
        for fold in np.arange(n_folds):

            labels_dir = f"{source_dir}/datasets/labels/split_{n_folds}/{dataset}/{fold}"
            d_labels[dataset]["train"].append(
                np.load(f"{labels_dir}/train.npy"))
            d_labels[dataset]["test"].append(np.load(f"{labels_dir}/test.npy"))

    return d_labels


def load_probas(source_dir: str,
               dataset: str,
               clf_set: list,
               fold: int,
               n_folds: int):
    
    probas = {}
    for clf, proba_type in clf_set:
        probas[clf] = {}
        probas_dir = f"{source_dir}/{proba_type}/split_{n_folds}/{dataset}/{n_folds}_folds/{clf}/{fold}"
        probas[clf]["train"] = np.load(f"{probas_dir}/train.npz")["X_train"]
        probas[clf]["test"] = np.load(f"{probas_dir}/test.npz")["X_test"]
    
    return probas

def load_full_probas(source_dir: str, datasets: list, clf_set: list) -> dict:
    
    d_probas = {}
    # For each dataset.
    for dataset, n_folds in datasets:
        d_probas[dataset] = {}
        # For each CLF.
        for clf, proba_type in clf_set:
            # If clf not in d_probas yet.
            if clf not in d_probas:
                d_probas[dataset][clf] = {}
            # For each fold.
            for fold in np.arange(n_folds):
                probas_dir = f"{source_dir}/{proba_type}/split_{n_folds}/{dataset}/{n_folds}_folds/{clf}/{fold}"

                if f"train" not in d_probas[dataset][clf]:
                    d_probas[dataset][clf]["train"] = []
                d_probas[dataset][clf]["train"].append(
                    np.load(f"{probas_dir}/train.npz")["X_train"])

                if f"test" not in d_probas[dataset][clf]:
                    d_probas[dataset][clf]["test"] = []
                d_probas[dataset][clf]["test"].append(
                    np.load(f"{probas_dir}/test.npz")["X_test"])

            d_probas[dataset][clf]
    return d_probas


def get_single_label(dataset: str,
                     clf_set: list,
                     probas: dict,
                     labels: dict,
                     fold: int,
                     train_test: str,
                     label_type: str) -> tuple:

    n_docs = probas[dataset]["bert"][train_test][fold].shape[0]
    new_labels = []
    missed_by_all = []
    # For each document.
    for doc_idx in np.arange(n_docs):
        max_conf = -1
        best_clf = -1
        min_conf = 2
        worst_clf = -1
        # For each classifier.
        for clf_idx, (clf, _) in enumerate(clf_set):
            prob = probas[dataset][clf][train_test][fold][doc_idx]
            pred = prob.argmax()
            conf = prob[pred]
            y = labels[dataset][train_test][fold][doc_idx]
            # If the classifier predicted correctly and his confidence is higher.
            if y == pred and max_conf < conf:
                max_conf = conf
                best_clf = clf_idx
            # If the classifier predicted wrong and his confidence is low (he missed with caution).
            if y != pred and min_conf > conf:
                min_conf = conf
                worst_clf = clf_idx

        # If none of the classifiers hit, set it as bert.
        if max_conf == -1:
            if label_type == "less_conf":
                new_labels.append(worst_clf)
            else:
                new_labels.append(np.random.randint(len(clf_set)))
            missed_by_all.append(1)
        else:
            new_labels.append(best_clf)
            missed_by_all.append(0)
    return np.array(new_labels), np.array(missed_by_all)


def get_multi_labels(dataset: str,
                     clf_set: list,
                     probas: dict,
                     labels: dict,
                     fold: int,
                     train_test: str) -> tuple:

    n_docs = probas[dataset][clf_set[0][0]][train_test][fold].shape[0]
    new_labels = []
    missed_by_all = []
    # For each document.
    for doc_idx in np.arange(n_docs):
        # For each classifier.
        doc_labels = []
        for clf_idx, (clf, _) in enumerate(clf_set):
            pred = probas[dataset][clf][train_test][fold][doc_idx].argmax()
            y = labels[dataset][train_test][fold][doc_idx]
            # If the classifier predicted correctly.
            if y == pred:
                doc_labels.append(clf_idx)

        # If none of the classifiers hit, chose a random classifier.
        if not doc_labels:
            doc_labels.append(np.random.randint(len(clf_set)))
            missed_by_all.append(1)
        new_labels.append(doc_labels)

    # Formating labels for multiclass problem.
    mb = MultiLabelBinarizer()
    return mb.fit_transform(new_labels), np.array(missed_by_all)


def get_new_labels(meta_layer: str,
                   dataset: str,
                   clf_set: list,
                   probas: dict,
                   labels: dict,
                   fold: int,
                   train_test: str,
                   label_type: str):

    if meta_layer == "multi_output":

        return get_multi_labels(dataset,
                                clf_set,
                                probas,
                                labels,
                                fold,
                                train_test)
    else:
        return get_single_label(dataset,
                                clf_set,
                                probas,
                                labels,
                                fold,
                                train_test,
                                label_type)


def get_clf(clf: str):

    if clf == "logistic_regression":
        return LogisticRegression(max_iter=1000, n_jobs=30, random_state=42)
    elif clf == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=30, random_state=42)
    else:
        raise ("Classifier option not valid.")

def load_meta_features(data_source: str,
                       dataset: str,
                       n_classes: int,
                       clf_set: list,
                       clf_sufix: str,
                       fold: int,
                       features_set: list,
                       mf_set: str,
                       n_folds: int):
    
    train, test = [], []
    for clf, _ in clf_set:
        loader = np.load(f"{data_source}/meta_features/error_estimation/probas-based/{dataset}/{fold}/{clf_sufix}/{clf}/feats.npz")
        train.append(loader["train"])
        test.append(loader["test"])
    
    return np.hstack(train), np.hstack(test)

"""
def load_meta_features(data_source: str,
                       dataset: str,
                       n_classes: int,
                       clf_set: list,
                       clf_sufix: str,
                       fold: int,
                       features_set: list,
                       mf_set: str,
                       n_folds: int):

    X_train, X_test = [], []
    flag = True

    for clf, _ in clf_set:

        feats = {"centroids-ratios", "neigborhood", "probas", "probas-based"}
        if len(feats.intersection(set(features_set))) > 0:

            rep_source = f"{data_source}/meta_features/error_estimation/{dataset}/{n_folds}_folds/{clf_sufix}/{mf_set}/{clf}/{fold}"
            train = np.load(f"{rep_source}/x_train.npz")["X_train"]
            test = np.load(f"{rep_source}/x_test.npz")["X_test"]

            if "probas-based" in features_set:

                # Handcrafted.
                X_train.append(train[:, :7])
                X_test.append(test[:, :7])

            if "probas" in features_set:

                # Probabilities.
                X_train.append(train[:, 7:7+n_classes])
                X_test.append(test[:, 7:7+n_classes])

            if flag and ("centroids-ratios" in features_set or "neigborhood" in features_set):
                X_train.append(train[:, -3:])
                X_test.append(test[:, -3:])
                flag = False

    if "bert" in features_set:

        rep_source = f"{data_source}/representations/{dataset}/{n_folds}_folds/bert_250/{fold}"
        bert_train = np.load(f"{rep_source}/train.npz")["X_train"]
        bert_test = np.load(f"{rep_source}/test.npz")["X_test"]

        X_train.append(bert_train)
        X_test.append(bert_test)

    if "dist-20" in features_set:

        rep_source = f"{data_source}/meta_features/features/bert_dists_20/{fold}/{dataset}"
        mf_train = np.load(f"{rep_source}/train.npz")["X_train"]
        mf_test = np.load(f"{rep_source}/test.npz")["X_test"]

        X_train.append(mf_train)
        X_test.append(mf_test)

    if not X_train:
        raise ("No meta-feature selected.")

    X_train = np.hstack(X_train)
    X_test = np.hstack(X_test)

    return X_train, X_test
"""


def predict(dataset: str,
            clf_set: str,
            probas: dict,
            n_docs: int,
            fold: int,
            best_clfs_est: list):

    y_pred = []
    # Getting the documents predictions.
    for doc_idx in np.arange(n_docs):
        best_clf_idx = best_clfs_est[doc_idx]
        best_clf = clf_set[best_clf_idx][0]
        pred = probas[dataset][best_clf]["test"][fold][doc_idx].argmax()
        y_pred.append(pred)

    return y_pred


def get_multi_output_clf():

    model = get_clf("logistic_regression")
    model.set_params(**{"n_jobs": 5})
    return MultiOutputClassifier(model, n_jobs=12)


def multi_output_predict(dataset: str,
                         clf_set: list,
                         probas: dict,
                         n_docs: int,
                         fold: int,
                         best_clfs_est: list):
    y_pred = []
    best_clf_est = []
    n_clfs = len(best_clfs_est)
    # For each document.
    for doc_idx in np.arange(n_docs):
        max_conf = -1
        best_clf_idx = -1
        # For each class (clf).
        for clf_idx in np.arange(n_clfs):
            prob = best_clfs_est[clf_idx][doc_idx][1]
            # If the model estimated that the clf_idx hit the prediction.
            if prob > 0.5 and prob > max_conf:
                best_clf_idx = clf_idx
                max_conf = prob
        best_clf_est.append(best_clf_idx)
        best_clf = clf_set[best_clf_idx][0]
        pred = probas[dataset][best_clf]["test"][fold][doc_idx].argmax()
        y_pred.append(pred)

    return y_pred, best_clf_est
