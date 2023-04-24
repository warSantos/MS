import os
import jsonlines
import pandas as pd
import numpy as np
from typing import Tuple
from collections import Counter
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

from sys import path
path.append('../utils')
from local_utils import build_clf_beans


def replace_nan_inf(a, value=0):

    a[np.isinf(a) | np.isnan(a)] = value
    return a


def load_bert_reps(dataset: str, fold: int) -> Tuple[np.ndarray, np.ndarray]:

    idxs = pd.read_pickle(
        f"/home/welton/data/datasets/data/{dataset}/splits/split_10_with_val.pkl")
    lines = jsonlines.open(
        f"/home/welton/data/kaggle/{dataset}/{dataset}_bert{fold}.json")
    reps = [[l["id"], l["bert"]] for l in lines]
    df = pd.DataFrame(reps, columns=["id", "reps"])

    X_train = np.vstack(
        df.query(f"id == {idxs['train_idxs'][fold]}").reps.values.tolist())
    X_val = np.vstack(
        df.query(f"id == {idxs['val_idxs'][fold]}").reps.values.tolist())
    X_test = np.vstack(
        df.query(f"id == {idxs['test_idxs'][fold]}").reps.values.tolist())

    sort = np.hstack(
        [idxs['train_idxs'][fold], idxs['val_idxs'][fold]]).argsort()
    X_train = np.vstack([X_train, X_val])[sort]

    return X_train, X_test


def get_neigbors_entropy(X_train: np.ndarray,
                         X_test: np.ndarray,
                         y_train: np.ndarray,
                         train_test: str):

    dists = pairwise_distances(X_test,
                                X_train,
                                metric="euclidean",
                                n_jobs=10)

    if train_test == "test":
        start, end = 0, 30
    else:
        start, end = 1, 31
    features = []
    for doc_dists in dists:
        sort = doc_dists.argsort()
        class_neighbors = [y_train[sort[idx]] for idx in np.arange(start, end)]
        features.append([
            entropy(class_neighbors),
            len(set(class_neighbors)),
        ])

    features = np.array(features)
    replace_nan_inf(features)
    return features


def neigborhood_mfs(dataset: str, y_train: np.ndarray, fold: int):
    
    features_dir = f"/home/welton/data/meta_features/error_estimation/dists/{dataset}/{fold}/neigbors"
    os.makedirs(features_dir, exist_ok=True)

    train_dists_path = f"{features_dir}/train.npz"
    test_dists_path = f"{features_dir}/test.npz"
    
    if os.path.exists(train_dists_path) and os.path.exists(test_dists_path):

        print(f"\tLoading pre-computed neigbors features: {train_dists_path}...")
        train_features = np.load(train_dists_path)["dists"]

        print(f"\tLoading pre-computed neigbors features: {test_dists_path}...")
        test_features = np.load(test_dists_path)["dists"]
    else:

        X_train, X_test = load_bert_reps(dataset, fold)

        train_features = get_neigbors_entropy(X_train, X_train, y_train, "train")
        np.savez(train_dists_path, dists=train_features)
    
        test_features = get_neigbors_entropy(X_train, X_test, y_train, "test")
        np.savez(test_dists_path, dists=test_features)
    
    return train_features, test_features


def centroides_ratio(X_train: np.ndarray,
                     X_test: np.ndarray,
                     y_train: np.ndarray) -> np.ndarray:

    
    centroides = np.vstack([
        np.mean(X_train[y_train == label], axis=0)
        for label in set(y_train)
    ])
    dists = pairwise_distances(X_test,
                                centroides,
                                metric="euclidean",
                                n_jobs=10)
    ratios = []
    for doc_dists in dists:

        sort = doc_dists.argsort()
        """
        doc_ratios = [ doc_dists[sort[0]] / doc_dists[sort[idx]]
            for idx in np.arange(1, sort.shape[0]) ]
        """
        ratios.append(doc_dists[sort[0]] / doc_dists[sort[1]])
    feats = np.array(ratios).reshape(-1,1)
    replace_nan_inf(feats)
    return feats

def get_centroides_ratios_mfs(dataset: str, y_train: np.ndarray, fold: int):

    features_dir = f"/home/welton/data/meta_features/error_estimation/dists/{dataset}/{fold}/cent"
    os.makedirs(features_dir, exist_ok=True)

    # Loading features if it already exist. If not, build them.
    train_dists_path = f"{features_dir}/train.npz"
    test_dists_path = f"{features_dir}/test.npz"
    
    if os.path.exists(train_dists_path) and os.path.exists(test_dists_path):
        print(f"\tLoading pre-computed centroids ratios features: {train_dists_path}...")
        train_features = np.load(train_dists_path)["dists"]
        
        print(f"\tLoading pre-computed centroids ratios features: {test_dists_path}...")
        test_features = np.load(test_dists_path)["dists"]

    else:
        X_train, X_test = load_bert_reps(dataset, fold)
        
        train_features = centroides_ratio(X_train, X_train, y_train)
        np.savez(train_dists_path, dists=train_features)
    
        test_features = centroides_ratio(X_train, X_test, y_train)
        np.savez(test_dists_path, dists=test_features)

    return train_features, test_features


def agreement_mfs(probas, clf_target, fold, train_test):

    main_preds = probas[clf_target][fold][train_test].argmax(axis=1)
    preds = [probas[clf][fold][train_test].argmax(
        axis=1) for clf in probas if clf != clf_target]
    preds = np.vstack(preds).T

    div = []
    agree_sizes = []
    num_classes = []
    pred_entropy = []
    # For each document.
    for idx in np.arange(main_preds.shape[0]):
        counts = Counter(preds[idx])
        pred_class, agree_size = counts.most_common()[0]
        if pred_class == main_preds[idx]:
            div.append(0)
        else:
            div.append(1)
        agree_sizes.append(agree_size)
        total = len(counts)
        num_classes.append(total)
        pred_entropy.append(entropy([counts[k]/total for k in counts]))

    return np.array(div), np.array(agree_sizes), np.array(num_classes), np.array(pred_entropy)


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


def make_mfs(mf_input_dir: str,
             data_source: str,
             dataset: str,
             clf: str,
             probas: dict,
             y_train: np.ndarray,
             y_test: np.ndarray,
             fold: int,
             meta_features_set: list):

    X_train_path = f"{mf_input_dir}/x_train.npz"
    X_test_path = f"{mf_input_dir}/x_test.npz"
    upper_train_path = f"{mf_input_dir}/upper_train.npz"
    upper_test_path = f"{mf_input_dir}/upper_test.npz"

    paths = [X_test_path, X_train_path, upper_test_path, upper_train_path]

    # If all MFs are already built, just load it.
    if np.all([os.path.exists(p) for p in (paths)]):

        X_train = np.load(X_train_path)["X_train"]
        X_test = np.load(X_test_path)["X_test"]

        upper_train = np.load(upper_train_path)["y"]
        upper_test = np.load(upper_test_path)["y"]

        return X_train, X_test, upper_train, upper_test

    pack_meta_features = {
        "train": [],
        "test": []
    }

    probas_train = probas[clf][fold]["train"]
    probas_test = probas[clf][fold]["test"]

    if "probas-based" in meta_features_set:

        print("loading probas-based...")

        # Building Meta-Features.
        cw_train = class_weights(probas_train, y_train)
        cw_test = class_weights(probas_test, y_test)
        hrc_train = hits_rate_by_class(probas_train, y_train)
        hrc_test = hits_rate_by_class(probas_test, y_test)
        conf_train = confidence_rate(probas_train, y_train)
        conf_test = confidence_rate(probas_test, y_test)
        div_train, ags_train, num_classes_train, entropy_train = agreement_mfs(
            probas, clf, fold, "train")
        div_test, ags_test, num_classes_test, entropy_test = agreement_mfs(
            probas, clf, fold, "test")

        scaled_ags_train = MinMaxScaler().fit_transform(
            ags_train.reshape(-1, 1)).reshape(-1)
        scaled_ags_test = MinMaxScaler().fit_transform(
            ags_test.reshape(-1, 1)).reshape(-1)
        scaled_num_classes_train = MinMaxScaler().fit_transform(
            num_classes_train.reshape(-1, 1)).reshape(-1)
        scaled_num_classes_test = MinMaxScaler().fit_transform(
            num_classes_test.reshape(-1, 1)).reshape(-1)

        # Joining Meta-Features.
        probas_based_train = np.vstack([
            cw_train,
            hrc_train,
            conf_train,
            div_train,
            scaled_ags_train,
            scaled_num_classes_train,
            entropy_train
        ]).T

        probas_based_test = np.vstack([
            cw_test,
            hrc_test,
            conf_test,
            div_test,
            scaled_ags_test,
            scaled_num_classes_test,
            entropy_test
        ]).T

        pack_meta_features["train"].append(probas_based_train)
        pack_meta_features["test"].append(probas_based_test)

    if "probas" in meta_features_set:

        print("loading probas...")

        pack_meta_features["train"].append(probas_train)
        pack_meta_features["test"].append(probas_test)

    if "dist" in meta_features_set:

        print("loading dist...")

        # Load fold Meta-Features (Washington).
        dist_train = csr_matrix(np.load(
            f"{data_source}/meta_features/features/bert_dists/{fold}/{dataset}/train.npz")["X_train"]).toarray()
        dist_test = csr_matrix(np.load(
            f"{data_source}/meta_features/features/bert_dists/{fold}/{dataset}/test.npz")["X_test"]).toarray()

        replace_nan_inf(dist_train)
        replace_nan_inf(dist_test)
        pack_meta_features["train"].append(dist_train)
        pack_meta_features["test"].append(dist_test)

    if "neigborhood" in meta_features_set:

        print("loading neighborhood...")

        n_train, n_test = neigborhood_mfs(dataset, y_train, fold)
        pack_meta_features["train"].append(n_train)
        pack_meta_features["test"].append(n_test)

    if "centroids-ratios" in meta_features_set:

        print("loading centroids...")

        c_train, c_test = get_centroides_ratios_mfs(dataset, y_train, fold)
        pack_meta_features["train"].append(c_train)
        pack_meta_features["test"].append(c_test)

    X_train = np.hstack(pack_meta_features["train"])
    X_test = np.hstack(pack_meta_features["test"])

    # Making labels (hit or missed)
    preds_train = probas_train.argmax(axis=1)
    upper_train = np.zeros(preds_train.shape[0])
    upper_train[preds_train == y_train] = 1

    preds_test = probas_test.argmax(axis=1)
    upper_test = np.zeros(preds_test.shape[0])
    upper_test[preds_test == y_test] = 1

    os.makedirs(mf_input_dir, exist_ok=True)
    np.savez(X_train_path, X_train=X_train)
    np.savez(X_test_path, X_test=X_test)
    np.savez(upper_test_path, y=upper_test)
    np.savez(upper_train_path, y=upper_train)

    return X_train, X_test, upper_train, upper_test
