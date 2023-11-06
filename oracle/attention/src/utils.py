from typing import Tuple
import numpy as np


def transform_probas(probas_set: list):

    features = []
    for doc_idx in np.arange(probas_set[0].shape[0]):
        features.append(
            np.vstack([probas_set[clf_idx][doc_idx]
                      for clf_idx in np.arange(len(probas_set))])
        )

    return features


def load_probs_fold(data_source: str,
                    dataset: str,
                    clfs: list,
                    fold: int,
                    n_folds: int,
                    attention_mode=True):

    clfs_probs_train = []
    clfs_probs_test = []
    for clf, proba_type in clfs:
        probs_dir = f"{data_source}/{proba_type}/split_{n_folds}/{dataset}/{n_folds}_folds/{clf}/{fold}/"
        clfs_probs_train.append(np.load(f"{probs_dir}/train.npz")["X_train"])
        clfs_probs_test.append(np.load(f"{probs_dir}/test.npz")["X_test"])

    if attention_mode:
        return transform_probas(clfs_probs_train), transform_probas(clfs_probs_test)
    return np.hstack(clfs_probs_train), np.hstack(clfs_probs_test)


def load_upper_bound(data_source: str,
                     dataset: str,
                     clfs: list,
                     fold: int,
                     n_folds: int):

    uppers = {}
    for clf, proba_type in clfs:
        upper_dir = f"{data_source}/oracle/upper_bound/{proba_type}/{dataset}/{n_folds}_folds/{clf}/{fold}"
        uppers[clf] = {}
        uppers[clf]["train"] = np.load(f"{upper_dir}/train.npz")['y']
        uppers[clf]["test"] = np.load(f"{upper_dir}/test.npz")['y']

    return uppers

def labels_to_att_labels(uppers: dict,
                         clfs: list,
                         train_test: str,
                         optim_combination: bool = True):

    n_docs = uppers[clfs[0][0]][train_test].shape[0]
    attention_labels = []
    for idx in np.arange(n_docs):
        clfs_rows = []
        if optim_combination:
            for clf, _ in clfs:
                clfs_rows.append(
                    uppers[clf][train_test][idx]
                )
            attention_labels.append(np.array(clfs_rows).T)
        else:
            clfs_rows = []
            for clf, _ in clfs:
                clfs_rows.append(
                    np.zeros(len(clfs)) + uppers[clf][train_test][idx]
                )
            attention_labels.append(np.array(clfs_rows).T)

    return attention_labels


def get_attention_labels(data_source: str,
                         dataset: str,
                         clfs: list,
                         fold: int,
                         n_folds: int,
                         opt_comb: bool):
    
    uppers = load_upper_bound(data_source, dataset, clfs, fold, n_folds)
    train = labels_to_att_labels(uppers, clfs, "train", opt_comb)
    test = labels_to_att_labels(uppers, clfs, "test", opt_comb)

    return train, test


def load_labels_fold(data_source: str, dataset: str, fold: int, n_folds: int):

    y_train = np.load(
        f"{data_source}/datasets/labels/split_{n_folds}/{dataset}/{fold}/train.npy")
    y_test = np.load(
        f"{data_source}/datasets/labels/split_{n_folds}/{dataset}/{fold}/test.npy")
    return y_train, y_test


def get_truth_table(rows: int, cols: int):

    truth_table = np.array([[bool(i & (1 << j)) for j in range(cols)]
                           for i in range(rows)], dtype=bool)
    truth_table = np.where(truth_table == 1, 1, 0)
    return truth_table


def expand_probas(probas: np.ndarray, clfs_number: int) -> Tuple[np.ndarray, np.ndarray]:

    nrows = 2 ** clfs_number
    truth_table = get_truth_table(nrows, clfs_number)

    combs = []
    for p in probas:
        clf_probas = p.reshape(clfs_number, -1)
        for comb in truth_table:
            combs.append((clf_probas.T * comb).T.reshape(-1))
    return np.vstack(combs), truth_table


def expand_labels(labels: np.ndarray, clfs_number: int) -> np.ndarray:

    return np.tile(labels, 2 ** clfs_number).reshape(-1, labels.shape[0]).T.reshape(-1)
