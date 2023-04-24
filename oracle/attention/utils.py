import numpy as np

def transform_probas(probas_set: list):

    features = []
    for doc_idx in np.arange(probas_set[0].shape[0]):
        features.append(
            np.vstack([ probas_set[clf_idx][doc_idx] for clf_idx in np.arange(len(probas_set)) ])
        )

    return features

def load_probs_fold(dataset: str, clfs: list, fold: int):

    clfs_probs_train = []
    clfs_probs_test = []
    for clf, proba_type in clfs:
        probs_dir = f"/home/welton/data/{proba_type}/split_10/{dataset}/10_folds/{clf}/{fold}/"
        clfs_probs_train.append(np.load(f"{probs_dir}/train.npz")["X_train"])
        clfs_probs_test.append(np.load(f"{probs_dir}/test.npz")["X_test"])
    
    return transform_probas(clfs_probs_train), transform_probas(clfs_probs_test)

def load_upper_bound(dataset: str, clfs: list, fold: int):

    uppers = {}
    for clf, proba_type in clfs:
        upper_dir = f"/home/welton/data/oracle/upper_bound/{proba_type}/{dataset}/10_folds/{clf}/{fold}"
        uppers[clf] = {}
        uppers[clf]["train"] = np.load(f"{upper_dir}/train.npz")['y']
        uppers[clf]["test"] = np.load(f"{upper_dir}/test.npz")['y']
    
    return uppers

def get_attention_labels(clfs: list, uppers: dict, train_test: str):

    n_docs = uppers[clfs[0][0]][train_test].shape[0]
    attention_labels = []
    for idx in np.arange(n_docs):
        clfs_rows = []
        for clf, _ in clfs:
            clfs_rows.append(
                np.zeros(len(clfs)) + uppers[clf][train_test][idx]
            )
        attention_labels.append(np.array(clfs_rows).T)
    
    return attention_labels


def load_labels_fold(dataset: str, fold: int):

    y_train = np.load(f"/home/welton/data/datasets/labels/split_10/{dataset}/{fold}/train.npy")
    y_test = np.load(f"/home/welton/data/datasets/labels/split_10/{dataset}/{fold}/test.npy")
    return y_train, y_test