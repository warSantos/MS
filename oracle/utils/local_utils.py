import numpy as np
from typing import Tuple, List


def build_clf_beans(clf_probas, label):
    predictions = clf_probas.argmax(axis=1)
    confidence_freq = {}
    hits = {}
    # For each prediction
    for idx, predicted_class in enumerate(predictions):

        # Getting the probability of the predicted class
        probability = clf_probas[idx][predicted_class] * 10
        bean = np.trunc(probability) / 10
        bean = 0.9 if bean >= 1 else bean
        # Adding the bean in confidence if is not there yet.
        if bean not in confidence_freq:
            confidence_freq[bean] = 0
        confidence_freq[bean] += 1
        # Veryfing if the predicted class was right.
        if predicted_class == label[idx]:
            if bean not in hits:
                hits[bean] = 0
            hits[bean] += 1
    return confidence_freq, hits


def get_miss_predictor(confidence_freq, hits, threshold=0.3):

    predictor = {}
    # For each confidence interval.
    for bean in hits:
        # Get the hit rate.
        hits_rate = hits[bean] / confidence_freq[bean]

        if hits_rate < threshold:
            predictor[bean] = True
    return predictor


def predict(X, estimator):

    estimates = []
    predictions = X.argmax(axis=1)
    # For each prediction.
    for idx, predicted_class in enumerate(predictions):
        probability = X[idx][predicted_class] * 10
        bean = np.trunc(probability) / 10
        bean = 0.9 if bean >= 1 else bean
        # If this confidence has a miss rate greater than THRESHOLD (wether it is in the dictionary or not)
        if bean in estimator:
            estimates.append(0)
        else:
            estimates.append(1)
    return np.array(estimates)


def load_x_y(
        file: str,
        test_train: str
) -> Tuple[np.ndarray, np.ndarray]:

    loaded = np.load(file, allow_pickle=True)
    X = loaded[f"X_{test_train}"]

    if f"y_{test_train}" not in loaded:
        return X, None

    y = loaded[f"y_{test_train}"]

    if X.size == 1:
        X = X.item()

    return X, y


def read_train_test_meta(
        dir_meta_input: str,
        dataset: str,
        n_folds: int,
        fold_id: int,
        algorithms: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    Xs_train, Xs_test = [], []

    for alg in algorithms:
        file_train_meta = f"{dir_meta_input}/{dataset}/{n_folds}_folds/{alg}/{fold_id}/train.npz"
        file_test_meta = f"{dir_meta_input}/{dataset}/{n_folds}_folds/{alg}/{fold_id}/test.npz"

        X_train_meta, _ = load_x_y(file_train_meta, 'train')
        X_test_meta, _ = load_x_y(file_test_meta, 'test')

        Xs_train.append(X_train_meta)
        Xs_test.append(X_test_meta)

    X_train_meta = np.hstack(Xs_train)
    X_test_meta = np.hstack(Xs_test)

    return X_train_meta, X_test_meta


def load_clfs_probas(data_source: str,
                     dataset: str,
                     CLFS_SET: list,
                     n_folds: int,
                     train_test: str = "train"):

    probs = {}
    for clf, proba_type in CLFS_SET:
        probs[clf] = {}
        for fold in np.arange(n_folds):
            probs[clf][fold] = {}
            prob_dir = f"{data_source}/{proba_type}/split_{n_folds}/{dataset}/{n_folds}_folds/{clf}/{fold}/"
            if train_test == "train":
                file_path = f"{prob_dir}/train.npz"
                probs[clf][fold]["train"] = np.load(file_path)[f"X_tain"]
            elif train_test == "test":
                file_path = f"{prob_dir}/test.npz"
                probs[clf][fold]["test"] = np.load(file_path)[f"X_test"]
            else:
                file_path = f"{prob_dir}/train.npz"
                probs[clf][fold]["train"] = np.load(file_path)[f"X_train"]
                file_path = f"{prob_dir}/test.npz"
                probs[clf][fold]["test"] = np.load(file_path)[f"X_test"]
    return probs


def load_labels(dataset: str, fold: int, n_folds: int):

    file_path = f"/home/welton/data/clfs_output/split_{n_folds}/{dataset}/{n_folds}_folds/lfr/{fold}/train.npz"
    y_train = np.load(file_path)["y_train"]
    file_path = f"/home/welton/data/clfs_output/split_{n_folds}/{dataset}/{n_folds}_folds/lfr/{fold}/test.npz"
    y_test = np.load(file_path)["y_test"]
    return y_train, y_test
