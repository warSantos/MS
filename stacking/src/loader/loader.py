from typing import Tuple, List
import numpy as np

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


def read_meta_data(
        data_dir: str,
        dataset: str,
        clf_set: List[list],
        n_folds: int,
        fold_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    Xs_train, Xs_test = [], []

    for clf, source in clf_set:
        
        proba_dir = f"{data_dir}/{source}/split_{n_folds}/{dataset}/{n_folds}_folds/{clf}/{fold_id}"

        file_train_meta = f"{proba_dir}/train.npz"
        file_test_meta = f"{proba_dir}/test.npz"

        X_train_meta, _ = load_x_y(file_train_meta, 'train')
        X_test_meta, _ = load_x_y(file_test_meta, 'test')

        Xs_train.append(X_train_meta)
        Xs_test.append(X_test_meta)

    X_train_meta = np.hstack(Xs_train)
    X_test_meta = np.hstack(Xs_test)

    return X_train_meta, X_test_meta