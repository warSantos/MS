import os
import numpy as np

from src.utils.utils import (stratfied_cv,
                             replace_nan_inf,
                             save_mfs,
                             load_bert_reps,
                             load_y)

from src.dist_mfs.mf.centbased import MFCent
from src.dist_mfs.mf.knnbased import MFKnn
from src.dist_mfs.mf.bestk import kvalues

def fix_label(array: np.ndarray) -> np.ndarray:

    if np.min(array) > 0:
        return array - 1
    return array

def load_reps(input_dir: str):

    train_loader = np.load(f"{input_dir}/train.npz", allow_pickle=True)
    test_loader = np.load(f"{input_dir}/test.npz", allow_pickle=True)
    try:
        X_train, X_test = train_loader["X_train"].tolist(), test_loader["X_test"].tolist()
    except:
        X_train, X_test = train_loader["X_train"], test_loader["X_test"]

    y_train, y_test = train_loader["y_train"], test_loader["y_test"]

    return X_train.toarray(), X_test.toarray(), fix_label(y_train), fix_label(y_test)


class DistMFs:

    def transform(self,
                  dataset: str,
                  X_train: np.ndarray,
                  X_test: np.ndarray,
                  y_train: np.ndarray) -> np.ndarray:

        mfs = []
        cent_cosine = MFCent("cosine")
        cent_cosine.fit(X_train, y_train)
        mfs.append(cent_cosine.transform(X_test))

        cent_l2 = MFCent("l2")
        cent_l2.fit(X_train, y_train)
        mfs.append(cent_l2.transform(X_test))

        knn_cosine = MFKnn("cosine", kvalues[dataset])
        knn_cosine.fit(X_train, y_train)
        mfs.append(knn_cosine.transform(X_test))

        knn_l2 = MFKnn("l2", kvalues[dataset])
        knn_l2.fit(X_train, y_train)
        mfs.append(knn_l2.transform(X_test))

        return np.hstack(mfs)

    def build_train_mfs(self,
                        base_dir: str,
                        dataset: str,
                        n_splits: int):

        # List of train mfs.
        train_mfs = []

        for fold in np.arange(n_splits):

            print(f"\t[{dataset.upper()} / FOLD: - {fold+1}/{n_splits} ]")

            input_dir = f"{base_dir}/sub_folds/{fold}"
            X_train, X_test, y_train, y_test = load_reps(input_dir)

            # Applying oversampling when it is needed.
            for c in set(y_test) - set(y_train):

                sintetic = np.zeros(X_train.shape[1])
                X_train = np.vstack([X_train, sintetic])
                y_train = np.hstack([y_train, [c]])

            new_mfs = self.transform(dataset,
                                     X_train,
                                     X_test,
                                     y_train)

            train_mfs.append(new_mfs)
        return np.vstack(train_mfs)

    def build_features(self,
                       reps_dir: str,
                       input_rep: str,
                       output_rep: str,
                       dataset: str,
                       folds: list,
                       n_folds: int,
                       n_sub_folds: int) -> None:

        for fold in folds:
            print(f"[{dataset.upper()} / FOLD: - {fold+1}/{n_folds} ]")
            
            # Loading represantations.
            input_dir = f"{reps_dir}/{dataset}/{n_folds}_folds/{input_rep}/{fold}"
            X_train, X_test, y_train, y_test = load_reps(input_dir)

            train_mfs = self.build_train_mfs(input_dir,
                                             dataset,
                                             n_sub_folds)
            
            # Generating test meta-features.
            test_mfs = self.transform(dataset,
                                      X_train,
                                      X_test,
                                      y_train)
            
            output_dir = f"{reps_dir}/{dataset}/{n_folds}_folds/{output_rep}/{fold}"
            save_mfs(output_dir,
                dataset,
                "bert_dists",
                fold,
                train_mfs,
                test_mfs,
                y_train,
                y_test)

    def build(self,
              reps_dir: str,
              input_rep: str,
              output_rep: str,
              datasets: list,
              n_sub_folds: int) -> None:

        for dataset, folds, n_folds in datasets:
            self.build_features(reps_dir,
                                input_rep,
                                output_rep,
                                dataset,
                                folds,
                                n_folds,
                                n_sub_folds)
