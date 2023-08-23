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
                        dataset: str,
                        fold: int,
                        X: np.ndarray,
                        y: np.ndarray,
                        n_splits: int,
                        save_cv: bool):

        # List of train mfs.
        train_mfs = []
        align = []

        # Make new splits to generate train MFs.
        splits = stratfied_cv(dataset,
                              fold,
                              X,
                              y,
                              n_splits,
                              save_cv)

        for fold_idx, split in enumerate(splits.itertuples()):

            print(f"\t[{dataset.upper()} / FOLD: - {fold_idx+1}/{n_splits} ]")

            X_train = X[split.train].copy()
            X_test = X[split.test]

            y_train = y[split.train].copy()
            y_test = y[split.test]

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
            align.append(split.align_test)

        align = np.hstack(align)
        sort = align.argsort()
        return np.vstack(train_mfs)[sort]

    def build_features(self,
                       labels_dir: str,
                       output_dir: str,
                       dataset: str,
                       folds: list,
                       n_folds: int,
                       n_sub_folds: int
                       ) -> None:

        # For the first cross-val level.
        for fold in folds:
            print(f"[{dataset.upper()} / FOLD: - {fold+1}/{n_folds} ]")
            X_train, X_test = load_bert_reps(dataset, fold)
            y_train, y_test = load_y(labels_dir, dataset, fold)

            
            train_mfs = self.build_train_mfs(
                dataset,
                fold,
                X_train,
                y_train,
                n_sub_folds,
                True)
            

            # Generating test meta-features.
            
            test_mfs = self.transform(dataset,
                                      X_train,
                                      X_test,
                                      y_train)

            
            #train_mfs = np.zeros(X_train.shape)
            #test_mfs = np.zeros(X_test.shape)
            #train_mfs = replace_nan_inf(train_mfs)
            #test_mfs = replace_nan_inf(test_mfs)

            save_mfs(output_dir,
                dataset,
                "bert_dists",
                fold,
                train_mfs,
                test_mfs,
                y_train,
                y_test)

    def build(self,
              labels_dir: str,
              output_dir: str,
              datasets: list,
              folds: list,
              n_folds: int,
              n_sub_folds: int) -> None:

        for dataset in datasets:
            self.build_features(labels_dir,
                                output_dir,
                                dataset,
                                folds,
                                n_folds,
                                n_sub_folds)
