import os
from sys import prefix
import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import vstack
from scipy.sparse import csr_matrix

from src.utils.utils import (stratfied_cv, replace_nan_inf, save_mfs)

from src.dist_mfs.mf.centbased import MFCent
from src.dist_mfs.mf.knnbased import MFKnn
from src.dist_mfs.mf.bestk import kvalues


class DistMFs:

    def transform(self, X_train, y_train, X_test, dset):

        mfs = []
        cent_cosine = MFCent("cosine")
        cent_cosine.fit(X_train, y_train)
        mfs.append(cent_cosine.transform(X_test))
        
        #cent_l2 = MFCent("l2")
        #cent_l2.fit(X_train, y_train)
        #mfs.append(cent_l2.transform(X_test))

        #knn_cosine = MFKnn("cosine", kvalues[dset])
        #knn_cosine.fit(X_train, y_train)
        #mfs.append(knn_cosine.transform(X_test))
        #
        #knn_l2 = MFKnn("l2", kvalues[dset])
        #knn_l2.fit(X_train, y_train)
        #mfs.append(knn_l2.transform(X_test))

        return np.hstack(mfs)

    def build_features(self, dset):

        dir_reps = f"data/embeddings/bert/base/{dset}/None"

        # For the first cross-val level.
        for fold in np.arange(0, 10):
            print(f"\tfold: {fold}".upper())
            reps_train = load_svmlight_file(
                f"{dir_reps}/train{fold}.gz")
            X_train = reps_train[0]
            y_train = reps_train[1]
            # List of train mfs.
            train_mfs = []
            align = []
            # Make new splits to generate train MFs.
            splits = stratfied_cv(X_train, y_train, dset=dset, fold=fold, load_splits=False)
            for inner_fold in splits.itertuples():

                inner_X_train = X_train[inner_fold.train].copy()
                inner_y_train = y_train[inner_fold.train].copy()
                inner_X_test = X_train[inner_fold.test]
                inner_y_test = y_train[inner_fold.test]

                # Applying oversampling when it is needed.
                for c in set(inner_y_test) - set(inner_y_train):
                    
                    perturbation = np.random.rand(1, inner_X_train.shape[1]) / 100
                    sintetic = np.mean(X_train[y_train == c], axis=0) + perturbation
                    inner_X_train = csr_matrix(vstack([inner_X_train, sintetic]))
                    inner_y_train = np.hstack([inner_y_train, [c]])

                new_mfs = self.transform(
                    inner_X_train, inner_y_train, inner_X_test, dset)
                train_mfs.append(new_mfs)
                align.append(inner_fold.align_test)
                
            
            align = np.hstack(align)
            sorted_indexes = np.argsort(align)
            train_mfs = np.vstack(train_mfs)[sorted_indexes]

            # Generating test meta-features.
            reps_test = load_svmlight_file(
                f"{dir_reps}/test{fold}.gz")
            X_test = reps_test[0]
            test_mfs = self.transform(X_train, y_train, X_test, dset)

            train_mfs = replace_nan_inf(train_mfs)
            test_mfs = replace_nan_inf(test_mfs)

            save_mfs(dset, "centroid_cosine", fold, train_mfs, test_mfs)

    def build(self, datasets=["webkb", "20ng", "reut", "acm"]):

        for dset in datasets:
            print(f"dataset/{dset}".upper())
            self.build_features(dset)


if __name__=="__main__":

    d = DistMFs()
    d.build()

"""
from src.dist_mfs.dist import DistMFs
inf = DistMFs()
inf.build()
"""
