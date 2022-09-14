import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

import json
from itertools import product
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer

from sklearn.feature_selection import chi2
from scipy.stats import kurtosis, skew

from src.utils.utils import count_to_vector
from src.utils.utils import (
    get_train_test, stratfied_cv, replace_nan_inf, save_mfs, load_df, clean_text)


class InfoMFS:

    def description(self, values, simple=False):

        mean = np.mean(values, axis=1)
        if simple:
            return mean
        
        std = np.std(values, axis=1)
        median = np.median(values, axis=1)
        vmin = np.min(values, axis=1)
        vmax = np.max(values, axis=1)
        kt = kurtosis(values, axis=1)
        sk = skew(values, axis=1)
        feats = np.vstack([mean,std,median,vmin,vmax,kt,sk]).T
        qt = np.quantile(values, [0, 0.25, 0.5, 0.75, 1], axis=1).T
        feats = np.hstack([feats, qt])
        return feats

    # Get the medium TF-IDF from documents.
    # X_train: count word matrix from train documents.
    # X_test: count word matrix from train documents.
    def tfidf(self, X_train, X_test):

        tf = TfidfTransformer()
        tf.fit(X_train)
        idfs_rep = tf.transform(X_test).toarray()
        docs_idf = np.dot(idfs_rep > 0, tf.idf_)
        adjust = idfs_rep.shape[1] + 1
        norm = adjust - np.count_nonzero(idfs_rep, axis=1)
        return docs_idf / norm

    def tfidf_topnw(self, X_train, X_test, topn=15, simple=False):

        tf = TfidfTransformer()
        tf.fit(X_train)
        idfs_rep = tf.transform(X_test).toarray()
        idfs_rep.sort(axis=1)
        #midf = np.sum(idfs_rep[:, -topn:], axis=1) / topn
        if topn > -1:
            return self.description(idfs_rep[:, -topn:], simple=simple)
        return self.description(idfs_rep, simple=simple)

    def chi2(self, X_train, X_test, y_train):

        chi2_values, _ = chi2(X_train, y_train)
        idfv = np.dot(X_test > 0, chi2_values)
        total = np.count_nonzero(X_test, axis=1)
        return idfv / total

    def chi2_topnw(self, X_train, X_test, y_train, topn=15, simple=False):

        chi2_values, _ = chi2(X_train, y_train)
        Xc = X_test.copy()
        Xc[Xc > 0] = 1
        cm = Xc * chi2_values
        cm.sort(axis=1)
        #mc = np.sum(cm[:, -topn:], axis=1) / topn
        if topn > -1:
            return self.description(cm[:, -topn:], simple=simple)
        return self.description(cm, simple=simple)

    def transform(self, train_texts, test_texts, train_classes, params=None):

        cv, X_train = count_to_vector(train_texts)
        X_test = cv.transform(test_texts).toarray()

        mf_idf = self.tfidf_topnw(X_train, X_test, topn=params["topn"], simple=params["simple"])
        mf_chi2 = self.chi2_topnw(
            X_train, X_test, train_classes, topn=params["topn"], simple=params["simple"])

        return mf_idf, mf_chi2
    
    def build_features(self, dset, params=None, params_prefix=""):

        df = load_df(f"data/datasets/{dset}.csv")

        # For the first cross-val level.
        for fold in np.arange(10):
            # Separate da data in train, test
            train, test = get_train_test(df, fold)
            # List of train mfs.
            train_mfs = []
            align = []
            # Make new splits to generate train MFs.
            splits = stratfied_cv(train.docs, train.classes, dset=dset, load_splits=False)
            for inner_fold in splits.itertuples():

                inner_train_texts = train.docs.values[inner_fold.train]
                inner_train_classes = train.classes.values[inner_fold.train]
                inner_test_texts = train.docs.values[inner_fold.test]

                inner_idf, inner_chi2 = self.transform(
                    inner_train_texts, inner_test_texts, inner_train_classes, params=params)

                new_mfs = np.hstack([inner_idf, inner_chi2])#.T
                train_mfs.append(new_mfs)
                align.append(inner_fold.align_test)

            align = np.hstack(align)
            sorted_indexes = np.argsort(align)
            train_mfs = np.vstack(train_mfs)[sorted_indexes]
            
            # Generating test meta-features.
            outer_idf, outer_chi2 = self.transform(train.docs.values, test.docs.values,
                                               train.classes.values, params=params)
            test_mfs = np.hstack([outer_idf, outer_chi2])#.T

            #train_mfs = replace_nan_inf(train_mfs)
            test_mfs = replace_nan_inf(test_mfs)

            save_mfs(dset, "info", fold, train_mfs, test_mfs, params_prefix=params_prefix)
    
    def build(self, datasets=["webkb", "20ng", "reut", "acm"]):
        
        with open("data/configs/info/info.json", 'r') as fd:
            
            params = json.load(fd)
            for dset, topn in product(datasets, params["topn"]):
                iter_params = {
                    "topn": topn,
                    "simple": params["simple"]
                }
                params_prefix = f"topn/{topn}/simple/{str(params['simple'])}"
                print(f"Dataset: {dset}\n\tParameters: {params_prefix}")
                self.build_features(dset, iter_params, params_prefix=params_prefix)

    def lazy_description(self, values, simple=False):

        if values.shape[0] == 0:
            return np.zeros(12)
        
        mean = np.mean(values)
        if simple:
            return mean
        
        std = np.std(values)
        median = np.median(values)
        vmin = np.min(values)
        vmax = np.max(values)
        kt = kurtosis(values)
        sk = skew(values)
        feats = np.array([mean, std, median, vmin, vmax, kt, sk])
        qt = np.quantile(values, [0, 0.25, 0.5, 0.75, 1])
        feats = np.hstack([feats, qt])
        return feats

    def lazy_tfidf(self, train, test):
        tf = TfidfVectorizer()
        tf.fit(train)
        return tf.transform(test).toarray()

    def lazy_chi2(self, train, test, train_classes):

        cv = CountVectorizer()
        cv.fit(train)
        X_train = cv.transform(train)
        Xc = cv.transform(test).toarray()
        chi2_values, _ = chi2(X_train, train_classes)
        Xc[Xc > 0] = 1
        return Xc * chi2_values

    def lazy_topw(self, matrix, params):

        reps = []
        # For each document.
        for d in matrix:
            document = np.copy(d)
            num_words = np.count_nonzero(document)
            topn = int(params["topn"] * num_words)
            document[::-1].sort()
            reps.append(self.lazy_description(document[:topn]))
        
        return reps

    def lazy_transform(self, train_texts, test_texts, train_classes, params=None):

        clean_train = [ clean_text(text) for text in train_texts ]
        clean_test = [ clean_text(text) for text in test_texts ]

        if params["strategy"] == "tfidf":
            test_matrix = self.lazy_tfidf(clean_train, clean_test)
            return np.array(self.lazy_topw(test_matrix, params))
        elif params["strategy"] == "chi2":
            test_matrix = self.lazy_chi2(clean_train, clean_test, train_classes)
            return np.array(self.lazy_topw(test_matrix, params))
        else:
            test_matrix = self.lazy_tfidf(clean_train, clean_test)
            tf = np.array(self.lazy_topw(test_matrix, params))
            test_matrix = self.lazy_chi2(clean_train, clean_test, train_classes)
            ch = np.array(self.lazy_topw(test_matrix, params))
            return np.hstack([tf, ch])

    def lazy_build_features(self, dset, params=None, params_prefix=""):

        df = load_df(f"data/datasets/{dset}.csv")

        # For the first cross-val level.
        for fold in np.arange(10):
            # Separate da data in train, test
            train, test = get_train_test(df, fold)
            # List of train mfs.
            train_mfs = []
            align = []
            # Make new splits to generate train MFs.
            splits = stratfied_cv(train.docs, train.classes, dset=dset, load_splits=False)
            for inner_fold in splits.itertuples():

                inner_train_texts = train.docs.values[inner_fold.train]
                inner_train_classes = train.classes.values[inner_fold.train]
                inner_test_texts = train.docs.values[inner_fold.test]
                
                new_mfs = self.lazy_transform(
                   inner_train_texts, inner_test_texts, inner_train_classes, params=params)

                train_mfs.append(new_mfs)
                align.append(inner_fold.align_test)

            align = np.hstack(align)
            sorted_indexes = np.argsort(align)
            train_mfs = np.vstack(train_mfs)[sorted_indexes]
            
            # Generating test meta-features.
            test_mfs = self.lazy_transform(train.docs.values, test.docs.values,
                                               train.classes.values, params=params)
            test_mfs = replace_nan_inf(test_mfs)

            save_mfs(dset, params["strategy"], fold, train_mfs, test_mfs, params_prefix=params_prefix)

    def lazy_build(self, datasets=["webkb", "20ng", "reut", "acm"]):
        
        with open("data/configs/info/special_info.json", 'r') as fd:
            
            params = json.load(fd)
            for dset, topn, strategy in product(datasets, params["topn"], params["strategy"]):
                iter_params = {
                    "topn": topn,
                    "simple": params["simple"],
                    "strategy": strategy
                }
                params_prefix = f"topn/{topn}/simple/{str(params['simple'])}"
                print(f"Dataset: {dset}\n\tParameters: {params_prefix}")
                self.lazy_build_features(dset, iter_params, params_prefix=params_prefix)
    

"""
from src.stat_mfs.info import InfoMFS
inf = InfoMFS()
inf.build()
"""
