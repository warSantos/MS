import numpy as np
from src.utils.utils import (
    count_to_vector, clean_text, save_mfs, get_train_test, replace_nan_inf, load_df)


class StatisticalMFs:

    def doc_len(self, X):

        # return np.array([np.where(r > 0)[0].shape[0] for r in X])
        return np.count_nonzero(X, axis=1)

    def uniq_terms(self, X):

        return np.count_nonzero(X == 1, axis=1)

    def repeated_terms(self, X):

        # return np.array([np.where(r > 1)[0].shape[0] for r in X])
        return np.count_nonzero(X > 1, axis=1)

    def mean_word_len(self, texts, tolwer=True):

        sents = [clean_text(text, tolower=tolwer, tokenized=True)
                 for text in texts]
        return np.round([np.mean([len(word) for word in sent]) for sent in sents], decimals=2)

    def transform(self, texts):

        cv, X = count_to_vector(texts)
        X = X.toarray()

        mfs = []
        mfs.append(self.doc_len(X))
        mfs.append(self.uniq_terms(X))
        mfs.append(self.repeated_terms(X))
        mfs.append(self.mean_word_len(texts))

        return np.vstack(mfs).T

    def build_features(self, dset):

        df = load_df(f"data/datasets/{dset}.csv")
        # For the first cross-val level.
        for fold in np.arange(10):
            # Separate da data in train, test.
            train, test = get_train_test(df, fold)

            train_mfs = self.transform(train.docs.values)
            test_mfs = self.transform(test.docs.values)

            train_mfs = replace_nan_inf(train_mfs)
            test_mfs = replace_nan_inf(test_mfs)

            save_mfs(dset, "stat", fold, train_mfs, test_mfs)

    def build(self, datasets=["webkb", "20ng", "reut", "acm"]):

        for dset in datasets:
            self.build_features(dset)


"""
from src.stat_mfs.stat import doc_len
import numpy as np
m = np.random.randint(0,2, size=(5,5))
doc_len(m)

from src.stat_mfs.stat import mean_word_len
x = ["hey, my name is michal scotth. Don't we have talk before?", "In a sunny day, can't be sad."]
mean_word_len(x)
"""
