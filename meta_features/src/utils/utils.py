# File with shared functions.
import os
import nltk
import string
import jsonlines
import numpy as np
import pandas as pd
from typing import Tuple, List
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold


tokenizer = nltk.RegexpTokenizer(r"\w+")
dstpw = {word: True for word in set(nltk.corpus.stopwords.words('english'))}

# Load a pre processed dataset.
# path_df: path to the pre processed dataset. It must be a CSV file.


def load_df(path_df):

    return pd.read_csv(path_df)

# Replace nan and inf values by another value (default is zero)
# a: array to replace the values.
# value: value to replace nan and inf.


def replace_nan_inf(a, value=0):

    a[np.isinf(a) | np.isnan(a)] = value
    return a


def save_mfs(output_dir,
             dataset: str,
             type_mf: str,
             fold: int,
             X_train: np.ndarray,
             X_test: np.ndarray,
             y_train: np.ndarray,
             y_test: np.ndarray,
             params_prefix: str = ""):

    base_dir = f"{output_dir}/features/{type_mf}/{params_prefix}/{fold}/{dataset}/"
    print(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    np.savez(f"{base_dir}/train", X_train=X_train, y_train=y_train)
    np.savez(f"{base_dir}/test", X_test=X_test, y_test=y_test)

# Remove punctuation from a text.
# text: string to remove punctuation from.


def remove_punct(text):

    return text.translate(str.maketrans('', '', string.punctuation))

# Remove punctuation from a text.
# text: string to remove stop words from.


def remove_stopwords(text):

    return ' '.join([word for word in text.split() if word not in dstpw])

# Clean a string (pre processing).
# text: string/text to be clean.
# tolower: convert string to lowercase.
# tokenized: return the srting as a list of tokens.


def clean_text(text, tolower=True, tokenized=False):

    t = text.replace("'", ' ')
    t = remove_punct(t)

    if tolower:
        t = t.lower()

    t = remove_stopwords(t)

    if tokenized:
        t = t.split()

    return t


# Count words freq from a collection of documents.
# texts: text to count words freq.
# preproc: pre process the text (remove stopwords,
#   punctuation and converto lowercase) or not.
def count_to_vector(texts, preproc=True, pretrained=True):

    t = texts
    if preproc:
        t = [clean_text(text) for text in texts]

    cv = CountVectorizer()
    cv.fit(t)
    features = cv.transform(t)
    return cv, features


# Load arrays from a especfific directory in a cross-validation fashion.
# vectors_dir: directory with vectors.
# fold: which fold the function shall load.
def get_train_test(df, fold, fold_col="folds_id"):

    test = df[df[fold_col] == fold]
    train = df[df[fold_col] != fold]

    return train, test

# Separate data in cv splits with stratified cross validation.
# X: preditivie features.
# y: labels (class of the documents).
# dataset: dataset's name.
# fold: if True, load the split settings instead make new ones.
# save_cv: save split settings. It only works if the dataset is not None and lfold is not None.


def stratfied_cv(dataset: str,
                 fold: int,
                 X: np.ndarray,
                 y: np.ndarray,
                 n_splits: int,
                 save_cv: bool,
                 load_splits: bool = True):

    sp_dir = f"data/configs/splits/{dataset}/{fold}"
    sp_path = f"{sp_dir}/{n_splits}_splits.pkl"

    if load_splits and os.path.exists(sp_path):

        return pd.read_pickle(sp_path)

    sfk = StratifiedKFold(n_splits=n_splits)
    align_indexes = np.arange(X.shape[0])
    sfk.get_n_splits(X, y)
    indexes = [[fold, train_idxs, test_idxs, align_indexes[train_idxs], align_indexes[test_idxs]]
               for fold, (train_idxs, test_idxs) in enumerate(sfk.split(X, y))]

    splits = pd.DataFrame(
        indexes, columns=["fold", "train", "test", "align_train", "align_test"])

    if save_cv:
        os.makedirs(sp_dir, exist_ok=True)
        splits.to_pickle(sp_path)

    return splits


def load_x_y(
        file: str,
        test_train: str
) -> Tuple[np.ndarray, np.ndarray]:
    loaded = np.load(file, allow_pickle=True)

    X = loaded[f"X_{test_train}"]
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


def load_y(labels_dir: str,
           dataset: str,
           fold: int):

    ldir = f"{labels_dir}/{dataset}/{fold}"
    return np.load(f"{ldir}/train.npy"), np.load(f"{ldir}/test.npy")


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


"""
from src.utils.utils import stratfied_cv
import numpy as np
X = np.random.randint(0,2, size=(100,5))
y = np.random.randint(0,4, 100)
st = stratfied_cv(X,y)

"""
