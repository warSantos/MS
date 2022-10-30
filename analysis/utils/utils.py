import os
import json
import pickle
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from collections import Counter

def load_pred_from_json(j):
    with open(j, 'r') as fd:
        try:
            return json.load(fd)["y_pred"]
        except:
            print(j)
            traceback.print_exc()
            return False

def load_preds(DATASETS, ALGORITHMS, preds_path="data/preds.pickle", from_scratch=True):
    
    if os.path.exists(preds_path) and not from_scratch:
        with open(preds_path, 'rb') as handle:
            print("LOADING PRE PROCESSED PREDS.")
            return pickle.load(handle)
    
    print("LOADING PREDS FROM SCRATCH")
    d_preds = {}
    for dset, alg, fold in tqdm(product(DATASETS, ALGORITHMS, np.arange(10))):
        
        if dset not in d_preds:
            d_preds[dset] = {}
        
        if alg == "bert":
            j = f"/home/claudiovaliense/projetos/kaggle/{dset}_bert{fold}_pred.json"
            if not os.path.exists(j):
                j = f"/home/welton/data/kaggle/{dset}_bert{fold}_pred.json"
        else:
            j = f"/home/claudiovaliense/projetos/kaggle/{dset}_bert{fold}_{alg}"
        
        y_pred = load_pred_from_json(j)
        if not y_pred:
            return None

        if alg not in d_preds[dset]:
            d_preds[dset][alg] = y_pred
        else:
            d_preds[dset][alg] = np.hstack([d_preds[dset][alg], y_pred])
    
    with open(preds_path, 'wb') as handle:
        pickle.dump(d_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return d_preds

def load_preds_clfs_bert(DATASETS, ALGORITHMS, preds_path="data/bert_clfs_preds.pickle", from_scratch=True):
    
    if os.path.exists(preds_path) and not from_scratch:
        with open(preds_path, 'rb') as handle:
            print("LOADING PRE PROCESSED PREDS.")
            return pickle.load(handle)
    
    print("LOADING PREDS FROM SCRATCH")
    d_preds = {}
    for dset, alg, fold in tqdm(product(DATASETS, ALGORITHMS, np.arange(10))):
        
        if dset not in d_preds:
            d_preds[dset] = {}
        
        p = f"/home/welton/data/clfs_output/split_10_with_val/{dset}/10_folds/{alg}/{fold}/probas.npy"
        y_pred = np.load(p).argmax(axis=1)

        if alg not in d_preds[dset]:
            d_preds[dset][alg] = y_pred
        else:
            d_preds[dset][alg] = np.hstack([d_preds[dset][alg], y_pred])
    
    with open(preds_path, "wb") as handle:
        pickle.dump(d_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return d_preds


def get_datasets(DATASETS, path="../../../stacking/output/datasets/__dset__.csv", sep=','):

    pd_datasets = {}
    for dset in DATASETS:
        path_df = path.replace("__dset__", dset)
        pd_datasets[dset] = pd.read_csv(path_df, sep=sep)
    return pd_datasets


def make_pandas_df(d_preds):

    for dset in d_preds:
        m = pd.DataFrame(d_preds[dset])
        base = pd_datasets[dset]

        hits_count = np.sum(m.values == base.classes.values[:, None], axis=1)

        conc_pred = []
        conc_size = []
        folds_ids = []
        for idx in np.arange(m.values.shape[0]):
            counts = Counter(m.values[idx].tolist())
            mc = counts.most_common()[0]
            conc_pred.append(mc[0])
            conc_size.append(mc[1])

        m["conc_pred"] = conc_pred
        m["conc_size"] = conc_size
        m["classes"] = base.classes.values
        m["docs"] = base.docs.values
        m["folds_id"] = pd_datasets[dset].folds_id

        df_path = f"/home/welton/data/datasets/pandas/bert/{dset}.csv"
        m.to_csv(df_path, sep=';', index=False)


IDS_MODELS = {
    "linear_svm/fast_text_1/raw_folds": "sfr",
    "linear_svm/pte_1/raw_folds": "spr",
    "linear_svm/tf_idf_1/fs": "str",
    "linear_svm/tf_idf_1/meta_features_1/knn_cos": "stmk",
    "knn/fast_text_1/raw_folds": "kfr",
    "knn/pte_1/raw_folds": "kpr",
    "knn/tf_idf_1/fs": "ktr",
    "knn/tf_idf_1/meta_features_1/knn_cos": "ktmk",
    "lr/fast_text_1/raw_folds": "lfr",
    "lr/pte_1/raw_folds": "lpr",
    "lr/tf_idf_1/fs": "ltr",
    "lr/tf_idf_1/meta_features_1/knn_cos": "ltmk",
    "xgboost/fast_text_1/raw_folds": "xfr",
    "xgboost/pte_1/raw_folds": "xpr",
    "xgboost/tf_idf_1/fs": "xtr",
    "xgboost/tf_idf_1/meta_features_1/knn_cos": "xtmk",
    "xlnet": "xlnet",
    "bert": "bert",
    "bert/rep_paper": "rep_bert"
}