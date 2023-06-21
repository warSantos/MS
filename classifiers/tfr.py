import os
import json
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import torch
from torch.nn.functional import softmax
from src.nn.data_loader import Loader
from src.nn.model import Transformer, TextFormater
from src.nn.data_loader import get_doc_by_id
from src.nn.do_train import get_train_probas

with open("data/nn_settings.json", 'r') as fd:
    settings = json.load(fd)

SEED = settings["SEED"]
DATA_DIR = settings["DATA_DIR"]
DATA_SOURCE = settings["DATA_SOURCE"]
DATASETS = settings["DATASETS"]
CLFS = settings["CLFS"]
n_sub_folds = settings["N_SUB_FOLDS"]

iters = product(DATASETS, CLFS)

for (dataset, n_folds), (clf_name, clf_short_name) in iters:
    
    text_params = settings["TEXT_PARAMS"].copy()
    text_params["model_name"] = clf_name
    
    for fold in np.arange(n_folds):
                
        model_params = settings["MODEL_PARAMS"][clf_short_name].copy()

        print(f"[{dataset.upper()} / {model_params['model_name']}] - FOLD: {fold}")
        base_path = f"{DATA_DIR}/normal_probas/split_{n_folds}/{dataset}/{n_folds}_folds/{clf_short_name}/{fold}"
        os.makedirs(base_path, exist_ok=True)
        test_path = f"{base_path}/test"
        eval_path = f"{base_path}/eval"
        eval_logits_path = f"{base_path}/eval_logits"
        test_logits_path = f"{base_path}/test_logits"

        data_handler = Loader(DATA_SOURCE, dataset, fold, n_folds)

        # Se este fold ainda não foi executado.
        if not os.path.exists(test_path):
            print("Builind test probabilities...")
            
            
            # Preparing data.
            X_train, y_train = data_handler.get_X_y(fold, "train")
            X_test, y_test = data_handler.get_X_y(fold, "test")
            X_val, y_val = data_handler.get_X_y(fold, "val")
            text_formater = TextFormater(**text_params)
            train = text_formater.prepare_data(X_train, y_train)
            test = text_formater.prepare_data(X_test, y_test)
            val = text_formater.prepare_data(X_val, y_val)

            # Setting model's parameters.
            model_params["len_data_loader"] = len(train)
            model_params["num_classes"] = data_handler.num_labels
            model = Transformer(**model_params)

            # Traning model.
            model.fit(train, val)

            # Predicting.
            test_l = model.predict(test)
            eval_l = model.predict(val)

            # Saving outputs.
            np.savez(test_path, X_test=softmax(test_l, axis=1))
            np.savez(eval_logits_path, X_eval=eval_l, y_eval=y_val)
            np.savez(test_logits_path, X_test=test_l, y_test=y_test)
            
            # Printing model performance.
            y_pred = test_l.argmax(axis=1)
            print(f"Macro: {f1_score(y_test, y_pred, average='macro')}")
            print(f"Micro: {f1_score(y_test, y_pred, average='micro')}")
            
        train_path = f"{base_path}/train"
        # If train probabilities weren't computed yet.
        if not os.path.exists(train_path):
            print("Builind train probabilities...")
            # Joining train indexes with validation indexes.
            idxs = data_handler.split_settings.iloc[fold]["train_idxs"]
            idxs += data_handler.split_settings.iloc[fold]["val_idxs"]
            # Sorting indexes. It's important to match the train document's probabilities
            # of data split with validantion and without validation (just train and test).
            sort = np.array(idxs).argsort()
            # Computing train probabilities.
            X = X_train + X_val
            get_train_probas(base_path,
                             get_doc_by_id(X, sort),
                             np.hstack([y_train, y_val])[sort],
                             data_handler.num_labels,
                             clf_name,
                             n_sub_folds)