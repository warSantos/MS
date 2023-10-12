import os

import json
from itertools import product

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

from src.nn.data_loader import Loader
from src.nn.model import Transformer, TextFormater, FitHelper
from src.nn.do_train import get_train_probas
from src.aws.awsio import (aws_path_exists,
                           aws_stop_instance,
                           store_nparrays_in_aws)

with open("data/nn_settings.json", 'r') as fd:
    settings = json.load(fd)

SEED = settings["SEED"]
DATA_SOURCE = settings["DATA_SOURCE"]
DATASETS = settings["DATASETS"]
CLFS = settings["CLFS"]
DO_TEST = settings["DO_TEST"]
DO_TRAIN = settings["DO_TRAIN"]
n_sub_folds = settings["N_SUB_FOLDS"]
os.environ["DATA_SOURCE"] = DATA_SOURCE
os.environ["AWS_BUCKET"] = settings["AWS_BUCKET"]
os.environ["AWS_PROFILE"] = settings["AWS_PROFILE"]

iters = product(DATASETS, CLFS)

for (dataset, n_folds), (clf_name, clf_short_name) in iters:
    
    data_handler = Loader(DATA_SOURCE, dataset, n_folds)
    text_params = settings["TEXT_PARAMS"].copy()
    text_params["model_name"] = clf_name
    
    for fold in np.arange(n_folds):
                
        model_params = settings["MODEL_PARAMS"][clf_short_name].copy()
        model_params["num_labels"] = data_handler.num_labels

        print(f"[{dataset.upper()} / {model_params['model_name']}] - FOLD: {fold}")
        output_dir = f"{DATA_SOURCE}/normal_probas/split_{n_folds}/{dataset}/{n_folds}_folds/{clf_short_name}/{fold}"
        os.makedirs(output_dir, exist_ok=True)
        test_path = f"{output_dir}/test.npz"
        eval_logits_path = f"{output_dir}/eval_logits.npz"
        test_logits_path = f"{output_dir}/test_logits.npz"

        print(test_path)
        # Se este fold ainda não foi executado.
        if DO_TEST and not aws_path_exists(test_path):
            print("Builind test probabilities...")
            
            # Preparing data.
            X_train, y_train = data_handler.get_X_y(fold, "train")
            X_test, y_test = data_handler.get_X_y(fold, "test")
            X_val, y_val = data_handler.get_X_y(fold, "val")
            text_formater = TextFormater(**text_params)
            train = text_formater.prepare_data(X_train, y_train, shuffle=True)
            test = text_formater.prepare_data(X_test, y_test)
            val = text_formater.prepare_data(X_val, y_val)

            # Setting model's parameters.
            model_params["len_data_loader"] = len(train)
            model = Transformer(**model_params)

            # Traning model.
            trainer = FitHelper().fit(model, train, val, model.max_epochs, model.seed)
            
            # Predicting.
            test_l = np.vstack([ l["logits"] for l in trainer.predict(model, test) ])
            eval_l = np.vstack([ l["logits"] for l in trainer.predict(model, val) ])

            # Saving outputs.
            store_nparrays_in_aws(test_path, {"X_test": softmax(test_l, axis=1),
                                              "y_test": y_test})
            store_nparrays_in_aws(eval_logits_path, {"X_eval": eval_l,
                                                     "y_eval": y_val})
            store_nparrays_in_aws(test_logits_path, {"X_test": test_l,
                                                     "y_test": y_test})
            
            # Printing model performance.
            y_pred = test_l.argmax(axis=1)
            print(f"Macro: {f1_score(y_test, y_pred, average='macro')}")
            print(f"Micro: {f1_score(y_test, y_pred, average='micro')}")
            
        train_path = f"{output_dir}/train"
        # If train probabilities weren't computed yet.
        if DO_TRAIN and not aws_path_exists(train_path):
            print("Builind train probabilities...")
            # Computing train probabilities.
            get_train_probas(data_handler,
                             output_dir,
                             fold,
                             n_sub_folds,
                             model_params,
                             text_params)

aws_stop_instance()