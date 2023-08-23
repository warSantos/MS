import os
import json
from sys import exit
from itertools import product

import numpy as np
from sklearn.metrics import f1_score

import pytorch_lightning as pl
from pytorch_lightning import loggers

from src.cross_attention import (CrossAttDataHandler,
                                 CrossAttMetaLayer,
                                 DualCrossAttDataHandler,
                                 DualCrossAttMetaLayer)

with open("data/cross_attention.json", 'r') as fd:
    settings = json.load(fd)

DATASETS = settings["DATASETS"]
CLFS_SETS = settings["CLFS_SETS"]
DATA_SOURCE = settings["DATA_SOURCE"]
META_LAYERS = settings["META_LAYER"]
batch_size = settings["BATCH_SIZE"]
base_output = settings["OUTPUT"]
apply_upper = settings["APPLY_UPPER"]
max_epochs = settings["EPOCHS"]

iters = product(DATASETS,
                CLFS_SETS,
                META_LAYERS)

for dataset_setup, c_set, meta_layer_name in iters:

    dataset, class_number, n_folds = dataset_setup
    clf_set = settings[c_set]
    clfs_number = len(clf_set)
    data_handler_init = CrossAttDataHandler if meta_layer_name == "cross_attention" else DualCrossAttDataHandler
    model_type = CrossAttMetaLayer if meta_layer_name == "cross_attention" else DualCrossAttMetaLayer
    
    for fold in np.arange(n_folds):

        # Making logdir.
        clf_sufix = '/'.join(sorted([f"{c[0]}_{c[1]}" for c in clf_set]))
        setup_str = f"{dataset}/{n_folds}_folds/attention/{meta_layer_name}/use_mf_False/batch_{batch_size}/{clf_sufix}"
        output_dir = f"{base_output}/{setup_str}/fold_{fold}"
        os.makedirs(output_dir, exist_ok=True)

        data_handler = data_handler_init(DATA_SOURCE,
                                         dataset,
                                         settings[c_set],
                                         fold,
                                         n_folds,
                                         batch_size,
                                         6)

        model = model_type(data_handler.n_labels,
                           data_handler.query_dim,
                           data_handler.n_clfs,
                           data_handler.n_clfs,
                           0.1,
                           False)

        train = data_handler.data_prepare("train")
        val = data_handler.data_prepare("val")

        trainer = pl.Trainer(accelerator="gpu",
                             callbacks=[model.get_stoper(epochs=max_epochs)],
                             max_epochs=max_epochs)
        trainer.fit(model, train, val)

        test = data_handler.data_prepare("test")
        preds = trainer.predict(model, dataloaders=test)
        preds = np.vstack(preds).argmax(axis=1)

        macro = f1_score(data_handler.y_test, preds, average="macro")
        micro = f1_score(data_handler.y_test, preds, average="micro")

        print(
            f"[DATASET: {dataset} / FOLD - {fold}] Macro: {macro} - Micro: {micro}")

        with open(f"{output_dir}/scoring.json", 'w') as fd:
            scoring = {
                "f1_macro": macro,
                "f1_micro": micro
            }
            json.dump(scoring, fd)
        print(f"\t{output_dir}")