import os
import json
from itertools import product

import numpy as np
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import loggers

from src.models.attention_models import SingleOptimizerML, DualOptmizerML, ExpandedAttention
from src.models.models import MLP
from src.attn_mfs import get_error_est_mfs

with open("data/settings.json", 'r') as fd:
    settings = json.load(fd)

DATASETS = settings["DATASETS"]
CLFS_SETS = settings["CLFS_SETS"]
DATA_SOURCE = settings["DATA_SOURCE"]
META_LAYERS = settings["META_LAYER"]
USE_MF = settings["USE_MF"]
BATCH_SIZE = settings["BATCH_SIZE"]
base_output = settings["OUTPUT"]
mf_input_dir = settings["MF_INPUT_DIR"]
apply_upper = settings["APPLY_UPPER"]
max_epochs = settings["EPOCHS"]

iters = product(DATASETS,
                CLFS_SETS,
                USE_MF,
                META_LAYERS)

for dataset_setup, c_set, use_mf, meta_layer_name in iters:

    dataset, class_number, n_folds = dataset_setup
    clf_set = settings[c_set]
    clfs_number = len(clf_set)

    for fold in np.arange(n_folds):

        attn_x_train, attn_x_test = get_error_est_mfs(DATA_SOURCE,
                                                      dataset,
                                                      clf_set,
                                                      fold,
                                                      n_folds)

        if meta_layer_name == "single_opt":

            meta_layer = SingleOptimizerML(class_number,
                                        clfs_number,
                                        attn_x_train[0].shape[1],
                                        1,
                                        0.1,
                                        apply_upper,
                                        False)
            print("Single optimizer meta layer loaded.")

        elif meta_layer_name == "dual_opt":

            meta_layer = DualOptmizerML(class_number,
                                        clfs_number,
                                        attn_x_train[0].shape[1],
                                        1,
                                        0.1,
                                        apply_upper)
            print("Dual optimizer meta layer loaded.")
        
        elif meta_layer_name == "mlp":

            meta_layer = MLP(X_train.shape[1], X_train.shape[1], class_number)
        else:

            raise ("Model option not valid. The options are: single_opt and dual_opt.")

        # Making logdir.
        clf_sufix = '/'.join(sorted([f"{c[0]}_{c[1]}" for c in clf_set]))
        setup_str = f"{dataset}/{n_folds}_folds/attention/{meta_layer_name}/use_mf_{use_mf}/batch_{batch_size}/{clf_sufix}"
        output_dir = f"{base_output}/{setup_str}/fold_{fold}"
        os.makedirs(output_dir, exist_ok=True)

        
        tensorboard_dir = f"tensorboard/{setup_str}"
        tb_logger = loggers.TensorBoardLogger(
            save_dir=f"{tensorboard_dir}/tensorboard")

        preds = trainer.predict(meta_layer, dataloaders=test_loader)
        preds = np.hstack(preds)

        macro = f1_score(y_test, preds, average="macro")
        micro = f1_score(y_test, preds, average="micro")

        print(
            f"[DATASET: {dataset} / FOLD - {fold}] Macro: {macro} - Micro: {micro}")

        with open(f"{output_dir}/scoring.json", 'w') as fd:
            scoring = {
                "f1_macro": macro,
                "f1_micro": micro
            }
            json.dump(scoring, fd)
