import os
import json
import numpy as np
from itertools import product
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from utils import (transform_probas,
                   load_probs_fold,
                   load_upper_bound,
                   get_attention_labels,
                   load_labels_fold)

from attn_mfs import get_error_est_mfs

from models import StackingDataset, SingleOptimizerML, DualOptmizerML


with open("data/settings.json", 'r') as fd:
    settings = json.load(fd)

DATASETS = settings["DATASETS"]
CLFS_SETS = settings["CLFS_SETS"]
N_FOLDS = settings["N_FOLDS"]

meta_layer_name = settings["META_LAYER"]
use_mf = settings["USE_MF"]
base_output = settings["OUTPUT"]
mf_input_dir = settings["MF_INPUT_DIR"]
apply_upper = settings["APPLY_UPPER"]

iters = product(DATASETS, CLFS_SETS, np.arange(N_FOLDS))

probs_only = True

for dataset, c_set, fold in iters:

    clf_set = settings[c_set]

    labels_train, y_test = load_labels_fold(dataset, fold)
    probas_train, X_test = load_probs_fold(dataset, clf_set, fold)

    uppers = load_upper_bound(dataset, clf_set, fold)

    attn_y_train = get_attention_labels(clf_set, uppers, "train")
    attn_y_test = get_attention_labels(clf_set, uppers, "test")

    X_train, X_val, y_train, y_val, attn_y_train, attn_y_val = train_test_split(probas_train,
                                                                                labels_train,
                                                                                attn_y_train,
                                                                                stratify=labels_train,
                                                                                test_size=0.1,
                                                                                random_state=42)

    if not use_mf:
        attn_x_train = X_train
        attn_x_val = X_val
        attn_x_test = X_test
    else:
        attn_x_train, attn_x_test = get_error_est_mfs(mf_input_dir,
                                                      dataset,
                                                      clf_set,
                                                      fold,
                                                      10)

        attn_x_train, attn_x_val = train_test_split(attn_x_train,
                                                    stratify=labels_train,
                                                    test_size=0.1,
                                                    random_state=42)

    clfs_number = len(clf_set)
    classes_number = X_train[0].shape[1]
    max_epochs = 80
    batch_size = 128

    train_loader = DataLoader(dataset=StackingDataset(
        X_train, y_train, attn_x_train, attn_y_train), batch_size=batch_size, num_workers=16)
    val_loader = DataLoader(dataset=StackingDataset(
        X_val, y_val, attn_x_val, attn_y_val), batch_size=batch_size, num_workers=16)
    test_loader = DataLoader(dataset=StackingDataset(
        X_test, y_test, attn_x_test, attn_y_test), batch_size=batch_size, num_workers=16)

    if meta_layer_name == "single_opt":
        
        meta_layer = SingleOptimizerML(classes_number,
                                       clfs_number,
                                       attn_x_train[0].shape[1],
                                       1,
                                       0.1,
                                       apply_upper,
                                       settings["SUM_LOSSES"])
        print("Single optimizer meta layer loaded.")
    elif meta_layer_name == "dual_opt":

        meta_layer = DualOptmizerML(classes_number,
                                    clfs_number,
                                    attn_x_train[0].shape[1],
                                    classes_number,
                                    0.1,
                                    apply_upper)
        print("Dual optimizer meta layer loaded.")
    else:

        raise("Model option not valid. The options are: single_opt and dual_opt.")

    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[meta_layer.get_stoper()])

    trainer.fit(meta_layer, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    preds = trainer.predict(meta_layer, dataloaders=test_loader)
    preds = np.hstack(preds)

    macro = f1_score(y_test, preds, average="macro")
    micro = f1_score(y_test, preds, average="micro")

    print(f"[DATASET: {dataset} / FOLD - {fold}] Macro: {macro} - Micro: {micro}")

    sufix = '/'.join(sorted([f"{c[0]}/{c[1]}" for c in clf_set]))
    output_dir = f"{base_output}/{dataset}/{N_FOLDS}_folds/attention/{meta_layer_name}/use_mf_{use_mf}/{sufix}/{fold}"
    
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/scoring.json", 'w') as fd:
        scoring = {
            "macro": macro,
            "micro": micro
        }
        json.dump(scoring, fd)
