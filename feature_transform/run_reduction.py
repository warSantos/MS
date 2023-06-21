import os
import json
from itertools import product
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import Normalizer

import torch
torch.manual_seed(42)

from data_loader import AutoEncoderDataset
from autoencoder import AutoEncoder

from torch.utils.data import DataLoader

with open("data/settings.json", 'r') as fd:
    settings = json.load(fd)

def get_data_loaders(X_train: np.ndarray,
                     X_val: np.ndarray,
                     X_test: np.ndarray,
                     batch_size: int,
                     num_workers: int):

    train_data_loader = DataLoader(dataset=AutoEncoderDataset(X_train, X_train),
                                    num_workers=num_workers,
                                    batch_size=batch_size)
    val_data_loader = DataLoader(dataset=AutoEncoderDataset(X_val, X_val),
                                num_workers=num_workers,
                                batch_size=batch_size)
    test_data_loader = DataLoader(dataset=AutoEncoderDataset(X_test, X_test),
                                num_workers=num_workers,
                                batch_size=batch_size)
    
    return train_data_loader, val_data_loader, test_data_loader

def transform_train(
        X: np.ndarray,
        n_folds: int,
        epochs: int,
        batch_size: int,
        num_workers: int
):
    kf = KFold(n_splits=n_folds)
    align_idx = np.arange(X.shape[0])
    idxs_list = []
    features_list = []

    for fold, (train_idxs, test_idxs) in enumerate(kf.split(X)):
        
        X_train, X_test = X[train_idxs], X[test_idxs]
        X_train, X_val = train_test_split(X_train, test_size=0.1)
        idxs_list.append(align_idx[test_idxs])

        train_dl, val_dl, test_dl = get_data_loaders(X_train,
                         X_val,
                         X_test,
                         batch_size,
                         num_workers)
        
        auto_encoder = AutoEncoder(X_train.shape[1], 1000, 20, 0.3)
        trainer = auto_encoder.fit([train_dl, val_dl], epochs)
        features_list.append(np.vstack(trainer.predict(auto_encoder, dataloaders=test_dl)))
        
    sort = np.hstack(idxs_list).argsort()
    features_list = np.vstack(features_list)
    return features_list[sort]


DATASETS = settings["DATASETS"]
epochs = settings["EPOCHS"]
data_source = settings["DATA_SOURCE"]
batch_size = settings["BATCH_SIZE"]
num_workers = settings["NUM_WORKERS"]
n_train_splits = settings["N_TRAIN_SPLITS"]
output_dim = settings["OUTPUT_DIM"]

for dataset_setup in DATASETS:
    dataset, n_folds = dataset_setup
    for fold in np.arange(n_folds):

        if data_source.find("meta_features") >-1:
            source_dir = f"{data_source}/{fold}/{dataset}"
        elif data_source.find("representations") > -1:
            source_dir = f"{data_source}/{dataset}/{n_folds}_folds/bert/{fold}"
        else:
            raise("Data Source option not valid.")

        loader_train = np.load(f"{source_dir}/train.npz")
        X_train = loader_train["X_train"]
        loader_test = np.load(f"{source_dir}/test.npz")
        X_test = loader_test["X_test"]
        
        normalizer = Normalizer()
        normalizer.fit(X_train)
        X_train = normalizer.transform(X_train)
        X_test = normalizer.transform(X_test)

        """
        # Doing train reduction.
        reducted_train = transform_train(X_train,
                                         n_train_splits,
                                         epochs,
                                         batch_size,
                                         num_workers)

        """
        # Doing test reduction.

        big_train = DataLoader(dataset=AutoEncoderDataset(X_train, X_train),
                                num_workers=num_workers,
                                batch_size=batch_size)
        
        X_train, X_val = train_test_split(X_train, test_size=0.1)
        
        train_dl, val_dl, test_dl = get_data_loaders(X_train,
                                                     X_val,
                                                     X_test,
                                                     batch_size,
                                                     num_workers)

        auto_encoder = AutoEncoder(X_train.shape[1], 1000, output_dim, 0.3)
        trainer = auto_encoder.fit([train_dl, val_dl], epochs)

        reducted_test = np.vstack(trainer.predict(auto_encoder, dataloaders=test_dl))
        reducted_train = np.vstack(trainer.predict(auto_encoder, dataloaders=big_train))


        if data_source.find("meta_features") >-1:
            output_dir = f"{data_source}_{output_dim}/{fold}/{dataset}"      
        elif data_source.find("representations") > -1:
            output_dir = f"{data_source}/{dataset}/{n_folds}_folds/bert_{output_dim}/{fold}"
        else:
            raise("Data Source option not valid.")

        os.makedirs(output_dir, exist_ok=True)

        np.savez(f"{output_dir}/train", X_train=reducted_train, y_train=loader_train["y_train"])
        np.savez(f"{output_dir}/test", X_test=reducted_test, y_test=loader_test["y_test"])