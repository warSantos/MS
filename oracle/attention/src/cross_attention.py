from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics import F1Score

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from src.custom_loss import AttnLogLoss, ExpandedLoss

from src.attn_mfs import load_embeddings
from src.utils import load_labels_fold, load_probs_fold


class CrossAttDataset(Dataset):

    def __init__(self, query, value, y) -> None:
        super().__init__()

        self.query = query
        self.value = value
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "query": torch.tensor(self.query[idx], dtype=torch.float32),
            "value": torch.tensor(self.value[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx])
        }


class CrossAttDataHandler:

    def __init__(self,
                 data_source: str,
                 dataset: str,
                 clfs: list,
                 fold: int,
                 n_folds: int,
                 batch_size: int = 64,
                 workers: int = 6):

        self.n_clfs = len(clfs)
        self.batch_size = batch_size
        self.workers = workers

        query_train, q_test = load_embeddings(
            data_source, dataset, fold, n_folds, True, self.n_clfs)
        value_train, v_test = load_probs_fold(
            data_source, dataset, clfs, fold, n_folds, True)
        label_train, y_test = load_labels_fold(
            data_source, dataset, fold, n_folds)

        q_train, q_val, v_train, v_val, y_train, y_val = train_test_split(query_train,
                                                                          value_train,
                                                                          label_train,
                                                                          stratify=label_train)
        self.q_train = q_train
        self.q_val = q_val
        self.q_test = q_test

        self.v_train = v_train
        self.v_val = v_val
        self.v_test = v_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.n_labels = v_train[0].shape[1]
        self.query_dim = q_train.shape[1]

    def data_prepare(self, set_name: str):

        if set_name == "train":
            return DataLoader(CrossAttDataset(self.q_train,
                                              self.v_train,
                                              self.y_train),
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.workers)

        elif set_name == "val":
            return DataLoader(CrossAttDataset(self.q_val,
                                              self.v_val,
                                              self.y_val),
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.workers)

        elif set_name == "test":
            return DataLoader(CrossAttDataset(self.q_test,
                                              self.v_test,
                                              self.y_test),
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.workers)
        else:

            raise ValueError(f"Value: {set_name} for parameter 'set_name' is invalid. Valid options are: 'train', 'val' and 'test'")


class CrossAttention(pl.LightningModule):

    def __init__(self,
                 query_dim: int,
                 clfs_dim: int,
                 num_heads: int,
                 dropout: float,
                 apply_upper_bound: bool = False):

        super(CrossAttention, self).__init__()

        self.apply_upper_bound = apply_upper_bound
        print(f"Applying upper bound: {apply_upper_bound}")
        self.is_train = False

        self.query_fnn = nn.Linear(query_dim, query_dim)
        self.key_fnn = nn.Linear(clfs_dim, clfs_dim)
        self.value_fnn = nn.Linear(clfs_dim, clfs_dim)

        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=clfs_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout)

    def forward(self, query, value):

        query = self.query_fnn(query)
        key = self.key_fnn(value)
        value = self.value_fnn(value)

        # Computing attention weights.
        attn_out, _ = self.multihead_attn(query.transpose(dim0=1, dim1=2),
                                          key.transpose(dim0=1, dim1=2),
                                          value.transpose(dim0=1, dim1=2))

        weights = torch.mean(attn_out, dim=1)
        weights = F.softmax(weights, dim=1)
        return weights.transpose(dim0=1, dim1=2)


class CrossAttMetaLayer(pl.LightningModule):

    def __init__(self,
                 n_labels: int,
                 query_dim: int,
                 clfs_dim: int,
                 num_heads: int,
                 dropout: float,
                 apply_upper_bound: bool = False) -> None:

        super(CrossAttMetaLayer, self).__init__()

        self.cross_att = CrossAttention(query_dim,
                                        clfs_dim,
                                        num_heads,
                                        dropout,
                                        apply_upper_bound)

        self.linear = nn.Linear(n_labels * clfs_dim, n_labels)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, query, probas) -> Any:

        weights = self.cross_att(query, probas)
        wp = probas * weights.unsqueeze(-1)
        return self.linear(torch.flatten(wp, start_dim=1))

    def training_step(self, batch, batch_idx):

        query, value, y = batch["query"], batch["value"], batch["y"]
        y_hat = self(query, value)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):

        query, value, y = batch["query"], batch["value"], batch["y"]
        y_hat = self(query, value)
        loss = self.loss(y_hat, y)
        self.log_dict({"val_loss": loss},
                      prog_bar=True,
                      on_epoch=True,
                      on_step=False)
        return loss

    def predict(self, batch, batch_idx):

        query, value = batch["query"], batch["value"]
        y_hat = self(query, value)
        return y_hat.tolist()

    def get_stoper(self,
                   monitor="val_loss",
                   min_delta=0.0,
                   epochs=64,
                   verbose=False,
                   mode='min'):

        patience = int(2 ** (np.log2(epochs) - 2))
        es = EarlyStopping(monitor=monitor,
                           min_delta=min_delta,
                           patience=patience,
                           verbose=verbose,
                           mode=mode)
        return es

    def configure_optimizers(self):

        opt = AdamW(self.parameters(), lr=1e-2)
        sch = ReduceLROnPlateau(opt, "min", patience=1, factor=0.02)

        return {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": "val_loss"
        }
