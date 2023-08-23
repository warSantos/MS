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

from src.attn_mfs import load_embeddings, load_mfs
from src.utils import load_labels_fold, load_probs_fold, get_attention_labels


class ExpandedLoss(nn.Module):

    def __init__(self):
        super(ExpandedLoss, self).__init__()

    def forward(self, output, target):

        n_docs = target.shape[0]
        """
        expanded_output = output.repeat(1, target.shape[1]).reshape(target.shape[0],
                                                                    target.shape[1],
                                                                    -1)
        """
        """
        # Comparing the attention weights with the matrix of good combinations.
        diff = torch.abs(target - expanded_output)
        # Lines with only zeros are hits.
        contrast = diff == mask
        # For document verify if is there any with only zeros.
        result = torch.all(contrast, dim=2)
        # Getting the documents with no zero lines.
        result = ~torch.any(result, dim=1)
        # Return the number of documents that the attention weigts
        # found no equivalent combination.
        return torch.sum(result)
        """
        #diff = torch.abs(target - expanded_output)
        diff = torch.abs(target - output)
        # Lines with only zeros are hits.
        zero_lines = torch.min(torch.sum(diff, dim=2), dim=1).values
        # For document verify if is there any with only zeros.
        #loss = torch.sum(zero_lines > 0)
        return torch.sum(zero_lines)


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
        self.query_dim = q_train.shape[2]

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
                              shuffle=False,
                              num_workers=self.workers)

        elif set_name == "test":
            return DataLoader(CrossAttDataset(self.q_test,
                                              self.v_test,
                                              self.y_test),
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.workers)
        else:

            raise ValueError(
                f"Value: {set_name} for parameter 'set_name' is invalid. Valid options are: 'train', 'val' and 'test'")


class CrossAttention(pl.LightningModule):

    def __init__(self,
                 query_dim: int,
                 clfs_dim: int,
                 n_labels: int,
                 num_heads: int,
                 dropout: float,
                 apply_upper_bound: bool = False):

        super(CrossAttention, self).__init__()

        self.apply_upper_bound = apply_upper_bound
        print(f"Applying upper bound: {apply_upper_bound}")
        self.is_train = False

        self.query_fnn = nn.Linear(query_dim, query_dim)
        # self.key_fnn = nn.Linear(n_labels, n_labels)
        self.key_fnn = nn.Linear(clfs_dim, clfs_dim)
        # self.value_fnn = nn.Linear(n_labels, n_labels)
        self.value_fnn = nn.Linear(clfs_dim, clfs_dim)

        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=clfs_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout)

    def forward(self, q, v):

        query = self.query_fnn(q.transpose(dim0=1, dim1=2))
        key = self.key_fnn(v.transpose(dim0=1, dim1=2))
        value = self.value_fnn(v.transpose(dim0=1, dim1=2))

        # Computing attention weights.
        attn_out, wm = self.multihead_attn(query,
                                           key,
                                           value)
        return attn_out, wm


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
                                        n_labels,
                                        num_heads,
                                        dropout,
                                        apply_upper_bound)

        self.linear = nn.Linear(n_labels * clfs_dim, n_labels)

        self.loss = nn.CrossEntropyLoss()
        self.f1 = F1Score(task="multiclass",
                          num_classes=n_labels, average="macro")

    def forward(self, query, probas):

        attn_out, _ = self.cross_att(query, probas)
        weights = torch.mean(attn_out, dim=1)
        weights = F.softmax(weights, dim=1)
        wp = probas * weights.unsqueeze(-1)
        return self.linear(torch.flatten(wp, start_dim=1))

    def training_step(self, batch, batch_idx):

        query, value, y = batch["query"], batch["value"], batch["y"]
        y_hat = self(query, value)
        loss = self.loss(y_hat, y)
        self.log_dict({"train_loss": loss},
                      prog_bar=True,
                      on_epoch=True,
                      on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):

        query, value, y = batch["query"], batch["value"], batch["y"]
        y_hat = self(query, value)
        loss = self.loss(y_hat, y)
        macro = self.f1(torch.argmax(y_hat, dim=1), y)
        self.log_dict({"val_loss": loss, "macro": macro},
                      prog_bar=True,
                      on_epoch=True,
                      on_step=False)
        return loss

    def predict_step(self, batch, batch_idx):

        query, value = batch["query"], batch["value"]
        y_hat = self(query, value)
        return y_hat.tolist()

    def get_stoper(self,
                   monitor="val_loss",
                   min_delta=0.0,
                   epochs=64,
                   verbose=False,
                   mode='min'):

        # patience = int(2 ** (np.log2(epochs) - 3))
        patience = 5
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


class DualCrossAttDataset(Dataset):

    def __init__(self, query, value, y, cross_y) -> None:
        super().__init__()

        self.query = query
        self.value = value
        self.y = y
        self.cross_y = cross_y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "query": torch.tensor(self.query[idx], dtype=torch.float32),
            "value": torch.tensor(self.value[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx]),
            "cross_y": torch.tensor(self.cross_y[idx], dtype=torch.float32)
        }


class DualCrossAttDataHandler:

    def __init__(self,
                 data_source: str,
                 dataset: str,
                 clfs: list,
                 fold: int,
                 n_folds: int,
                 batch_size: int = 64,
                 workers: int = 6,
                 q_type: str = "probas-based"):

        self.n_clfs = len(clfs)
        self.batch_size = batch_size
        self.workers = workers

        if q_type == "probas-based":
            query_train, q_test = load_mfs(data_source, dataset, clfs, fold)
        else:
            query_train, q_test = load_embeddings(
                data_source, dataset, fold, n_folds, True, self.n_clfs)

        value_train, v_test = load_probs_fold(
            data_source, dataset, clfs, fold, n_folds, True)

        label_train, y_test = load_labels_fold(
            data_source, dataset, fold, n_folds)

        cross_label_train, cross_y_test = get_attention_labels(
            data_source, dataset, clfs, fold, n_folds, True)

        splits = train_test_split(query_train,
                                  value_train,
                                  label_train,
                                  cross_label_train,
                                  stratify=label_train)
        q_train, q_val, v_train, v_val, y_train, y_val, cross_y_train, cross_y_val = splits

        self.q_train = q_train
        self.q_val = q_val
        self.q_test = q_test

        self.v_train = v_train
        self.v_val = v_val
        self.v_test = v_test

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.cross_y_train = cross_y_train
        self.cross_y_val = cross_y_val
        self.cross_y_test = cross_y_test

        self.n_labels = v_train[0].shape[1]
        self.query_dim = q_train.shape[2]

    def data_prepare(self, set_name: str):

        if set_name == "train":
            return DataLoader(DualCrossAttDataset(self.q_train,
                                                  self.v_train,
                                                  self.y_train,
                                                  self.cross_y_train),
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.workers)

        elif set_name == "val":
            return DataLoader(DualCrossAttDataset(self.q_val,
                                                  self.v_val,
                                                  self.y_val,
                                                  self.cross_y_val),
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.workers)

        elif set_name == "test":
            return DataLoader(DualCrossAttDataset(self.q_test,
                                                  self.v_test,
                                                  self.y_test,
                                                  self.cross_y_test),
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.workers)
        else:

            raise ValueError(
                f"Value: {set_name} for parameter 'set_name' is invalid. Valid options are: 'train', 'val' and 'test'")


class DualCrossAttMetaLayer(pl.LightningModule):

    def __init__(self,
                 n_labels: int,
                 query_dim: int,
                 clfs_dim: int,
                 num_heads: int,
                 dropout: float,
                 apply_upper_bound: bool = False) -> None:

        super(DualCrossAttMetaLayer, self).__init__()
        self.automatic_optimization = False
        self.cross_att = CrossAttention(query_dim,
                                        clfs_dim,
                                        n_labels,
                                        num_heads,
                                        dropout,
                                        apply_upper_bound)

        self.linear = nn.Linear(n_labels * clfs_dim, n_labels)

        self.loss = nn.CrossEntropyLoss()
        self.cross_loss = nn.MSELoss()
        self.f1 = F1Score(task="multiclass",
                          num_classes=n_labels, average="macro")

    def forward(self, query, probas):

        attn_out, wm = self.cross_att(query, probas)
        weights = torch.mean(attn_out, dim=1)
        weights = F.softmax(weights, dim=1)
        wp = probas * weights.unsqueeze(-1)
        return self.linear(torch.flatten(wp, start_dim=1))

    def training_step(self, batch, batch_idx):

        query, value, y, cross_y = batch["query"], batch["value"], batch["y"], batch["cross_y"]

        # Getting optmizers.
        cross_opt, mlp_opt = self.optimizers()

        # Computing cross attention loss.
        attn_out, wm = self.cross_att(query, value)
        cross_opt.zero_grad()
        attn_loss = self.cross_loss(
            F.softmax(torch.mean(attn_out, dim=1), dim=1), cross_y)
        self.manual_backward(attn_loss)
        cross_opt.step()

        # Computing MLP loss.
        y_hat = self(query, value)
        mlp_opt.zero_grad()
        loss = self.loss(y_hat, y)
        self.manual_backward(loss)
        mlp_opt.step()

        reporte = {
            "AttLoss": attn_loss,
            "CRELoss": loss,
            "f1_train": self.f1(torch.argmax(y_hat, dim=-1), y)}
        self.log_dict(reporte, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):

        query, value, y, cross_y = batch["query"], batch["value"], batch["y"], batch["cross_y"]
        attn_out, wm = self.cross_att(query, value)
        attn_loss = self.cross_loss(
            F.softmax(torch.mean(attn_out, dim=1), dim=1), cross_y)
        y_hat = self(query, value)
        loss = self.loss(y_hat, y)
        macro = self.f1(torch.argmax(y_hat, dim=1), y)
        self.log_dict({"ValCRELoss": loss, "ValAttLoss": attn_loss, "f1_val": macro},
                      prog_bar=True,
                      on_epoch=True,
                      on_step=False)
        return loss

    def predict_step(self, batch, batch_idx):

        query, value = batch["query"], batch["value"]
        y_hat = self(query, value)
        return y_hat.tolist()

    def get_stoper(self,
                   monitor="ValCRELoss",
                   min_delta=0.0,
                   epochs=64,
                   verbose=False,
                   mode='min'):

        # patience = int(2 ** (np.log2(epochs) - 3))
        patience = 5
        es = EarlyStopping(monitor=monitor,
                           min_delta=min_delta,
                           patience=patience,
                           verbose=verbose,
                           mode=mode)
        return es

    def configure_optimizers(self):

        opt = AdamW(self.linear.parameters(), lr=1e-2, weight_decay=1e-4)
        sch = ReduceLROnPlateau(opt, "min", patience=1, factor=0.02)
        mlp_opt = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "ValCRELoss"
            }
        }

        cross_opt = AdamW(self.cross_att.parameters(),
                          lr=1e-2, weight_decay=1e-4)
        cross_sch = ReduceLROnPlateau(opt, "min", patience=1, factor=0.02)
        attn_opt = {
            "optimizer": cross_opt,
            "lr_scheduler": {
                "scheduler": cross_sch,
                "monitor": "ValAttLoss"
            }
        }

        return attn_opt, mlp_opt


class ExpandedCrossAttDataset(Dataset):

    def __init__(self, query, value, y, cross_y) -> None:
        super().__init__()

        self.query = query
        self.value = value
        self.y = y
        self.cross_y = cross_y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "query": torch.tensor(self.query[idx], dtype=torch.float32),
            "value": torch.tensor(self.value[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx]),
            "cross_y": torch.tensor(self.cross_y[idx], dtype=torch.float32)
        }


class ExpandedCrossAttDataHandler:

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

        cross_label_train, cross_y_test = get_attention_labels(
            data_source, dataset, clfs, fold, n_folds, True)

        splits = train_test_split(query_train,
                                  value_train,
                                  label_train,
                                  cross_label_train,
                                  stratify=label_train,
                                  test_size=0.20)

        q_train, q_val, v_train, v_val, y_train, y_val, cross_y_train, cross_y_val = splits

        self.q_train = q_train
        self.q_val = q_val
        self.q_test = q_test

        self.v_train = np.array(v_train)
        self.v_val = np.array(v_val)
        self.v_test = np.array(v_test)

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.cross_y_train = np.array(cross_y_train)
        self.cross_y_val = np.array(cross_y_val)
        self.cross_y_test = np.array(cross_y_test)

        self.n_labels = v_train[0].shape[1]
        self.query_dim = q_train.shape[2]

    def to_dataloader(self, q, v, y, cross_y, shuffle):

        return DataLoader(ExpandedCrossAttDataset(q, v, y, cross_y),
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          num_workers=self.workers)

    def data_prepare(self, set_name: str):

        if set_name == "train":
            return DataLoader(ExpandedCrossAttDataset(self.q_train,
                                                      self.v_train,
                                                      self.y_train,
                                                      self.cross_y_train),
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.workers)

        elif set_name == "val":
            return DataLoader(ExpandedCrossAttDataset(self.q_val,
                                                      self.v_val,
                                                      self.y_val,
                                                      self.cross_y_val),
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.workers)

        elif set_name == "test":
            return DataLoader(ExpandedCrossAttDataset(self.q_test,
                                                      self.v_test,
                                                      self.y_test,
                                                      self.cross_y_test),
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.workers)
        else:

            raise ValueError(
                f"Value: {set_name} for parameter 'set_name' is invalid. Valid options are: 'train', 'val' and 'test'")


class ExpandedCrossAttMetaLayer(pl.LightningModule):

    def __init__(self,
                 n_labels: int,
                 query_dim: int,
                 clfs_dim: int,
                 num_heads: int,
                 dropout: float,
                 apply_upper_bound: bool = False) -> None:

        super(ExpandedCrossAttMetaLayer, self).__init__()
        self.automatic_optimization = False
        self.cross_att = CrossAttention(query_dim,
                                        clfs_dim,
                                        n_labels,
                                        num_heads,
                                        dropout,
                                        apply_upper_bound)

        self.linear = nn.Linear(n_labels * clfs_dim, n_labels)
        #self.register_buffer("mask", torch.zeros(clfs_dim, requires_grad=True))

        self.loss = nn.CrossEntropyLoss()
        self.cross_loss = ExpandedLoss()
        self.f1 = F1Score(task="multiclass",
                          num_classes=n_labels, average="macro")

    def forward(self, query, probas):

        attn_out, wm = self.cross_att(query, probas)
        weights = torch.mean(attn_out, dim=1)
        weights = F.softmax(weights, dim=1)
        wp = probas * weights.unsqueeze(-1)
        return self.linear(torch.flatten(wp, start_dim=1))

    def training_step(self, batch, batch_idx):

        query, value, y, cross_y = batch["query"], batch["value"], batch["y"], batch["cross_y"]

        # Getting optmizers.
        cross_opt, mlp_opt = self.optimizers()

        # Computing cross attention loss.
        attn_out, wm = self.cross_att(query, value)
        cross_opt.zero_grad()
        attn_out = F.softmax(torch.mean(attn_out, dim=1), dim=1)
        attn_out = attn_out.repeat(1, cross_y.shape[1]).reshape(cross_y.shape[0],
                                                                    cross_y.shape[1],
                                                                    -1)
                             
        attn_loss = self.cross_loss(attn_out, cross_y)
        self.manual_backward(attn_loss)
        cross_opt.step()

        # Computing MLP loss.
        y_hat = self(query, value)
        mlp_opt.zero_grad()
        loss = self.loss(y_hat, y)
        self.manual_backward(loss)
        mlp_opt.step()

        reporte = {
            "AttLoss": attn_loss,
            "CRELoss": loss,
            "f1_train": self.f1(torch.argmax(y_hat, dim=-1), y)}
        self.log_dict(reporte, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):

        query, value, y, cross_y = batch["query"], batch["value"], batch["y"], batch["cross_y"]
        attn_out, wm = self.cross_att(query, value)
        y_hat = self(query, value)
        attn_out = F.softmax(torch.mean(attn_out, dim=1), dim=1)
        attn_out = attn_out.repeat(1, cross_y.shape[1]).reshape(cross_y.shape[0],
                                                                    cross_y.shape[1],
                                                                    -1)
        attn_loss = self.cross_loss(attn_out, cross_y)
        loss = self.loss(y_hat, y)
        macro = self.f1(torch.argmax(y_hat, dim=1), y)
        self.log_dict({"ValCRELoss": loss, "ValAttLoss": attn_loss, "f1_val": macro},
                      prog_bar=True,
                      on_epoch=True,
                      on_step=False)
        return loss

    def predict_step(self, batch, batch_idx):

        query, value = batch["query"], batch["value"]
        y_hat = self(query, value)
        return y_hat.tolist()

    def get_stoper(self,
                   monitor="ValCRELoss",
                   min_delta=0.0,
                   epochs=64,
                   verbose=False,
                   mode='min'):

        # patience = int(2 ** (np.log2(epochs) - 3))
        patience = 5
        es = EarlyStopping(monitor=monitor,
                           min_delta=min_delta,
                           patience=patience,
                           verbose=verbose,
                           mode=mode)
        return es

    def configure_optimizers(self):

        opt = AdamW(self.linear.parameters(), lr=1e-2, weight_decay=1e-4)
        sch = ReduceLROnPlateau(opt, "min", patience=1, factor=0.02)
        mlp_opt = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "ValCRELoss"
            }
        }

        cross_opt = AdamW(self.cross_att.parameters(),
                          lr=1e-2, weight_decay=1e-4)
        cross_sch = ReduceLROnPlateau(opt, "min", patience=1, factor=0.02)
        attn_opt = {
            "optimizer": cross_opt,
            "lr_scheduler": {
                "scheduler": cross_sch,
                "monitor": "ValAttLoss"
            }
        }

        return attn_opt, mlp_opt
