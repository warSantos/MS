import os
import json
from itertools import product
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from torchmetrics import F1Score

from sklearn.metrics import f1_score

from utils import (transform_probas,
                   load_probs_fold,
                   load_upper_bound,
                   get_attention_labels,
                   load_labels_fold)

from attn_mfs import get_error_est_mfs

class StackingDataset(Dataset):

    def __init__(self, X, y, attn_x, attn_y) -> None:
        super().__init__()

        self.X = X
        self.y = y
        self.attn_x = attn_x
        self.attn_y = attn_y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.X[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx]),
            "attn_x": torch.tensor(self.attn_x[idx], dtype=torch.float32),
            "attn_y": torch.tensor(self.attn_y[idx], dtype=torch.float32)
        }


class DiffMatrixLoss(nn.Module):

    def __init__(self):
        super(DiffMatrixLoss, self).__init__()

    def forward(self, output, target):

        return torch.sum(torch.abs(output - target))


class MultiHeadAttentionEncoder(pl.LightningModule):

    def __init__(self, hidden, num_heads, dropout):
        super(MultiHeadAttentionEncoder, self).__init__()

        self.key_fnn = nn.Linear(hidden, hidden)
        self.query_fnn = nn.Linear(hidden, hidden)
        self.value_fnn = nn.Linear(hidden, hidden)
        self.multihead_att = torch.nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout)

    def forward(self, x, attn_x, attn_y):

        key = self.key_fnn(attn_x)
        query = self.query_fnn(attn_x)
        value = self.value_fnn(attn_x)

        attn_out, attn_weights = self.multihead_att(query, key, value)
        learned_w = torch.mean(attn_weights, dim=1)
        
        
        weighted_probs = x * learned_w.unsqueeze(-1)
        #weighted_probs = x * upper.unsqueeze(-1)
        
        return torch.flatten(weighted_probs, start_dim=1), attn_weights


class MetaLayer(pl.LightningModule):

    def __init__(self,
                 classes_number: int,
                 clfs_number: int,
                 attention_size: int,
                 num_heads: int,
                 dropout: float):

        super().__init__()

        #self.automatic_optimization = False

        self.attn_encoder = MultiHeadAttentionEncoder(
            attention_size,
            num_heads,
            dropout)

        self.attn_loss = DiffMatrixLoss()

        self.output_layer = nn.Linear(
            classes_number * clfs_number, classes_number)
        self.mlp_loss = nn.CrossEntropyLoss()

        self.softmax = nn.Softmax(dim=-1)
        self.f1 = F1Score(task="multiclass", num_classes=classes_number, average="macro")

    def configure_optimizers(self):


        opt = AdamW(self.parameters(), lr=1e-2)
        #sch = CosineAnnealingLR(opt, T_max=60)
        sch = ReduceLROnPlateau(opt, "max", patience=1, factor=0.02)
        
        return {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": "f1_val"
        }

    def forward(self, x, attn_x, attn_y):

        attn_out, _ = self.attn_encoder(x, attn_x, attn_y)
        return self.softmax(self.output_layer(attn_out))

    def training_step(self, batch):

        x, y, attn_x, attn_y = batch["x"], batch["y"], batch["attn_x"], batch["attn_y"]
        
        y_hat = self(x, attn_x, attn_y)
        
        loss = self.mlp_loss(y_hat, y)
        
        self.log_dict(
            {
                "f1_train": self.f1(torch.argmax(y_hat, dim=-1), y)
            },
            on_epoch=True,
            on_step=True,
            prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):

        x, y, attn_x, attn_y = batch["x"], batch["y"], batch["attn_x"], batch["attn_y"]
        y_hat = self(x, attn_x, attn_y)
        loss = self.mlp_loss(y_hat, y)
        self.log_dict(
            {
                "f1_val": self.f1(torch.argmax(y_hat, dim=-1), y),
                "cross_loss_val": loss
            },
            on_epoch=True,
            on_step=True,
            prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):

        x, _, attn_x, attn_y = batch["x"], batch["y"], batch["attn_x"], batch["attn_y"]
        y_hat = self(x, attn_x, attn_y)
        preds = torch.argmax(y_hat, dim=-1)
        return preds.cpu().numpy()


CLFS = [["bert", "temperature_scaling"],
        ["xlnet", "temperature_scaling"],
        ["ktmk", "isotonic"],
        ["ktr", "isotonic"],
        ["lstmk", "isotonic"],
        ["lstr", "isotonic"],
        ["ltr", "isotonic"]]

mf_input_source = "/home/welton/data/meta_features/error_estimation"

DATASETS = ["acm"]

iters = product(DATASETS, np.arange(10))

probs_only = True

for dataset, fold in iters:

    labels_train, y_test = load_labels_fold(dataset, fold)
    probas_train, X_test = load_probs_fold(dataset, CLFS, fold)

    uppers = load_upper_bound(dataset, CLFS, fold)
    
    attn_y_train = get_attention_labels(CLFS, uppers, "train")
    attn_y_test = get_attention_labels(CLFS, uppers, "test")

    X_train, X_val, y_train, y_val, attn_y_train, attn_y_val = train_test_split(probas_train,
                                                                                labels_train,
                                                                                attn_y_train,
                                                                                stratify=labels_train,
                                                                                test_size=0.1,
                                                                                random_state=42)
    
    if probs_only:
        attn_x_train = X_train
        attn_x_val = X_val
        attn_x_test = X_test
    else:
        attn_x_train, attn_x_test = get_error_est_mfs(mf_input_source,
                                                  dataset,
                                                  CLFS,
                                                  fold,
                                                  10)
        
        attn_x_train, attn_x_val = train_test_split(attn_x_train,
                                                    stratify=labels_train,
                                                    test_size=0.1,
                                                    random_state=42)

        
    
    clfs_number = len(CLFS)
    classes_number = X_train[0].shape[1]
    max_epochs = 80
    batch_size = 128

    train_loader = DataLoader(dataset=StackingDataset(
        X_train, y_train, attn_x_train, attn_y_train), batch_size=batch_size, num_workers=16)
    val_loader = DataLoader(dataset=StackingDataset(
        X_val, y_val, attn_x_val, attn_y_val), batch_size=batch_size, num_workers=16)
    test_loader = DataLoader(dataset=StackingDataset(
        X_test, y_test, attn_x_test, attn_y_test), batch_size=batch_size, num_workers=16)

    meta_layer = MetaLayer(classes_number, clfs_number, attn_x_train[0].shape[1], 1, 0.1)
    
    estop = EarlyStopping(monitor="f1_val", min_delta=0.0, patience=10, verbose=False, mode='max')
    trainer = pl.Trainer(accelerator='gpu', 
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[estop])
    
    trainer.fit(meta_layer, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    preds = trainer.predict(meta_layer, dataloaders=test_loader)
    preds = np.hstack(preds)

    macro = f1_score(y_test, preds, average="macro")
    micro = f1_score(y_test, preds, average="micro")

    print(f"Macro: {macro} - Micro: {micro}")

    
    output_dir = f"reports/{dataset}"
    sufix = '/'.join(sorted([f"{c[0]}/{c[1]}" for c in CLFS]))
    output_dir = f"{output_dir}/{sufix}/{fold}"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/scoring.json", 'w') as fd:
        scoring = {
            "macro": macro,
            "micro": micro
        }
        json.dump(scoring, fd)
