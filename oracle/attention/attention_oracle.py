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
from torchmetrics import F1Score

from sklearn.metrics import f1_score, precision_score, recall_score


def transform_probas(probas_set: list):

    features = []
    for doc_idx in np.arange(probas_set[0].shape[0]):
        features.append(
            np.vstack([probas_set[clf_idx][doc_idx]
                      for clf_idx in np.arange(len(probas_set))])
        )

    return features


def load_probs_fold(dataset: str, clfs: list, fold: int):

    clfs_probs_train = []
    clfs_probs_test = []
    for clf, proba_type in clfs:
        probs_dir = f"/home/welton/data/{proba_type}/split_10/{dataset}/10_folds/{clf}/{fold}/"
        clfs_probs_train.append(np.load(f"{probs_dir}/train.npz")["X_train"])
        clfs_probs_test.append(np.load(f"{probs_dir}/test.npz")["X_test"])

    return transform_probas(clfs_probs_train), transform_probas(clfs_probs_test)


def load_labels_fold(dataset: str, clfs: list, fold: int):

    train_list = []
    test_list = []
    for clf, proba_type in clfs:
        proba_dir = f"/home/welton/data/oracle/upper_bound/{proba_type}/{dataset}/10_folds/{clf}/{fold}"
        train_list.append(np.load(f"{proba_dir}/train.npz")['y'])
        test_list.append(np.load(f"{proba_dir}/test.npz")['y'])

    return np.vstack(train_list).T.astype("float32"), np.vstack(test_list).T.astype("float32")


class StackingDataset(Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.X[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx])
        }

class RoundActivation(torch.nn.Module):
    def forward(self, x):
        return torch.round(x)

class MHAE(pl.LightningModule):

    def __init__(self, classes_number: int, 
                 clfs_number: int, 
                 num_heads: int, 
                 dropout: float):
        
        super(MHAE, self).__init__()

        self.key_fnn = nn.Linear(classes_number, classes_number)
        self.query_fnn = nn.Linear(classes_number, classes_number)
        self.value_fnn = nn.Linear(classes_number, classes_number)

        self.multihead_att = torch.nn.MultiheadAttention(
            embed_dim=classes_number,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        self.output_layer = nn.Linear(classes_number * clfs_number, clfs_number)
        
        self.loss = nn.MSELoss()
        self.round_activation = RoundActivation()


    def forward(self, x):

        # Computing attention.
        key = self.key_fnn(x)
        query = self.query_fnn(x)
        value = self.value_fnn(x)
        attn_output, _ = self.multihead_att(query, key, value)
        out = torch.flatten(attn_output, start_dim=1)

        # Reducing the probabibilitie dimension to number of classifiers.
        out = self.output_layer(out)
        return self.round_activation(out)

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch):
        x, y = batch["x"], batch["y"]

        # forward pass.
        y_hat = self(x)

        # computing loss.
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch["x"], batch["y"]
        # forward pass.
        y_hat = self(x)
        self.log_dict(
            {"val_loss": self.loss(y_hat, y)}, prog_bar=True)

    """
    def validation_epoch_end(self, outs):
        self.loss(outs)
    """

    def test_step(self, batch, batch_idx):

        x, y = batch["x"], batch["y"]
        # forward pass.
        y_hat = self(x)
        self.log_dict({"MSE": self.loss(y_hat, y)}, prog_bar=True)
        return y_hat.cpu().numpy()

    def predict_step(self, batch, batch_idx):

        x, y = batch["x"], batch["y"]
        # forward pass.
        y_hat = self(x)
        return y_hat.cpu().numpy()


CLFS = [["bert", "normal_probas"],
        ["xlnet", "normal_probas"],
        ["ktmk", "normal_probas"]]

DATASETS = ["acm", "20ng"]

iters = product(DATASETS, np.arange(10))

for dataset, fold in iters:

    labels_train, y_test = load_labels_fold(dataset, CLFS, fold)

    probas_train, X_test = load_probs_fold(dataset, CLFS, fold)
    X_train, X_val, y_train, y_val = train_test_split(
        probas_train, labels_train, stratify=labels_train, test_size=0.1, random_state=42)

    clfs_number = len(CLFS)
    classes_number = X_train[0].shape[1]
    max_epochs = 10
    batch_size = 128

    train_loader = DataLoader(dataset=StackingDataset(
        X_train, y_train), batch_size=batch_size, num_workers=16)
    val_loader = DataLoader(dataset=StackingDataset(
        X_val, y_val), batch_size=batch_size, num_workers=16)
    test_loader = DataLoader(dataset=StackingDataset(
        X_test, y_test), batch_size=batch_size, num_workers=16)

    meta_layer = MHAE(classes_number, clfs_number, 1, 0.1)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs)
    trainer.fit(meta_layer, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    preds = trainer.predict(meta_layer, dataloaders=test_loader)
    
    preds = np.vstack(preds)

    bert_preds = preds[:, 0]
    bert_true = y_test[:, 0]

    macro = f1_score(bert_true, bert_preds, average='binary', pos_label=0)
    prec = precision_score(bert_true, bert_preds, average='binary', pos_label=0)
    rec = recall_score(bert_true, bert_preds, average='binary', pos_label=0)

    print(macro, prec, rec)

    output_dir = f"attention_oracle/{dataset}"
    sufix = '/'.join(sorted([f"{c[0]}/{c[1]}" for c in CLFS]))
    output_dir = f"{output_dir}/{sufix}/{fold}"
    #os.makedirs(output_dir, exist_ok=True)