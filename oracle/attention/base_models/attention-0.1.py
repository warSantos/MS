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

from sklearn.metrics import f1_score

def transform_probas(probas_set: list):

    features = []
    for doc_idx in np.arange(probas_set[0].shape[0]):
        features.append(
            np.vstack([ probas_set[clf_idx][doc_idx] for clf_idx in np.arange(len(probas_set)) ])
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

def load_upper_bound(dataset: str, clfs: list, fold: int):

    uppers = {}
    for clf, proba_type in clfs:
        upper_dir = f"/home/welton/data/oracle/upper_bound/{proba_type}/{dataset}/10_folds/{clf}/{fold}"
        uppers[clf] = {}
        uppers[clf]["train"] = np.load(f"{upper_dir}/train.npz")['y']
        uppers[clf]["test"] = np.load(f"{upper_dir}/test.npz")['y']
    
    return uppers

def get_attention_labels(clfs: list, uppers: dict, train_test: str):

    n_docs = uppers[clfs[0][0]][train_test].shape[0]
    attention_labels = []
    for idx in np.arange(n_docs):
        clfs_rows = []
        for clf, _ in clfs:
            clfs_rows.append(
                np.zeros(len(clfs)) + uppers[clf][train_test][idx]
            )
        attention_labels.append(np.array(clfs_rows).T)
    
    return attention_labels


def load_labels_fold(dataset: str, fold: int):

    y_train = np.load(f"/home/welton/data/datasets/labels/split_10/{dataset}/{fold}/train.npy")
    y_test = np.load(f"/home/welton/data/datasets/labels/split_10/{dataset}/{fold}/test.npy")
    return y_train, y_test

class StackingDataset(Dataset):

    def __init__(self, X, y, attn_y) -> None:
        super().__init__()

        self.X = X
        self.y = y
        self.attn_y = attn_y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.X[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx]),
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

    def forward(self, x):

        key = self.key_fnn(x)
        query = self.query_fnn(x)
        value = self.value_fnn(x)
        attn_output, attn_weights = self.multihead_att(query, key, value)
        
        w = torch.mean(attn_weights, dim=2)
        weighted_probs = x * w.unsqueeze(-1)
        return torch.flatten(weighted_probs, start_dim=1), attn_weights
    
class MetaLayer(pl.LightningModule):

    def __init__(self, classes_number, clfs_number, num_heads, dropout):

        super().__init__()
        
        self.automatic_optimization = False

        self.attn_encoder = MultiHeadAttentionEncoder(
            classes_number,
            num_heads,
            dropout)
        
        #self.attn_loss = nn.MSELoss()
        self.attn_loss = DiffMatrixLoss()
        
        self.output_layer = nn.Linear(classes_number * clfs_number, classes_number)
        self.mlp_loss = nn.CrossEntropyLoss()
        
        self.softmax = nn.Softmax(dim=-1)
        self.f1 = F1Score(task="multiclass", num_classes=classes_number)

    def forward(self, x):
        
        attn_out, _ = self.attn_encoder(x)
        return self.softmax(self.output_layer(attn_out))
        

    def configure_optimizers(self):
        
        attn_opt = AdamW(self.attn_encoder.parameters(), lr = 1e-3)
        mlp_opt = AdamW(self.output_layer.parameters(), lr = 1e-3)
        return attn_opt, mlp_opt
        
    def training_step(self, batch):
        
        x, y, attn_y = batch["x"], batch ["y"], batch["attn_y"]
        
        # Getting optmizers.
        attn_opt, mlp_opt = self.optimizers()
        
        # Computing attention loss.
        attn_output, attn_weights = self.attn_encoder(x)
        attn_loss = self.attn_loss(attn_weights, attn_y)
        
        attn_opt.zero_grad()
        self.manual_backward(attn_loss)
        attn_opt.step()

        # Computing MLP loss.
        y_hat = self(x)
        mlp_loss = self.mlp_loss(y_hat, y)

        mlp_opt.zero_grad()
        self.manual_backward(mlp_loss)
        mlp_opt.step()

        self.log_dict({"Attention MSELoss": attn_loss}, prog_bar=True)
        self.log_dict({"f1": self.f1(torch.argmax(y_hat, dim=-1), y)}, prog_bar=True)

    
    def validation_step(self, batch, batch_idx):
        
        x, y, attn_y = batch["x"], batch ["y"], batch["attn_y"]
        
        # forward pass.
        attn_output, attn_weights = self.attn_encoder(x)
        attn_loss = self.attn_loss(attn_weights, attn_y)

        y_hat = self.output_layer(attn_output)
        
        #self.log_dict({"Attention MSELoss": attn_loss}, prog_bar=True)
        #self.log_dict({"f1": self.f1(torch.argmax(y_hat, dim=-1), y)}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        
        x, y = batch["x"], batch ["y"]
        # forward pass.
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=-1)
        self.log_dict({"f1": self.f1(preds, y)}, prog_bar=True)
        return preds.cpu().numpy()

    def predict_step(self, batch, batch_idx):
        
        x, y = batch["x"], batch["y"]
        # forward pass.
        attn_output, _ = self.attn_encoder(x)
        y_hat = self.output_layer(attn_output)
        preds = torch.argmax(y_hat, dim=-1)
        return preds.cpu().numpy()


CLFS = [["bert", "normal_probas"],
        ["xlnet", "normal_probas"],
        ["ktmk", "normal_probas"],
        ["ktr", "normal_probas"],
        ["lstmk", "normal_probas"],
        ["lstr", "normal_probas"],
        ["ltr", "normal_probas"]]

DATASETS = ["acm"]

iters = product(DATASETS, np.arange(10))

for dataset, fold in iters:

    labels_train, y_test = load_labels_fold(dataset, fold)
    probas_train, X_test = load_probs_fold(dataset, CLFS, fold)

    
    uppers = load_upper_bound(dataset, CLFS, fold)
    attention_train = get_attention_labels(CLFS, uppers, "train")
    attn_test = get_attention_labels(CLFS, uppers, "test")

    X_train, X_val, y_train, y_val, attn_train, attn_val = train_test_split(probas_train,
                                                                          labels_train,
                                                                          attention_train,
                                                                          stratify=labels_train,
                                                                          test_size=0.1,
                                                                          random_state=42)
    
    clfs_number = len(CLFS)
    classes_number = X_train[0].shape[1]
    max_epochs = 30
    batch_size = 128

    train_loader = DataLoader(dataset = StackingDataset(X_train, y_train, attn_train), batch_size=batch_size, num_workers=16)
    val_loader = DataLoader(dataset = StackingDataset(X_val, y_val, attn_val), batch_size=batch_size, num_workers=16)
    test_loader = DataLoader(dataset = StackingDataset(X_test, y_test, attn_test), batch_size=batch_size, num_workers=16)

    meta_layer = MetaLayer(classes_number, clfs_number, 1, 0.1)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs)
    trainer.fit(meta_layer, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    preds = trainer.predict(meta_layer, dataloaders=test_loader)
    preds = np.hstack(preds)

    macro = f1_score(y_test, preds, average="macro")
    micro = f1_score(y_test, preds, average="micro")
    
    print(f"Macro: {macro} - Micro: {micro}")

    output_dir = f"reports/{dataset}"
    sufix = '/'.join(sorted([ f"{c[0]}/{c[1]}" for c in CLFS ]))
    output_dir = f"{output_dir}/{sufix}/{fold}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/scoring.json", 'w') as fd:
        scoring = {
            "macro": macro,
            "micro": micro
        }
        json.dump(scoring, fd)