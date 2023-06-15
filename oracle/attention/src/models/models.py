import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW

from torchmetrics import F1Score

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import numpy as np
torch.manual_seed(42)

class MLP(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # F1-Macro.
        self.f1 = F1Score(task="multiclass",
                          num_classes=output_dim,
                          average="macro")

        self.mlp_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self.forward(x)
        loss = self.mlp_loss(y_hat, y)

        reporte = {
            "train_loss": loss,
            "f1_train": self.f1(torch.argmax(y_hat, dim=-1), y)}
        self.log_dict(reporte, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        self.log_dict(
            {
                "val_loss": loss,
                "f1_val": self.f1(torch.argmax(y_hat, dim=-1), y)
            }, prog_bar=False, on_epoch=True, on_step=True)

        return loss

    def predict_step(self, batch, batch_idx):

        x, _ = batch["x"], batch["y"]
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=-1)
        return preds.cpu().numpy()

    def configure_optimizers(self):

        opt = AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = ReduceLROnPlateau(opt, "max", patience=1, factor=1e-3)
        # sch = CyclicLR(opt, base_lr=1e-5, max_lr=1e-2, mode='triangular2', cycle_momentum=False)

        optimizer = {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": "f1_val"
        }
        return optimizer

    def get_stoper(self,
                   monitor="f1_val",
                   min_delta=0.0,
                   epochs=64,
                   verbose=False,
                   mode='max'):

        patience = int(2 ** (np.log2(epochs) - 2))
        es = EarlyStopping(monitor=monitor,
                           min_delta=min_delta,
                           patience=patience,
                           verbose=verbose,
                           mode=mode)
        return es

    def predict(self):

        expanded_x_val, combinations = expand_probas(X_val, clfs_number)
        expanded_y_val = expand_labels(y_val, clfs_number)
        expanded_val_loader = DataLoader(dataset=MLPDataset(
            expanded_x_val, expanded_y_val), batch_size=batch_size, num_workers=6)
        expanded_preds = trainer.predict(meta_layer, dataloaders=expanded_val_loader)
        np.savez(f"{output_dir}/expanded_val", 
                 expanded_preds=np.hstack(expanded_preds),
                 expanded_labels=expanded_y_val,
                 combinations=combinations)