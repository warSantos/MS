import numpy as np

import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics import F1Score

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.custom_loss import AttnLogLoss, ExpandedLoss

class SelfAttention(pl.LightningModule):

    def __init__(self,
                 hidden: int,
                 num_heads: int,
                 dropout: float,
                 apply_upper_bound: bool = False):

        super(SelfAttention, self).__init__()

        self.apply_upper_bound = apply_upper_bound
        print(f"Applying upper bound: {apply_upper_bound}")
        self.is_train = False

        self.key_fnn = nn.Linear(hidden, hidden)
        self.query_fnn = nn.Linear(hidden, hidden)
        self.value_fnn = nn.Linear(hidden, hidden)

        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout)

    def forward(self, x, attn_x, attn_y):

        key = self.key_fnn(attn_x)
        query = self.query_fnn(attn_x)
        value = self.value_fnn(attn_x)

        # Computing attention weights.
        attn_out, attn_weights = self.multihead_attn(query, key, value)

        if self.apply_upper_bound or self.is_train:
            learned_w = torch.mean(attn_y, dim=1)
        else:
            learned_w = torch.mean(attn_weights, dim=1)

        # Applying attention weights for each classifier.
        weighted_probs = x * learned_w.unsqueeze(-1)
        return torch.flatten(weighted_probs, start_dim=1), attn_weights


class BaseMetaLayer(pl.LightningModule):

    def __init__(self,
                 classes_number: int,
                 clfs_number: int,
                 attention_size: int,
                 num_heads: int,
                 dropout: float,
                 apply_upper_bound: bool = False):
        super().__init__()

        self.attn_encoder = SelfAttention(
            attention_size,
            num_heads,
            dropout,
            apply_upper_bound)

        self.output_layer = nn.Linear(
            classes_number * clfs_number, classes_number)

        # MLP loss function.
        self.mlp_loss = nn.CrossEntropyLoss()
        
        # Attention Loss.
        self.attn_loss = AttnLogLoss()

        # MLP activation function.
        self.softmax = nn.Softmax(dim=-1)

        # F1-Macro.
        self.f1 = F1Score(task="multiclass",
                          num_classes=classes_number,
                          average="macro")

    def forward(self, x, attn_x, attn_y):

        attn_out, _ = self.attn_encoder(x, attn_x, attn_y)
        return self.output_layer(attn_out)

    def validation_step(self, batch, batch_idx):

        x, y, attn_x, attn_y = batch["x"], batch["y"], batch["attn_x"], batch["attn_y"]
        y_hat = self(x, attn_x, attn_y)
        loss = self.mlp_loss(y_hat, y)

        self.log_dict(
            {
                "f1_val": self.f1(torch.argmax(y_hat, dim=-1), y),
                "cross_loss_val": loss
            },
            prog_bar=False,
            on_epoch=True,
            on_step=True)

        return loss

    def predict_step(self, batch, batch_idx):

        x, _, attn_x, attn_y = batch["x"], batch["y"], batch["attn_x"], batch["attn_y"]
        y_hat = self(x, attn_x, attn_y)
        preds = torch.argmax(y_hat, dim=-1)
        return preds.cpu().numpy()

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
    
    def fit(self,
            max_epochs,
            tb_logger: loggers.TensorBoardLogger):

        trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[self.get_stoper(epochs=max_epochs)],
                         logger=tb_logger)

        trainer.fit(meta_layer, train_dataloaders=train_loader,
                    val_dataloaders=val_loader)


class SingleOptimizerML(BaseMetaLayer):

    def __init__(self,
                 classes_number: int,
                 clfs_number: int,
                 attention_size: int,
                 num_heads: int,
                 dropout: float,
                 apply_upper_bound: bool = False,
                 join_losses: bool = False):

        super().__init__(classes_number,
                         clfs_number,
                         attention_size,
                         num_heads,
                         dropout,
                         apply_upper_bound)

    def training_step(self, batch):

        reporte = {}

        x, y, attn_x, attn_y = batch["x"], batch["y"], batch["attn_x"], batch["attn_y"]
        
        # Computing attention loss.
        attn_loss = self.attn_loss(attn_weights, attn_y)
        _, attn_weights = self.attn_encoder(x, attn_x, attn_y)
        reporte["attn_loss"] = attn_loss

        # Computing MLP loss.
        y_hat = self(x, attn_x, attn_y)
        loss = self.mlp_loss(y_hat, y)
        reporte["CRELoss"] = loss
        
        # Evaluating the model.
        reporte["f1_train"] = self.f1(torch.argmax(y_hat, dim=-1), y)

        self.log_dict(reporte, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):

        opt = AdamW(self.parameters(), lr=1e-2)
        sch = ReduceLROnPlateau(opt, "max", patience=1, factor=0.02)

        return {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": "f1_val"
        }


class DualOptmizerML(BaseMetaLayer):

    def __init__(self,
                 classes_number: int,
                 clfs_number: int,
                 attention_size: int,
                 num_heads: int,
                 dropout: float,
                 apply_upper_bound: bool = False):

        super().__init__(classes_number,
                         clfs_number,
                         attention_size,
                         num_heads,
                         dropout)

        self.automatic_optimization = False

    def configure_optimizers(self):

        opt = AdamW(self.output_layer.parameters(), lr=1e-2, weight_decay=1e-4)
        sch = ReduceLROnPlateau(opt, "max", patience=1, factor=0.02)
        # sch = CyclicLR(opt, base_lr=1e-5, max_lr=1e-2, mode='triangular2', cycle_momentum=False, step_size_up=100)

        mlp_opt = {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": "f1_val"
        }

        attn_opt = AdamW(self.attn_encoder.parameters(),
                         lr=1e-2, weight_decay=1e-4)
        attn_sch = ReduceLROnPlateau(opt, "min", patience=1, factor=0.02)
        # attn_sch = CyclicLR(attn_opt, base_lr=1e-5, max_lr=1e-2, mode='triangular2', cycle_momentum=False, step_size_up=100)

        attn_opt = {
            "optimizer": attn_opt,
            "lr_scheduler": attn_sch,
            "monitor": "attn_val"
        }

        return attn_opt, mlp_opt

    def training_step(self, batch):

        # self.attn_encoder.is_train = True
        x, y, attn_x, attn_y = batch["x"], batch["y"], batch["attn_x"], batch["attn_y"]

        # Getting optmizers.
        attn_opt, mlp_opt = self.optimizers()

        # Computing attention loss.
        _, attn_weights = self.attn_encoder(x, attn_x, attn_y)
        attn_loss = self.attn_loss(attn_weights, attn_y)

        attn_opt.zero_grad()
        self.manual_backward(attn_loss)
        attn_opt.step()

        # Computing MLP loss.
        y_hat = self(x, attn_x, attn_y)
        mlp_loss = self.mlp_loss(y_hat, y)

        mlp_opt.zero_grad()
        self.manual_backward(mlp_loss)
        mlp_opt.step()

        reporte = {
            "attn_loss": attn_loss,
            "CRELoss": mlp_loss,
            "f1_train": self.f1(torch.argmax(y_hat, dim=-1), y)}
        self.log_dict(reporte, prog_bar=True, on_epoch=True, on_step=False)
        # self.attn_encoder.is_train = False


class ExpandedAttention(pl.LightningModule):

    def __init__(self,
                 classes_number: int,
                 clfs_number: int,
                 attention_size: int,
                 num_heads: int,
                 dropout: float,) -> None:

        super().__init__()

        self.apply_upper_bound = False
        self.class_number = classes_number
        self.clfs_number = clfs_number
        self.expansion_factor = 2 ** self.clfs_number

        self.attn_encoder = SelfAttention(attention_size,
                                                      num_heads,
                                                      dropout,
                                                      self.apply_upper_bound)
        #self.attn_loss = ExpandedLoss()
        self.attn_loss = torch.nn.MSELoss()

    def forward(self, x):

        _, attn_weights = self.attn_encoder(x)
        # Getting the mean values and rounding them.
        mw = torch.round(torch.mean(attn_weights, dim=1))
        n_docs = mw.size[0]
        expanded = mw.tile(1, self.expansion_factor).reshape(
            n_docs, self.expansion_factor, -1)
        return attn_weights, expanded

    def training_step(self, batch, batch_idx):

        x, y = batch["x"], batch["y"]
        _, expanded = self.forward(x)
        loss = self.attn_loss(expanded, y)
        self.log_dict({"train_loss": loss},
                      prog_bar=True,
                      on_epoch=True,
                      on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch["x"], batch["y"]

        _, expanded = self.forward(x)
        loss = self.attn_loss(expanded, y)
        self.log_dict({"val_loss": loss},
                      prog_bar=True,
                      on_epoch=True,
                      on_step=False)
        return loss

    """
    def predict_step(self, batch, batch_idx):

        x, _ = batch["x"], batch["y"]
    """

    def configure_optimizers(self):

        opt = AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = ReduceLROnPlateau(opt, "max", patience=1, factor=1e-3)

        optimizer = {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": "val_loss"
        }
        return optimizer

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
