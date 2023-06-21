import numpy as np

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class AutoEncoder(pl.LightningModule):

    def __init__(self, input_size: int,
                 hidden_size: int,
                 output_size: int,
                 dropout: float) -> None:

        super(AutoEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Encoder.
        self.e1 = nn.Linear(self.input_size, self.hidden_size)
        self.dp1 = nn.Dropout(dropout)
        self.e2 = nn.Linear(self.hidden_size, self.output_size)
        
        # Decoder.
        self.d1 = nn.Linear(self.output_size, self.hidden_size)
        self.dp2 = nn.Dropout(dropout)
        self.d2 = nn.Linear(self.hidden_size, self.input_size)

        # Activtion functions.
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

        # Loss Function.
        self.loss = nn.MSELoss()
    
    def encode(self, x):

        out = self.relu(self.e1(x))
        out = self.leaky_relu(self.dp1(out))
        out = self.leaky_relu(self.e2(out))
        return out

    def decode(self, x):

        out = self.d1(x)
        out = self.dp2(out)
        out = self.leaky_relu(self.d2(out))
        return out

    def forward(self, x):
        
        # Encoding.
        out = self.encode(x)
        # Decoding.
        out = self.decode(out)
        return out

    def configure_optimizers(self):
        
        optimizer = AdamW(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      mode='min',
                                      factor=0.0003,
                                      patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss"
        }
    
    def get_stoper(self,
                   monitor: str,
                   epochs: int,
                   min_delta: float,
                   mode: str,
                   verbose: bool):

        patience = int(2 ** (np.log2(epochs) - 3))
        
        return EarlyStopping(monitor=monitor,
                               min_delta=min_delta,
                               patience=patience,
                               mode=mode,
                               verbose=verbose)
    
    def training_step(self, batch, batch_idx):

        x, y = batch["x"], batch["y"]

        out = self(x)
        loss = self.loss(out, y)

        self.log_dict({"train_loss": loss},
                      prog_bar=True,
                      on_epoch=True,
                      on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x, y = batch["x"], batch["y"]

        out = self(x)
        loss = self.loss(out, y)

        self.log_dict({"val_loss": loss},
                      prog_bar=True,
                      on_epoch=True,
                      on_step=False)
        return loss
    
    def predict_step(self, batch, batch_idx):

        x = batch["x"]
        return self.encode(x).cpu().numpy()
    
    def fit(self, data, epochs):

        stoper = self.get_stoper("val_loss",
                                 epochs,
                                 0.0,
                                 'min',
                                 False)

        trainer = pl.Trainer(accelerator='gpu',
                             devices=1,
                             max_epochs=epochs,
                             callbacks=[stoper])
        
        train_data_loader, val_data_loader = data

        trainer.fit(self,
                    train_dataloaders=train_data_loader,
                    val_dataloaders=val_data_loader)
        
        return trainer