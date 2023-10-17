import os
from glob import glob
from pathlib import Path
from typing import Any, List, Sequence, Optional
import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_scheduler, AutoModelForSequenceClassification
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import BasePredictionWriter
from torchmetrics import F1Score


class CustomDataset(Dataset):

    def __init__(self, x, y):

        self.x = x
        self.y = y

    def __getitem__(self, idx):
        
        return {
            "input_ids": self.x["input_ids"][idx].clone().detach(),
            "attention_mask": self.x["attention_mask"][idx].clone().detach(),
            "token_type_ids": self.x["token_type_ids"][idx].clone().detach(),
            "labels": torch.tensor(self.y[idx])}

    def __len__(self):

        return len(self.y)

class TextFormater:

    def __init__(self, model_name,
                 max_length,
                 padding,
                 truncation,
                 batch_size,
                 seed):

        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.batch_size = batch_size
        self.seed = seed
        

    def prepare_data(self, X, y, shuffle=False):

        # Applying text encoding.
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X_enc = tokenizer(X,
                          max_length=self.max_length,
                          padding=self.padding,
                          truncation=self.truncation,
                          return_tensors="pt")

        # Formating dataset on Pytorch manners.
        data = CustomDataset(X_enc, y)

        # Splitting data in batches.
        data_loader = DataLoader(data,
                                 shuffle=shuffle,
                                 batch_size=self.batch_size,
                                 pin_memory=True,
                                 num_workers=os.cpu_count() - 1)
        return data_loader
    

class Transformer(pl.LightningModule):

    def __init__(self,
                 model_name,
                 len_data_loader,
                 num_labels,
                 limit_k=20,
                 max_epochs=5,
                 lr=5e-5,
                 limit_patient=3,
                 seed=42):
        super(Transformer, self).__init__()
        
        self.model_name = model_name
        self.len_data_loader = len_data_loader
        self.limit_k = limit_k
        self.max_epochs = max_epochs
        self.lr = lr
        self.limit_patient = limit_patient
        self.seed = seed
        self.num_labels = num_labels

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels)
        self.f1 = F1Score(task="multiclass", average="macro", num_classes=self.num_labels)


    def configure_optimizers(self):

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        num_training_steps = self.max_epochs * self.len_data_loader
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps)
        
        return { "optimizer": optimizer, "scheduler": lr_scheduler }

    def forward(self, batch):
        
        return self.model(**batch)

    def training_step(self, batch):

        output = self(batch)
        return output.loss
    
    def validation_step(self, batch, batch_idx):
        
        output = self(batch)
        y_hat = torch.argmax(output.logits, dim=1)
        
        self.log_dict({"val_f1": self.f1(y_hat, batch["labels"])}, prog_bar=True)

    """
    def validation_epoch_end(self, outs):

        self.f1.compute()
    """    
        
    def predict(self, batch):

        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        y_hat = self(input_ids, attention_mask)
        return y_hat.logits.cpu()

class FitHelper:

    def fit(self, model, train, val, max_epochs, seed = 42):

        seed_everything(seed, workers=True)
        trainer = pl.Trainer(accelerator="gpu",
                             devices=1,
                             max_epochs=max_epochs,
                             callbacks=[PredictionWriter()])
        

        trainer.fit(model, train_dataloaders=train, val_dataloaders=val)
        return trainer
    
    def load_logits_batches(self):

        idxs = []
        for f in glob(".preds/*"):
            idxs.append(int(f.split('/')[1].split('.')[0]))
        idxs.sort()

        logits = []
        for i in idxs:
            logits.append(np.array(torch.load(f".preds/{i}.prd")))
        logits = np.vstack(logits)
        os.system("rm -rf .preds/*")
        return logits

class PredictionWriter(BasePredictionWriter):

    def __init__(self):
        super(PredictionWriter, self).__init__()
        self.checkpoint_dir = ".preds/"
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


    def write_on_epoch_end(self,
                           trainer: "pl.Trainer",
                           pl_module: "pl.LightningModule",
                           predictions: Sequence[Any],
                           batch_indices: Optional[Sequence[Any]]) -> None:
        pass

    def write_on_batch_end(self,
                           trainer,
                           pl_module,
                           prediction: Any,
                           batch_indices: List[int],
                           batch: Any,
                           batch_idx: int,
                           dataloader_idx: int):

        predictions = []
        for logit in prediction["logits"].tolist():
            predictions.append(logit)

        self._checkpoint(predictions, dataloader_idx, batch_idx)

    def _checkpoint(self, predictions, dataloader_idx, batch_idx):
        
        torch.save(
            predictions,
            f"{self.checkpoint_dir}{batch_idx}.prd"
        )