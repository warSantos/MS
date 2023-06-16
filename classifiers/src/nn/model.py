import torch
from torch.optim import AdamW
from transformers import get_scheduler, AutoModelForSequenceClassification
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torchmetrics import F1Score


class CustomDataset(Dataset):

    def __init__(self, x, y):

        self.x = x
        self.y = y

    def __getitem__(self, idx):
        
        return {
            "input_ids": torch.tensor(self.x["input_ids"][idx]),
            "attention_mask": torch.tensor(self.x["attention_mask"][idx]),
            "token_type_ids": torch.tensor(self.x["token_type_ids"][idx]),
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
        

    def prepare_data(self, X, y):

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
                                 shuffle=True,
                                 batch_size=self.batch_size,
                                 worker_init_fn=self.seed,
                                 pin_memory=True)
        return data_loader
    

class Transformer(pl.LightningModule):

    def __init__(self,
                 model_name,
                 len_data_loader,
                 num_classes,
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
        self.num_classes = num_classes

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.trainer = pl.Trainer(accelerator="gpu",
                                  devices=1,
                                  max_epochs=self.max_epochs)
        self.f1 = F1Score(task="multiclass", average="macro", num_classes=num_classes)


    def configure_optimizers(self):

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        num_training_steps = self.max_epochs * self.len_data_loader
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps)
        
        return { "optimizer": optimizer, "scheduler": lr_scheduler }

    def forward(self, **batch):

        return self.model(**batch)

    def training_step(self, batch):

        output = self(**batch)
        return output.loss
    
    def validation_step(self, batch, batch_idx):
        
        output = self(**batch)
        y_hat = torch.argmax(output.logits, dim=1)
        
        self.log_dict({"val_f1": self.f1(y_hat, batch["labels"])}, prog_bar=True)
    
    def on_validation_epoch_end(self, outs):

        self.f1.compute()
        
    def predict(self, batch):

        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        y_hat = self(input_ids, attention_mask)
        return y_hat.logits.cpu().numpy()

    def fit(self, train, val):
        
        seed_everything(self.seed, workers=True)
        self.trainer.fit(self,
                         train_dataloaders=train,
                         val_dataloaders=val)