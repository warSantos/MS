import torch
from torch import nn
from torch.optim import AdamW

from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics import F1Score


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

class AttnLogLoss(nn.Module):

    def __init__(self):
        super(AttnLogLoss, self).__init__()

    def forward(self, output, target):

        inner_mult = torch.mul(output, target)
        diff = - torch.log(0.00001 + torch.sum(inner_mult, dim=2))
        sum_target = torch.sum(target, dim=2)

        return torch.sum(1 / target.shape[0] * (torch.mul(diff, sum_target)))

class MultiHeadAttentionEncoder(pl.LightningModule):

    def __init__(self,
                 hidden: int,
                 num_heads: int,
                 dropout: float,
                 apply_upper_bound: bool = False):

        super(MultiHeadAttentionEncoder, self).__init__()

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
        
        
        self.attn_encoder = MultiHeadAttentionEncoder(
            attention_size,
            num_heads,
            dropout,
            apply_upper_bound)

        self.output_layer = nn.Linear(
            classes_number * clfs_number, classes_number)

        # MLP loss function.
        self.mlp_loss = nn.CrossEntropyLoss()

        # MLP activation function.
        self.softmax = nn.Softmax(dim=-1)

        # F1-Macro.
        self.f1 = F1Score(task="multiclass",
                          num_classes=classes_number,
                          average="macro")

    def forward(self, x, attn_x, attn_y):

        attn_out, _ = self.attn_encoder(x, attn_x, attn_y)
        return self.softmax(self.output_layer(attn_out))

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
            on_step=False)
        
        return loss

    def predict_step(self, batch, batch_idx):

        x, _, attn_x, attn_y = batch["x"], batch["y"], batch["attn_x"], batch["attn_y"]
        y_hat = self(x, attn_x, attn_y)
        preds = torch.argmax(y_hat, dim=-1)
        return preds.cpu().numpy()


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
        
        self.join_losses = join_losses
        if self.join_losses:
            self.attn_loss = AttnLogLoss()

    def training_step(self, batch):
        
        #self.attn_encoder.is_train = True
        reporte = {}

        x, y, attn_x, attn_y = batch["x"], batch["y"], batch["attn_x"], batch["attn_y"]
        y_hat = self(x, attn_x, attn_y)
        
        loss = self.mlp_loss(y_hat, y)

        reporte["mlp_loss"] = loss

        if self.join_losses and not self.attn_encoder.apply_upper_bound:

            _, attn_weights = self.attn_encoder(x, attn_x, attn_y)
            attn_loss = self.attn_loss(attn_weights, attn_y)
            # Combining the two losses.
            loss = 0.7 * loss + 0.3 * attn_loss
            
            reporte["attn_loss"] = attn_loss

            reporte["comb_loss"] = loss

        reporte["f1_train"] = self.f1(torch.argmax(y_hat, dim=-1), y)
        
        self.log_dict(reporte, prog_bar=True, on_epoch=True, on_step=False)

        #self.attn_encoder.is_train = False

        return loss

    def configure_optimizers(self):

        opt = AdamW(self.parameters(), lr=1e-2)
        sch = ReduceLROnPlateau(opt, "max", patience=1, factor=0.02)

        return {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": "f1_val"
        }

    def get_stoper(self,
                   monitor="f1_val",
                   min_delta=0.0,
                   patience=5,
                   verbose=False,
                   mode='max'):

        es = EarlyStopping(monitor=monitor,
                             min_delta=min_delta,
                             patience=patience,
                             verbose=verbose,
                             mode=mode)
        return es
    


class DualOptmizerML(BaseMetaLayer):

    def __init__(self,
                 classes_number: int,
                 clfs_number: int,
                 attention_size: int,
                 num_heads: int,
                 dropout: float):

        super().__init__(classes_number,
                         clfs_number,
                         attention_size,
                         num_heads,
                         dropout)

        self.automatic_optimization = False
        self.attn_loss = AttnLogLoss()

    def configure_optimizers(self):

        opt = AdamW(self.output_layer.parameters(), lr=1e-2)
        sch = ReduceLROnPlateau(opt, "max", patience=1, factor=0.02)

        mlp_opt = {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": "f1_val"
        }

        attn_opt = AdamW(self.attn_encoder.parameters(), lr=1e-2)
        attn_sch = ReduceLROnPlateau(opt, "min", patience=1, factor=0.02)

        attn_opt = {
            "optimizer": attn_opt,
            "lr_scheduler": attn_sch,
            "monitor": "attn_val"
        }

        return mlp_opt, attn_opt

    def training_step(self, batch):
        
        x, y, attn_y = batch["x"], batch ["y"], batch["attn_y"]
        
        # Getting optmizers.
        attn_opt, mlp_opt = self.optimizers()
        
        # Computing attention loss.
        attn_output, attn_weights = self.attn_encoder(x, attn_y)
        attn_loss = self.attn_loss(attn_weights, attn_y)
        
        attn_opt.zero_grad()
        self.manual_backward(attn_loss)
        attn_opt.step()

        # Computing MLP loss.
        y_hat = self(x, attn_y)
        mlp_loss = self.mlp_loss(y_hat, y)

        mlp_opt.zero_grad()
        self.manual_backward(mlp_loss)
        mlp_opt.step()

        self.log_dict({
            "attn_loss": attn_loss,
            "CRELoss": mlp_loss,
            "f1_train": self.f1(torch.argmax(y_hat, dim=-1), y)})