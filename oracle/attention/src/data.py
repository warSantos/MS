import torch
from torch.utils.data import Dataset

class MLPDataset(Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.X[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx])
        }


class AttentionDataset(Dataset):

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
