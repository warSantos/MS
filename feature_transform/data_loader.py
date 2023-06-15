import torch
from torch.utils.data import Dataset

class AutoEncoderDataset(Dataset):

    def __init__(self, X, y):

        super().__init__()

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):

        return {
            'x': torch.tensor(self.X[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx], dtype=torch.float32)
        }