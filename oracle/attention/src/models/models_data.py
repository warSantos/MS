from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils import (load_labels_fold,
                       load_probs_fold,
                       expand_labels,
                       expand_probas)

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

def data_loader(dataset: str,
                clf_set: list,
                fold: int,
                batch_size: int):

    labels_train, y_test = load_labels_fold(dataset, fold)
    probas_train, X_test = load_probs_fold(dataset, clf_set, fold, attention_mode=False)
    X_train, X_val, y_train, y_val = train_test_split(probas_train,
                                                        labels_train,
                                                        stratify=labels_train,
                                                        test_size=0.1,
                                                        random_state=42)

    train_loader = DataLoader(dataset=MLPDataset(
        X_train, y_train), batch_size=batch_size, num_workers=6)
    val_loader = DataLoader(dataset=MLPDataset(
        X_val, y_val), batch_size=batch_size, num_workers=6)
    test_loader = DataLoader(dataset=MLPDataset(
        X_test, y_test), batch_size=batch_size, num_workers=6)
    
    return train_loader, val_loader, test_loader