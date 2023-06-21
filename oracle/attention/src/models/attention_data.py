import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from src.utils import (load_probs_fold,
                       load_upper_bound,
                       load_labels_fold,
                       get_attention_labels)

from src.attn_mfs import get_error_est_mfs
from src.data import AttentionDataset


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


def attention_data_loader(dataset: str,
                          clf_set: str,
                          fold: int,
                          n_folds: int,
                          batch_size: int,
                          mf_input_dir: str,
                          use_mf: bool = False):
    
    labels_train, y_test = load_labels_fold(dataset, fold)
    probas_train, X_test = load_probs_fold(dataset, clf_set, fold)
    uppers = load_upper_bound(dataset, clf_set, fold)
    attn_y_train = get_attention_labels(clf_set, uppers, "train")
    attn_y_test = get_attention_labels(clf_set, uppers, "test")

    X_train, X_val, y_train, y_val, attn_y_train, attn_y_val = train_test_split(probas_train,
                                                                                labels_train,
                                                                                attn_y_train,
                                                                                stratify=labels_train,
                                                                                test_size=0.1,
                                                                                random_state=42)
    if not use_mf:
        attn_x_train = X_train
        attn_x_val = X_val
        attn_x_test = X_test
    else:
        attn_x_train, attn_x_test = get_error_est_mfs(mf_input_dir,
                                                        dataset,
                                                        clf_set,
                                                        fold,
                                                        n_folds)

        attn_x_train, attn_x_val = train_test_split(attn_x_train,
                                                    stratify=labels_train,
                                                    test_size=0.1,
                                                    random_state=42)
    train_loader = DataLoader(dataset=AttentionDataset(
        X_train, y_train, attn_x_train, attn_y_train), batch_size=batch_size, num_workers=6)
    val_loader = DataLoader(dataset=AttentionDataset(
        X_val, y_val, attn_x_val, attn_y_val), batch_size=batch_size, num_workers=6)
    test_loader = DataLoader(dataset=AttentionDataset(
        X_test, y_test, attn_x_test, attn_y_test), batch_size=batch_size, num_workers=6)
    
    return train_loader, val_loader, test_loader