import io
import numpy as np
import pandas as pd

def get_doc_by_id(X, idxs):

    docs = []
    for idx in idxs:
        docs.append(X[idx])
    return docs

class Loader:
    
    def __init__(self,
                 data_dir: str,
                 dataset: str,
                 fold: int,
                 n_folds: int) -> None:

        self.data_dir = data_dir
        self.dataset = dataset
        self.fold = fold
        self.n_folds = n_folds

        # Loading spliting settings.
        self.split_settings = pd.read_pickle(f"{self.data_dir}/{self.dataset}/splits/split_{self.n_folds}_with_val.pkl")
        
        # Loading raw documents.
        with io.open(f"{self.data_dir}/{self.dataset}/documents.txt", newline='\n', errors='ignore') as read:
            X = []
            for row in read:
                X.append(row.strip())
            self.X = X

        # Loading and formating labels.
        y = []
        with open(f"{self.data_dir}/{self.dataset}/classes.txt") as fd:
            for line in fd:
                y.append(int(line.strip()))
        y = np.array(y)
        
        # For binary classification if there is a -1 label, replace it for 0.
        y[y == -1] = 0
        
        # If the min class valeu is 1, subtract 1 from every label.
        if np.min(y == 1):
            y = y - 1
        self.y = y

        self.num_labels = len(np.unique(y))

    def get_X_y(self, fold: int, set_name: str):

        sp_set = self.split_settings[self.split_settings.fold_id == fold]

        if set_name == "train":
            
            X_train = get_doc_by_id(self.X, sp_set.train_idxs.tolist()[0])
            y_train = self.y[sp_set.train_idxs.iloc[0]]
            return X_train, y_train
        
        elif set_name == "test":

            X_test = get_doc_by_id(self.X, sp_set.test_idxs.tolist()[0])
            y_test = self.y[sp_set.test_idxs.iloc[0]]
            return X_test, y_test
        
        elif set_name == "val":
        
            X_val = get_doc_by_id(self.X, sp_set.val_idxs.tolist()[0])
            y_val = self.y[sp_set.val_idxs.iloc[0]]
            return X_val, y_val
        
        else:
            raise("Set option not valid. Valid options are: train, test and val.")