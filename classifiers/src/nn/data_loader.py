import io
import os
import numpy as np
import pandas as pd
from src.aws.awsio import load_file_from_aws

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def get_doc_by_id(X, idxs):

    docs = []
    for idx in idxs:
        docs.append(X[idx])
    return docs

class Loader:
    
    def __init__(self,
                 data_dir: str,
                 dataset: str,
                 n_folds: int) -> None:

        self.data_dir = f"{data_dir}/datasets/data"
        self.dataset = dataset
        self.n_folds = n_folds

        # Loading spliting settings with validation set.
        file_path = f"{self.data_dir}/{self.dataset}/splits/split_{self.n_folds}_with_val.pkl"
        data = load_file_from_aws(file_path)
        self.split_settings_with_val = pd.read_pickle(data)
        # Loading spliting settings without validation set.
        file_path = f"{self.data_dir}/{self.dataset}/splits/split_{self.n_folds}.pkl"
        data = load_file_from_aws(file_path)
        self.split_settings = pd.read_pickle(data)
        
        # Loading raw documents.
        file_path = f"{self.data_dir}/{self.dataset}/documents.txt"
        data = load_file_from_aws(file_path)
        self.X = [ text.strip() for text in data.read().decode().splitlines() ]

        # Loading and formating labels.
        file_path = f"{self.data_dir}/{self.dataset}/classes.txt"
        data = load_file_from_aws(file_path)
        y = np.array([ int(i.strip()) for i in data.read().decode().splitlines() ])

        # For binary classification if there is a -1 label, replace it for 0.
        y[y == -1] = 0
        
        # If the min class valeu is 1, subtract 1 from every label.
        if np.min(y) == 1:
            y = y - 1
        
        self.y = y
        self.num_labels = len(np.unique(y))

    def get_X_y(self, fold: int, set_name: str, with_val: bool = True):

        if with_val:
            sp_set = self.split_settings_with_val[self.split_settings_with_val.fold_id == fold]
        else:
            sp_set = self.split_settings[self.split_settings.index == fold]

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