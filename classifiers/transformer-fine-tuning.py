
# imports
import io
import os
import json
import random

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from tqdm.auto import tqdm


def get_doc_by_id(X, idxs):

    docs = []
    for idx in idxs:
        docs.append(X[idx])
    return docs


def data_process(data_dir, dataset, fold, n_folds):

    split_settings = pd.read_pickle(
        f"{data_dir}/{dataset}/splits/split_{n_folds}_with_val.pkl")
    
    with io.open(f"{data_dir}/{dataset}/documents.txt", newline='\n', errors='ignore') as read:
        X = []
        for row in read:
            X.append(row.strip())

    labels = []
    with open(f"{data_dir}/{dataset}/classes.txt") as fd:
        for line in fd:
            labels.append(int(line.strip()))
    labels = np.array(labels)
    num_labels = len(np.unique(labels))

    # For binary classification if there is a -1 label, replace it for 0.
    labels[labels == -1] = 0
    # If the min class valeu is 1, subtract 1 from every label.
    if np.min(labels == 1):
        labels = labels - 1

    sp_set = split_settings[split_settings.fold_id == fold]

    X_train = get_doc_by_id(X, sp_set.train_idxs.tolist()[0])
    X_test = get_doc_by_id(X, sp_set.test_idxs.tolist()[0])
    X_val = get_doc_by_id(X, sp_set.val_idxs.tolist()[0])

    y_train = labels[sp_set.train_idxs.iloc[0]]
    y_test = labels[sp_set.test_idxs.iloc[0]]
    y_val = labels[sp_set.val_idxs.iloc[0]]

    return X_train, X_test, X_val, y_train, y_test, y_val, split_settings, num_labels

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) 
          for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class Transformer():

    def __init__(
        self,
        model_name,
        batch_size=32,
        limit_k=20,
        max_epochs=5,
        lr=5e-5,
        max_length=128,
        limit_patient=3,
        padding=True,
        truncation=True,
        seed=42,
        load_model: bool = True
    ):
        self.batch_size = batch_size
        self.limit_k = limit_k
        self.max_epochs = max_epochs
        self.lr = lr
        self.max_length = max_length
        self.limit_patient = limit_patient
        self.model_name = model_name
        self.padding = padding
        self.truncation = truncation
        self.seed = seed
        self.load_model = load_model

    def encode_data(self, X_train, X_test, X_val, y_train, y_test, y_val):

        # Applying text encoding.
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        train_encodings = tokenizer(X_train, max_length=self.max_length,
                                    return_tensors="pt",  padding=self.padding, truncation=self.truncation)
        val_encodings = tokenizer(X_val, max_length=self.max_length,
                                  return_tensors="pt",  padding=self.padding, truncation=self.truncation)
        test_encodings = tokenizer(X_test, max_length=self.max_length,
                                   return_tensors="pt",  padding=self.padding, truncation=self.truncation)

        # Formating dataset on Pytorch manners.
        train_dataset = CustomDataset(train_encodings, y_train)
        val_dataset = CustomDataset(val_encodings, y_val)
        test_dataset = CustomDataset(test_encodings, y_test)

        # Splitting data in batches.
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.batch_size, worker_init_fn=self.seed)
        eval_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, worker_init_fn=self.seed)
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, worker_init_fn=self.seed)

        return train_dataloader, eval_dataloader, test_dataloader

    def fit_predict(self, X_train, X_test, X_val, y_train, y_test, y_val, num_labels):

        # Preparing data.
        train_dataloader, eval_dataloader, test_dataloader = self.encode_data(
            X_train, X_test, X_val, y_train, y_test, y_val)
        
        print(f"Loading {self.model_name} pre-trained model...")
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        print(f"Model {self.model_name} already in memory.")
        
        optimizer = AdamW(model.parameters(), lr=self.lr)
        num_training_steps = self.max_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        # Sending model to GPU.
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        progress_bar = tqdm(range(num_training_steps))
        cont_patient = 0
        min_loss_eval = 10000

        # For each epoch.
        for epoch in range(self.max_epochs):
            
            # ---------- TRAIN ---------- #
            model.train()
            # For each batch.
            for batch in train_dataloader:
                progress_bar.update(1)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ---------- VALIDATION ---------- #
            y_pred_list = []
            y_true_list = []
            eval_logits = []
            model.eval()
            for batch in eval_dataloader:

                batch = {k: v.to(device) for k, v in batch.items()}
                
                with torch.no_grad():
                    outputs = model(**batch)
                
                # Saving validation logits.
                eval_logits.append(outputs.logits.cpu().numpy().tolist())
                
                loss = outputs.loss
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                y_pred_list.append(predictions.tolist())
                y_true_list.append(list(batch["labels"].tolist()))

            y_pred_batch = []
            y_true_batch = []

            for y_batch in y_pred_list:  # y_batchs
                for y_doc in y_batch:
                    y_pred_batch.append(y_doc)

            for y_batch in y_true_list:  # y_batchs
                for y_doc in y_batch:
                    y_true_batch.append(y_doc)

            loss_eval_atual = loss.item()

            # Stop training if no improviment was showed.
            if loss_eval_atual < min_loss_eval:
                cont_patient = 0
                min_loss_eval = loss_eval_atual
            else:
                cont_patient += 1

            if cont_patient >= self.limit_patient:
                break

        # ---------- TESTE ---------- #
        test_probs = []
        test_logits = []
        model.eval()
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                # Saving test logits.
                test_logits.append(outputs.logits.cpu().numpy().tolist())
                # Applying softmax on logits and saving probabilities.
                norm = softmax(outputs.logits, dim=-1).tolist()
                test_probs.append(norm)

        test_probs = np.vstack(test_probs)
        eval_logits = np.vstack(eval_logits)
        test_logits = np.vstack(test_logits)
        return test_probs, test_logits, eval_logits

def get_train_probas(X, y, base_path, num_labels, model_name, n_splits=4):

    # Spliitng train probabilities.
    sfk = StratifiedKFold(n_splits=n_splits)
    sfk.get_n_splits(X, y)
    # This list will hold all the folds probabilities.
    probas = []
    # This list witl hold all the document's indexes to re sort later.
    # When fine-tuning is done.
    idx_list = []
    alig_idx = np.arange(y.shape[0])
    # For each fold.
    for fold, (train_index, test_index) in enumerate(sfk.split(X, y)):
                
        # Selecting train documents and labels.
        X_train = get_doc_by_id(X, train_index)
        y_train = y[train_index]
        
        # Spliting the train documents in train and validation.
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1)
        
        # Getting test documents.
        X_test = get_doc_by_id(X, test_index)
        y_test = y[test_index]
        align = alig_idx[test_index]
        
        idx_list.append(align)

        # Applying oversampling when it is needed. Sometimes a class doesn't have
        # enough documents from a class.
        for c in set(y_test) - set(y_train):

            sintetic = "fake document"
            X_train.append(sintetic)
            y_train = np.hstack([y_train, [c]])
        
        # Loading the pre-trained model into the memory.
        model = Transformer(model_name)
        # Training and predicing.
        test_p, test_l, eval_l = model.fit_predict(
            X_train, X_test, X_val, y_train, y_test, y_val, num_labels)
        
        # Saving train and validation logits.
        subfold_path = f"{base_path}/sub_fold/{fold}"
        os.makedirs(subfold_path, exist_ok=True)
        np.savez(f"{subfold_path}/eval_logits", X_eval=eval_l, y_eval=y_val)
        np.savez(f"{subfold_path}/test_logits", X_test=test_l, y_test=y_test)
        
        # Saving fold document's indexes.
        np.savez(f"{subfold_path}/align", align=align)
        
        # Saving fold's probabilities.
        probas.append(test_p)
        scoring = {}
        
        # Printing model's performance.
        y_pred = test_p.argmax(axis=1)
        scoring["macro"] = f1_score(y_test, y_pred, average="macro")
        scoring["micro"] = f1_score(y_test, y_pred, average="micro")
        print(
            f"\t\tSUB-FOLD {fold} - Macro: {scoring['macro']} - Micro {scoring['micro']}")
        torch.cuda.empty_cache()
    
    # Joining folds probabilities.
    probas = np.vstack(probas)
    
    # Resorting document's probabilities.
    sorted_idxs = np.hstack(idx_list).argsort()
    probas = probas[sorted_idxs]
    probas_path = f"{base_path}/train"
    # Saving document's proabilities.
    np.savez(probas_path, X_train=probas)


with open("data/nn_settings.json", 'r') as fd:
    settings = json.load(fd)

SEED = settings["SEED"]
data_dir = settings["DATA_DIR"]
data_source = settings["DATA_SOURCE"]
datasets = settings["DATASETS"]
classifiers = settings["CLFS"]
n_sub_folds = settings["N_SUB_FOLDS"]

# For each dataset.
for dataset, n_folds in datasets:
    for classifier_name, classifier_short_name in classifiers:
        print(f"CLF: {classifier_name.upper()}")
        for fold in np.arange(10):
            
            from transformers import logging
            logging.set_verbosity_error()        
            
            # Setting static seed to prevent variational results.
            random.seed(SEED)
            torch.manual_seed(SEED)
            np.random.seed(seed=SEED)
            
            print(f"{dataset.upper()} - FOLD: {fold}")

            # Splitting data.
            data_processed = data_process(data_source, dataset, fold, n_folds)
            X_train, X_test, X_val, y_train, y_test, y_val, sp_settings, num_labels = data_processed
            
            base_path = f"{data_dir}/normal_probas/split_10/{dataset}/10_folds/{classifier_short_name}/{fold}"
            
            os.makedirs(base_path, exist_ok=True)
            test_path = f"{base_path}/test"
            eval_path = f"{base_path}/eval"
            eval_logits_path = f"{base_path}/eval_logits"
            test_logits_path = f"{base_path}/test_logits"

            # Se este fold ainda não foi executado.
            if not os.path.exists(test_path):
                print("Builind test probabilities...")
                # Setting model's parameters.
                model = Transformer(classifier_name, padding=True, truncation=True, seed=SEED)
                
                # Training and predicting.
                test_p, test_l, eval_l = model.fit_predict(
                    X_train, X_test, X_val, y_train, y_test, y_val, num_labels)
                
                # Saving model's probabilities.
                np.savez(test_path, X_test=test_p)
                # Saving model's logits.
                np.savez(eval_logits_path, X_eval=eval_l, y_eval=y_val)
                np.savez(test_logits_path, X_test=test_l, y_test=y_test)
                # Printing model performance.
                y_pred = test_p.argmax(axis=1)
                print(f"Macro: {f1_score(y_test, y_pred, average='macro')}")
                print(f"Micro: {f1_score(y_test, y_pred, average='micro')}")
                torch.cuda.empty_cache()
                
            train_path = f"{base_path}/train"
            # If train probabilities weren't computed yet.
            if not os.path.exists(train_path):
                print("Builind train probabilities...")
                # Joining train indexes with validation indexes.
                idxs = sp_settings.iloc[fold]["train_idxs"] + sp_settings.iloc[fold]["val_idxs"]
                # Sorting indexes. It's important to match the train document's probabilities
                # of data split with validantion and without validation (just train and test).
                sort = np.array(idxs).argsort()
                # Computing train probabilities.
                get_train_probas(get_doc_by_id(X_train + X_val, sort),
                                 np.hstack([y_train, y_val])[sort], 
                                 base_path, 
                                 num_labels,
                                 classifier_name,
                                 n_sub_folds)