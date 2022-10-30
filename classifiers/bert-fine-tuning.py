#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('nvidia-smi ## Verificar placa de vídeo disponível no momento da execução')


# In[1]:


# imports
import io
import os
import pickle
import random
import jsonlines
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch.nn.functional import softmax
from torch.utils import data  # trabalhar com dados iterados
from transformers import BertModel, BertTokenizer, AutoTokenizer
from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from tqdm.auto import tqdm

import timeit  # calcular metrica de tempo


# In[28]:


def get_doc_by_id(X, idxs):

    docs = []
    for idx in idxs:
        docs.append(X[idx])
    return docs


def data_process(dataset, fold, fast_test=False, sample_size=100):

    split_settings = pd.read_pickle(
        f"/home/welton/data/datasets/data/{dataset}/splits/split_10_with_val.pkl")
    with io.open(f"/home/welton/data/datasets/data/{dataset}/texts.txt", newline='\n', errors='ignore') as read:
        X = []
        for row in read:
            X.append(row.strip())

    labels = []
    with open(f"/home/welton/data/datasets/data/{dataset}/score.txt") as fd:
        for line in fd:
            labels.append(int(line.strip()))
    labels = np.array(labels)

    # Ajustando labels.
    labels[labels == -1] = 0
    if np.min(labels == 1):
        labels = labels - 1

    sp_set = split_settings[split_settings.fold_id == fold]

    X_train = get_doc_by_id(X, sp_set.train_idxs.tolist()[0])
    X_test = get_doc_by_id(X, sp_set.test_idxs.tolist()[0])
    X_val = get_doc_by_id(X, sp_set.val_idxs.tolist()[0])

    y_train = labels[sp_set.train_idxs.iloc[0]]
    y_test = labels[sp_set.test_idxs.iloc[0]]
    y_val = labels[sp_set.val_idxs.iloc[0]]

    if fast_test == True:

        ss = min(sample_size, len(X_val))
        ids = np.arange(ss)
        size_s = ss - 1
        random_idxs = np.random.choice(ids, size_s, replace=False)
        X_train = get_doc_by_id(X_train, random_idxs)
        X_test = get_doc_by_id(X_test, random_idxs)
        X_val = get_doc_by_id(X_val, random_idxs)

        y_train = y_train[random_idxs]
        y_test = y_test[random_idxs]
        y_val = y_val[random_idxs]

    return X_train, X_test, X_val, y_train, y_test, y_val, split_settings


# In[4]:


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


class BERT():

    def __init__(
        self,
        batch_size=8,
        limit_k=20,
        max_epochs=5,
        lr=5e-5,
        max_length=256,
        limit_patient=3,
        model_path="bert-base-uncased",
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
        self.model_path = model_path
        self.padding = padding
        self.truncation = truncation
        self.seed = seed
        self.load_model = load_model

    def encode_data(self, X_train, X_test, X_val, y_train, y_test, y_val):

        # Codifica o texto.
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        train_encodings = tokenizer(X_train, max_length=self.max_length,
                                    return_tensors="pt",  padding=self.padding, truncation=self.truncation)
        val_encodings = tokenizer(X_val, max_length=self.max_length,
                                  return_tensors="pt",  padding=self.padding, truncation=self.truncation)
        test_encodings = tokenizer(X_test, max_length=self.max_length,
                                   return_tensors="pt",  padding=self.padding, truncation=self.truncation)

        # Converte o dado para o formato DataLoader.
        train_dataset = CustomDataset(train_encodings, y_train)
        val_dataset = CustomDataset(val_encodings, y_val)
        test_dataset = CustomDataset(test_encodings, y_test)

        # Coloca o processo em batch
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.batch_size, worker_init_fn=self.seed)
        eval_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, worker_init_fn=self.seed)
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, worker_init_fn=self.seed)

        return train_dataloader, eval_dataloader, test_dataloader

    def fit_predict(self, X_train, X_test, X_val, y_train, y_test, y_val):

        # Preparado dataloaders.
        train_dataloader, eval_dataloader, test_dataloader = self.encode_data(
            X_train, X_test, X_val, y_train, y_test, y_val)
        num_labels = np.unique(np.hstack([y_train, y_test, y_val])).shape[0]

        # Define o modelo de classificação
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=num_labels)
        optimizer = AdamW(model.parameters(), lr=self.lr)
        num_training_steps = self.max_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        # Coloca para o processamento ser feito na GPU
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # Treina o modelo
        progress_bar = tqdm(range(num_training_steps))
        cont_patient = 0
        min_loss_eval = 10000

        # Para cada época.
        for epoch in range(self.max_epochs):
            model.train()
            # Para cada batch.
            for batch in train_dataloader:
                progress_bar.update(1)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # validação
            y_pred_list = []
            y_true_list = []

            model.eval()  # define para não atualizar pesos na validação
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.no_grad():  # define para não atualizar pesos na validação
                    outputs = model(**batch)

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

            # armazena as metricas a partir das predicoes
            loss_eval_atual = loss.item()

            # parar de treinar se não houver melhoria
            if loss_eval_atual < min_loss_eval:
                cont_patient = 0
                min_loss_eval = loss_eval_atual
            else:
                cont_patient += 1

            if cont_patient >= self.limit_patient:
                break

        # ---------- TESTE ---------- #
        probs = []
        model.eval()
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                norm = softmax(outputs.logits, dim=-1).tolist()
                probs.append(norm)

        probs = np.vstack(probs)
        return probs, model


# In[5]:


def get_train_probas(X, y, base_path, n_splits=4):

    # Train probas.
    sfk = StratifiedKFold(n_splits=n_splits)
    sfk.get_n_splits(X, y)
    alig_idx = np.arange(y.shape[0])
    idx_list = []
    probas = []
    for fold, (train_index, test_index) in enumerate(sfk.split(X, y)):

        Xt = get_doc_by_id(X, train_index)
        Yt = y[train_index]

        X_train, X_val, y_train, y_val = train_test_split(
            Xt, Yt, test_size=0.1)

        X_test = get_doc_by_id(X, test_index)
        y_test = y[test_index]

        idx_list.append(alig_idx[test_index])

        # Applying oversampling when it is needed.
        for c in set(y_test) - set(y_train):

            sintetic = "fake document"
            X_train.append(sintetic)
            y_train = np.hstack([y_train, [c]])

        bert_model = BERT()
        p, _ = bert_model.fit_predict(
            X_train, X_test, X_val, y_train, y_test, y_val)
        probas.append(p)
        scoring = {}
        y_pred = p.argmax(axis=1)
        scoring["macro"] = f1_score(y_test, y_pred, average="macro")
        scoring["micro"] = f1_score(y_test, y_pred, average="micro")
        print(
            f"\t\tFOLD {fold} - Macro: {scoring['macro']} - Micro {scoring['micro']}")
        torch.cuda.empty_cache()

    probas = np.vstack(probas)
    sorted_idxs = np.hstack(idx_list).argsort()
    probas = probas[sorted_idxs]
    probas_path = f"{base_path}/train"
    np.savez(probas_path, X_train=probas)


# In[6]:


SEED = 42

# Hiperparametros do modelo
fast_test = False
sample_size = 400

# hyper-parameters
batch_size = 8
limit_k = 20
max_epochs = 5
lr = 5e-5
max_length = 256
limit_patient = 3

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(seed=SEED)

datasets = ["reut"]

# Trian probas parameters.

train_n_splits = 4


# In[7]:


for dataset in datasets:
    for fold in np.arange(10):
        print(f"{dataset.upper()} - FOLD: {fold}")

        # Separando o dataset.
        X_train, X_test, X_val, y_train, y_test, y_val, sp_settings = data_process(
            dataset, fold, fast_test=fast_test, sample_size=sample_size)

        base_path = f"/home/welton/data/clfs_output/split_10/{dataset}/10_folds/rep_bert/{fold}"
        os.makedirs(base_path, exist_ok=True)
        
        test_path = f"{base_path}/test.npz"
        # Se este fold ainda não foi executado.        
        if not os.path.exists(test_path):
            # Configurando os parâmetros do modelo.
            bert_model = BERT(
                batch_size=batch_size,
                limit_k=limit_k,
                max_epochs=max_epochs,
                lr=lr,
                max_length=max_length,
                limit_patient=limit_patient,
                padding=True,
                truncation=True,
                seed=SEED)
            # Treinando e realizando predições com o modelo.
            probs, model = bert_model.fit_predict(
                X_train, X_test, X_val, y_train, y_test, y_val)
            np.savez(test_path, X_test=probs)
            model_path = f"{base_path}/model"
            torch.save(model, model_path)
            y_pred = probs.argmax(axis=1)
            print(f"Macro: {f1_score(y_test, y_pred, average='macro')}")
            print(f"Macro: {f1_score(y_test, y_pred, average='micro')}")
            torch.cuda.empty_cache()

        
        train_path = f"{base_path}/train.npz"
        # Se as probabilidades do treino não foram geradas.
        if not os.path.exists(train_path):
            sort = np.array(
                sp_settings.iloc[fold]["train_idxs"] + sp_settings.iloc[fold]["val_idxs"]).argsort()
            get_train_probas(get_doc_by_id(X_train + X_val, sort),
                         np.hstack([y_train, y_val])[sort], base_path)
            #get_train_probas(X_train + X_test + X_val, np.hstack([y_train, y_test, y_val]), base_path)


# In[ ]:
