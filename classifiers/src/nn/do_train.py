import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.special import softmax

from src.nn.data_loader import Loader
from src.nn.data_loader import get_doc_by_id
from src.nn.model import Transformer, TextFormater, FitHelper
from src.aws.awsio import store_nparrays_in_aws, load_file_from_aws

def set_data_splits(X,
                    y,
                    train_index,
                    test_index):

    # Selecting train documents and labels.
    X_train = get_doc_by_id(X, train_index)
    y_train = y[train_index]

    # Spliting the train documents in train and validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1)

    # Getting test documents.
    X_test = get_doc_by_id(X, test_index)
    y_test = y[test_index]

    # Applying oversampling when it is needed. Sometimes a class doesn't have
    # enough documents from a class.
    for c in set(y_test) - set(y_train):

        sintetic = "fake document"
        X_train.append(sintetic)
        y_train = np.hstack([y_train, [c]])

    return X_train, X_test, X_val, y_train, y_test, y_val

def get_train_probas(data_handler: Loader,
                     output_dir: str,
                     parent_fold: int,
                     n_splits: int,
                     model_params: dict,
                     text_params: dict):

    # Loading the train set (Train-Test split without val).
    X, y = data_handler.get_X_y(parent_fold, "train", with_val=False)
    # This list will hold all the folds probabilities.
    probas = []
    # This list holds all the document's indexes to re sort later when fine-tuning is done.
    idx_list = []
    align_idx = np.arange(y.shape[0])
    # For each fold.
    for fold in np.arange(n_splits):
        
        pickle_path = f"{data_handler.data_dir}/{data_handler.dataset}/splits/sub_splits/{parent_fold}/split_4.pkl"
        data = load_file_from_aws(pickle_path)
        sub_split = pd.read_pickle(data)

        train_index = sub_split.iloc[fold].train_idxs
        test_index = sub_split.iloc[fold].test_idxs

        # Spliting data.
        X_train, X_test, X_val, y_train, y_test, y_val = set_data_splits(X,
                                                                         y,
                                                                         train_index,
                                                                         test_index)
        idx_list.append(align_idx[test_index])
        
        # Enconding train and val into input ids.
        text_formater = TextFormater(**text_params)
        train = text_formater.prepare_data(X_train, y_train, shuffle=True)
        val = text_formater.prepare_data(X_val, y_val)

        # Setting model parameters.
        model_params["len_data_loader"] = len(train)
        model = Transformer(**model_params)

        # Training model.
        fitter = FitHelper()
        trainer = fitter.fit(model, train, val, model.max_epochs, model.seed)
        
        # Enconding test.
        test = text_formater.prepare_data(X_test, y_test)
        
        # Predicting.
        trainer.predict(model, test)
        test_l = fitter.load_logits_batches()
        trainer.predict(model, val)
        eval_l = fitter.load_logits_batches()

        # Saving validation and test logits.
        subfold_path = f"{output_dir}/sub_fold/{fold}"
        os.makedirs(subfold_path, exist_ok=True)
        store_nparrays_in_aws(f"{subfold_path}/eval_logits.npz", {"X_eval": eval_l,
                                                              "y_eval": y_val})
        store_nparrays_in_aws(f"{subfold_path}/test_logits.npz", {"X_test": test_l,
                                                              "y_test": y_test})
        # Saving fold document's indexes.
        store_nparrays_in_aws(f"{subfold_path}/align.npz", {"align": idx_list[-1]})

        # Saving fold's probabilities.
        probas.append(softmax(test_l, axis=1))
        scoring = {}

        # Printing model's performance.
        y_pred = test_l.argmax(axis=1)
        scoring["macro"] = f1_score(y_test, y_pred, average="macro")
        scoring["micro"] = f1_score(y_test, y_pred, average="micro")
        print(
            f"\t\tSUB-FOLD {fold} - Macro: {scoring['macro']} - Micro {scoring['micro']}")

        os.system("rm -rf lightning_logs")

    # Joining folds probabilities.
    probas = np.vstack(probas)

    # Resorting document's probabilities.
    sorted_idxs = np.hstack(idx_list).argsort()
    probas = probas[sorted_idxs]
    probas_path = f"{output_dir}/train.npz"
    # Saving document's proabilities.
    store_nparrays_in_aws(probas_path, {"X_train": probas, "y_train": y})