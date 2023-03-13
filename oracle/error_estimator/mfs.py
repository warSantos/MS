import numpy as np
from collections import Counter
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler

from sys import path
path.append('../utils')
from local_utils import build_clf_beans


def agreement_mfs(probas, clf_target, fold, train_test):

    main_preds = probas[clf_target][fold][train_test].argmax(axis=1)
    preds = [probas[clf][fold][train_test].argmax(
        axis=1) for clf in probas if clf != clf_target]
    preds = np.vstack(preds).T

    div = []
    agree_sizes = []
    num_classes = []
    pred_entropy = []
    # For each document.
    for idx in np.arange(main_preds.shape[0]):
        counts = Counter(preds[idx])
        pred_class, agree_size = counts.most_common()[0]
        if pred_class == main_preds[idx]:
            div.append(0)
        else:
            div.append(1)
        agree_sizes.append(agree_size)
        total = len(counts)
        num_classes.append(total)
        pred_entropy.append(entropy([ counts[k]/total for k in counts ]))

    return np.array(div), np.array(agree_sizes), np.array(num_classes), np.array(pred_entropy)

def confidence_rate(probas, labels):

    conf_hit = []
    conf_freq, hits = build_clf_beans(probas, labels)
    hits_rate = {np.trunc(bean*10)/10: hits[bean] / conf_freq[bean]
                 if bean in hits else 0 for bean in np.arange(0, 1, 0.1)}
    preds = probas.argmax(axis=1)
    for idx, predicted_class in enumerate(preds):
        # Getting the probability of the predicted class
        probability = probas[idx][predicted_class] * 10
        bean = np.trunc(probability) / 10
        bean = 0.9 if bean >= 1 else bean
        conf_hit.append(hits_rate[bean])
    return np.array(conf_hit)


def hits_rate_by_class(probas, labels):

    class_hits_rate = {}
    preds = probas.argmax(axis=1)
    # Vector with hits and misses.
    hits = preds == labels
    # For each label.
    for label in np.unique(labels):
        # Get the docs of the label.
        class_docs = labels == label
        class_hits_rate[label] = np.sum(hits[class_docs]) / np.sum(class_docs)
    return np.array([class_hits_rate[p] for p in preds])


def class_weights(probas, labels):

    cw = {label: np.sum(labels == label) /
          labels.shape[0] for label in np.unique(labels)}
    preds = probas.argmax(axis=1)
    return np.array([cw[p] for p in preds])

def make_mfs(data_source: str,
             dataset: str,
             clf: str,
             probas: dict,
             y_train: np.ndarray,
             y_test: np.ndarray,
             fold: int,
             meta_features_set: list):

    pack_meta_features = {
        "train": [],
        "test": []
    }
    
    probas_train = probas[clf][fold]["train"]
    probas_test = probas[clf][fold]["test"]
    
    
    if "probas_based" in meta_features_set:
        # Building Meta-Features.
        cw_train = class_weights(probas_train, y_train)
        cw_test = class_weights(probas_test, y_test)
        hrc_train = hits_rate_by_class(probas_train, y_train)
        hrc_test = hits_rate_by_class(probas_test, y_test)
        conf_train = confidence_rate(probas_train, y_train)
        conf_test = confidence_rate(probas_test, y_test)
        div_train, ags_train, num_classes_train, entropy_train = agreement_mfs(probas, clf, fold, "train")
        div_test, ags_test, num_classes_test, entropy_test = agreement_mfs(probas, clf, fold, "test")
        
        scaled_ags_train = MinMaxScaler().fit_transform(
            ags_train.reshape(-1, 1)).reshape(-1)
        scaled_ags_test = MinMaxScaler().fit_transform(
            ags_test.reshape(-1, 1)).reshape(-1)
        scaled_num_classes_train = MinMaxScaler().fit_transform(
            num_classes_train.reshape(-1, 1)).reshape(-1)
        scaled_num_classes_test = MinMaxScaler().fit_transform(
            num_classes_test.reshape(-1, 1)).reshape(-1)
            
        # Joining Meta-Features.
        probas_based_train = np.vstack([
            cw_train,
            hrc_train,
            conf_train,
            div_train,
            scaled_ags_train,
            scaled_num_classes_train,
            entropy_train
        ]).T

        probas_based_test = np.vstack([
            cw_test,
            hrc_test,
            conf_test,
            div_test,
            scaled_ags_test,
            scaled_num_classes_test,
            entropy_test
        ]).T
        
        pack_meta_features["train"].append( probas_based_train )
        pack_meta_features["test"].append( probas_based_test )

    elif "probas" in meta_features_set:
        pack_meta_features["train"].append( probas_train )
        pack_meta_features["test"].append( probas_test )

    elif "dist" in meta_features_set:

        # Load fold Meta-Features (Washington).
        dist_train = csr_matrix(np.load(
            f"{data_source}/meta_features/features/dist/{fold}/{dataset}/train.npz")["X_train"]).toarray()
        dist_test = csr_matrix(np.load(
            f"{data_source}/meta_features/features/dist/{fold}/{dataset}/test.npz")["X_test"]).toarray()

        pack_meta_features["train"].append(dist_train)
        pack_meta_features["test"].append(dist_test)
    
    X_train = np.hstack(pack_meta_features["train"])
    X_test = np.hstack(pack_meta_features["test"])
    
    # Making labels (hit or missed)
    preds_train = probas_train.argmax(axis=1)
    upper_train = np.zeros(preds_train.shape[0])
    upper_train[preds_train == y_train] = 1

    preds_test = probas_test.argmax(axis=1)
    upper_test = np.zeros(preds_test.shape[0])
    upper_test[preds_test == y_test] = 1

    return X_train, X_test, upper_train, upper_test