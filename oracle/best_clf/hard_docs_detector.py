import os
import json
import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from utils import load_labels, load_probas, load_meta_features, save_pickle


def make_labels(clf_set: list,
                probas: dict,
                y_true: np.ndarray,
                train_test: str):

    new_labels, hits_counts = [], []
    for doc_idx in np.arange(probas[clf_set[0][0]][train_test].shape[0]):
        n_hits = 0
        for clf, _ in CLFS_SET:
            pred = probas[clf][train_test][doc_idx].argmax()
            if pred == y_true[doc_idx]:
                n_hits += 1
        if n_hits < 4:
            new_labels.append(1)
        else:
            new_labels.append(0)

        hits_counts.append(n_hits)

    return new_labels, hits_counts


if __name__ == "__main__":

    with open("data/hard_docs_detector.json", "rb") as fd:
        settings = json.load(fd)

    DATA_SOURCE = settings["DATA_SOURCE"]
    DATASETS = settings["DATASETS"]
    CLFS_SET = settings["CLFS_SET"]
    FEATURES_SET = settings["FEATURES_SET"]

    CLFS_SET.sort(key=lambda x: x[0])
    FEATURES_SET.sort()
    idx_to_clf = {idx: clf for idx, (clf, _) in enumerate(CLFS_SET)}

    clf_sufix = '/'.join([f"{c[0]}_{c[1]}" for c in CLFS_SET])
    mf_sufix = '_'.join(FEATURES_SET)

    STACKING_DIR = f"{DATA_SOURCE}/stacking/stacking_output"

    for dataset, n_folds in DATASETS:
        metrics = {}
        for fold in np.arange(n_folds):

            probas = load_probas(DATA_SOURCE, dataset, CLFS_SET, fold, n_folds)
            y_train, y_test = load_labels(DATA_SOURCE, dataset, fold, n_folds)

            n_classes = probas[CLFS_SET[0][0]]["train"].shape[1]
            
            y_train, train_counts = make_labels(CLFS_SET, probas, y_train, "train")
            y_test, test_counts = make_labels(CLFS_SET, probas, y_test, "test")
            
            X_train, X_test = load_meta_features(DATA_SOURCE,
                                                 dataset,
                                                 n_classes,
                                                 CLFS_SET,
                                                 clf_sufix,
                                                 fold,
                                                 FEATURES_SET,
                                                 mf_sufix,
                                                 n_folds)

            clf = RandomForestClassifier(n_estimators=300,
                                         max_depth=8,
                                         class_weight="balanced_subsample",
                                         n_jobs=60,
                                         random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            macro = f1_score(y_test, y_pred, average="macro")
            full_macro = f1_score(y_test, y_pred, average=None)

            metrics[fold] = {
                "avg_macro": macro,
                "full_macro": full_macro,
                "train_labels": y_train,
                "test_labels": y_test,
                "train_counts": train_counts,
                "test_counts": test_counts,
                "y_pred": y_pred,
                "importances": clf.feature_importances_
            }
            print(f"\tDATASET: {dataset.upper()} / FOLD: {fold} - AvgMacro: {macro} - FullMacro: {full_macro}")

            output_dir = f"{DATA_SOURCE}/oracle/hard_docs/{dataset}/{clf_sufix}/{mf_sufix}/fold_{fold}"
            print(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            save_pickle(metrics[fold], f"{output_dir}/output.pkl")
            dump(clf, f"{output_dir}/model.joblib")

    
        scores = []
        for fold in metrics:
            y_pred = metrics[fold]["y_pred"]
            y_test = metrics[fold]["test_labels"]
            scores.append([
                precision_score(y_test, y_pred),
                recall_score(y_test, y_pred),
                f1_score(y_test, y_pred, average="binary")]
            )
        prec, rec, mac = np.mean(scores, axis=0).reshape(-1).tolist()
        print(f"DATASET: {dataset.upper()} - Prec: {prec} - Rec: {rec} - Mac: {mac}")
