import os
import json
import numpy as np

from utils import *

if __name__ == "__main__":

    with open("data/settings.json", 'r') as fd:
        settings = json.load(fd)

    DATA_SOURCE = settings["DATA_SOURCE"]
    DATASETS = settings["DATASETS"]
    mf_set = settings["MF_SET"]
    features_set = settings["FEATURES_SET"]
    features_set.sort()
    mf_sufix = "_".join(features_set)
    meta_layer = settings["META_LAYER"]
    label_type = settings["LABEL_TYPE"]
    CLF_SET = settings["CLFS_SET"]
    CLF_SET.sort(key=lambda c: c[0])

    clf_sufix = '/'.join([f"{c[0]}_{c[1]}" for c in CLF_SET])

    labels = load_labels(DATA_SOURCE, DATASETS)
    probas = load_probas(DATA_SOURCE, DATASETS, CLF_SET)

    for dataset, n_folds in DATASETS:
        scores = []
        n_classes = len(set(labels[dataset]["train"][0]))
        for fold in np.arange(n_folds):

            upper_train, missed_train = get_new_labels(meta_layer,
                                                       dataset,
                                                       CLF_SET,
                                                       probas,
                                                       labels,
                                                       fold,
                                                       "train",
                                                       label_type)
            upper_test, missed_test = get_new_labels(meta_layer,
                                                     dataset,
                                                     CLF_SET,
                                                     probas,
                                                     labels,
                                                     fold,
                                                     "test",
                                                     label_type)

            # Loading meta-features.
            X_train, X_test = load_meta_features(DATA_SOURCE,
                                                 dataset,
                                                 n_classes,
                                                 CLF_SET,
                                                 clf_sufix,
                                                 fold,
                                                 features_set,
                                                 mf_set,
                                                 n_folds)

            y_test = labels[dataset]["test"][fold]
            y_train = labels[dataset]["train"][fold]

            n_docs = labels[dataset]["test"][fold].shape[0]
            if meta_layer == "upper_bound":
                
                # Making output dir.
                output_dir = f"{DATA_SOURCE}/stacking/stacking_output/{dataset}/{n_folds}_folds/{meta_layer}/best_clf/{clf_sufix}/{mf_sufix}/fold_{fold}"
                os.makedirs(output_dir, exist_ok=True)

                y_pred = predict(dataset, CLF_SET, probas, fold, upper_test)
                np.savez(f"{output_dir}/output", upper_test=upper_test)
            elif meta_layer == "multi_output":
                
                # Making output dir.
                output_dir = f"{DATA_SOURCE}/stacking/stacking_output/{dataset}/{n_folds}_folds/{meta_layer}/{clf_sufix}/{mf_sufix}/fold_{fold}"
                print(f"\t{output_dir}")
                os.makedirs(output_dir, exist_ok=True)

                model = get_multi_output_clf()
                _ = model.fit(X_train, upper_train)
                multi_best_clf_est = model.predict_proba(X_test)
                y_pred, best_clf_est = multi_output_predict(dataset,
                                              CLF_SET,
                                              probas,
                                              n_docs,
                                              fold,
                                              multi_best_clf_est)
                
                np.savez(f"{output_dir}/output",
                         upper_test=upper_test,
                         multi_best_clf_est=multi_best_clf_est,
                         best_est=best_clf_est,
                         y_test=y_test,
                         y_train=y_train,
                         y_pred=y_pred,
                         missed_train=missed_train,
                         missed_test=missed_test)

            else:

                # Making output dir.
                output_dir = f"{DATA_SOURCE}/stacking/stacking_output/{dataset}/{n_folds}_folds/{meta_layer}/{label_type}/{clf_sufix}/{mf_sufix}/fold_{fold}"
                os.makedirs(output_dir, exist_ok=True)

                model = get_clf(meta_layer)
                _ = model.fit(X_train, upper_train)
                best_clf_est = model.predict(X_test).tolist()
                y_pred = predict(dataset,
                                 CLF_SET,
                                 probas,
                                 n_docs,
                                 fold,
                                 best_clf_est)

                np.savez(f"{output_dir}/output",
                         upper_test=upper_test,
                         upper_train=upper_train,
                         best_est=best_clf_est,
                         y_test=y_test,
                         y_train=y_train,
                         y_pred=y_pred,
                         missed_train=missed_train,
                         missed_test=missed_test)

            print(f"[ \t{dataset.upper()} / {label_type} / {mf_sufix} ] FOLD: {fold}")

            scores.append(report_scoring(y_test, y_pred, output_dir))

        print(f"{dataset.upper()} - MeanMacro: {np.mean(scores)}")
