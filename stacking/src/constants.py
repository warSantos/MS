#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Constants use in all codes to keep a clean format of input/output."""

# dataset_name / num_classes
DATASETS_INFOS = {
    "20ng": 20,
    "acm": 11,
    "agnews": 4,
    "imdb_reviews": 10,
    "reut": 90,
    "sogou": 5,
    "yahoo": 10,
    "yelp_2015": 5,
    "webkb": 7
}

IDS_REPRESENTATIONS = {
    "fast_text_mf_knn": "fmk",
    "fast_text_raw": "fr",
    "pte_mf_knn": "pmk",
    "pte_raw": "pr",
    "tfidf_mf_knn": "tmk",
    "tfidf_raw": "tr"
}

IDS_MODELS = {
    "linear_svm/fast_text_1/raw_folds": "sfr",
    "linear_svm/pte_1/raw_folds": "spr",
    "linear_svm/tf_idf_1/fs": "str",
    "linear_svm/tf_idf_1/meta_features_1/knn_cos": "stmk",
    "knn/fast_text_1/raw_folds": "kfr",
    "knn/pte_1/raw_folds": "kpr",
    "knn/tf_idf_1/fs": "ktr",
    "knn/tf_idf_1/meta_features_1/knn_cos": "ktmk",
    "lr/fast_text_1/raw_folds": "lfr",
    "lr/pte_1/raw_folds": "lpr",
    "lr/tf_idf_1/fs": "ltr",
    "lr/tf_idf_1/meta_features_1/knn_cos": "ltmk",
    "xgboost/fast_text_1/raw_folds": "xfr",
    "xgboost/pte_1/raw_folds": "xpr",
    "xgboost/tf_idf_1/fs": "xtr",
    "xgboost/tf_idf_1/meta_features_1/knn_cos": "xtmk",
    "xlnet": "xlnet",
    "bert": "bert"
}

REP_CLFS = ["rep_bert", "sfr", "lpr", "ltr", "xtmk"]
MIX_REPS_MINUS_BERT = ["sfr", "lpr", "ltr", "xtmk"]
NEW_CLFS = ["kpr", "ktr", "lpr", "ltr", "sfr", "stmk", "xfr", "xpr", "xtr", "kfr", "ktmk", "lfr", "ltmk", "spr", "str", "xlnet_softmax", "xtmk", "rep_bert"]
PARTIAL_STACKING = ["kfr", "kpr", "ktmk", "ktr", "lfr", "lpr", "ltmk", "rep_bert"]