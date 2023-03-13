import numpy as np
from typing import Tuple
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator

from optimization import execute_optimization, get_clf

def parse_params(params: dict) -> dict:

    return {
        p_name.replace("classifier__", ''): params[p_name]
        for p_name in params if p_name.find("classifier__") > -1
    }


def get_f1(X_train: np.ndarray,
           X_test: np.ndarray,
           y_train: np.ndarray,
           y_test: np.ndarray,
           clf: str,
           params: dict) -> Tuple[np.float64, BaseEstimator]:

    estimator, _, default_hyp = get_clf(clf)
    # If the estimator was previous optmized.
    if len(params) > 0:
        best_params = params
    else:
        best_params = default_hyp

    estimator.set_params(**best_params)

    # Training estimator.
    _ = estimator.fit(X_train, y_train)

    # Evaluating estimator's performance.
    y_pred = estimator.predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label=0)

    return (f1, estimator)


def fast_feature_selection(X_train: np.ndarray,
                           X_test: np.ndarray,
                           y_train: np.ndarray,
                           y_test: np.ndarray,
                           name_error_est: str,
                           file_model: str,
                           do_optimization: bool,
                           n_feats=30) -> Tuple[int, np.float64, BaseEstimator, BaseEstimator]:

    # Training RF and building feature importance ranking.
    if do_optimization:
        opt = execute_optimization("rf",
                                      f"{file_model}_opt",
                                      X_train,
                                      y_train,
                                      load_model=True).best_estimator_
        best_params = parse_params(opt.get_params())
        forest, _, _ = get_clf("rf")
        forest.set_params(**best_params)
    else:
        best_params = {}
        forest, _, default_hyp = get_clf("rf")
        forest.set_params(**default_hyp)

    _ = forest.fit(X_train, y_train)

    importances = forest.feature_importances_
    ranking = (1 - importances).argsort()

    feats_ids = ranking[:n_feats]
    f1, estimator = get_f1(X_train[:, feats_ids], 
                           X_test[:, feats_ids],
                           y_train, 
                           y_test, 
                           name_error_est,
                           best_params)

    return (n_feats, f1, forest, estimator)


def feature_selection(X_train, X_test, y_train, y_test, name_error_est):

    # Training RF and building feature importance ranking.
    forest = RandomForestClassifier(random_state=42, n_jobs=25)
    _ = forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    ranking = (1 - importances).argsort()
    best_model = None

    pick = 20
    gap = 20
    best_f1 = -1
    best_pos = -1
    # se é para subir a busca ou descer.
    improve = True

    # Enquanto houver pontos para busca.
    while True:
        # Teste o modelo com 'pick' features (Aqui pode ser o XGBoost na GPU).
        feats_ids = ranking[:pick]
        f1, xgb = get_f1(X_train[:, feats_ids], X_test[:, feats_ids],
                         y_train, y_test, name_error_est)
        # Se a macro de agora for melhor que a última.
        if best_f1 < f1:
            best_f1 = f1
            best_pos = pick
            improve = True
            best_model = xgb
        # Se a macro de agora não for melhor que a última encurte o salto.
        else:
            improve = False
            gap = max(gap // 2, 1)
            improve = False
        pick = best_pos + gap

        if gap == 1 and not improve:
            break

    gap = 20
    improve = True
    pick = best_pos - gap + 1
    first_best_pos = best_pos
    # Enquanto houver pontos para busca.
    while pick < first_best_pos:
        # Teste o modelo com 'pick' features (Aqui pode ser o XGBoost na GPU).
        feats_ids = ranking[:pick]
        f1, xgb = get_f1(X_train[:, feats_ids.tolist()], X_test[:, feats_ids.tolist(
        )], y_train, y_test, name_error_est)
        # Se a macro de agora for melhor que a última.
        if best_f1 < f1:
            best_f1 = f1
            best_pos = pick
            improve = True
            best_model = xgb
        # Se a macro de agora não for melhor que a última encurte o salto.
        else:
            improve = False
            gap = gap // 2
            improve = False

        pick = first_best_pos - gap
    print(f"\tF1: {best_f1} POS: {best_pos} PICK: {pick}")

    return best_pos, forest, best_model
