import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from src.models.models import get_classifier

def probabilities_calibration(model: object,
                X_train: np.ndarray,
                y_train: np.ndarray,
                model_name: str,
                n_jobs: int = -1):

    # Getting the best parameters.
    pset = {}
    best_params = model.best_estimator_.get_params()
    for param in best_params:
        if param.find("classifier__") > -1:
            param_name = param.split("__")[-1]
            pset[param_name] = best_params[param]

    clf, _ = get_classifier(model_name, n_jobs=n_jobs)
    clf.set_params(**pset)

    calibrated_model = CalibratedClassifierCV(clf, cv=4, method="sigmoid")
    calibrated_model.fit(X_train, y_train)

    return calibrated_model
