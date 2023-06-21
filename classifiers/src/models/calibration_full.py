import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

from src.models.models import get_classifier

def probabilities_calibration(estimator_name: str,
                estimator_params: dict,
                X_train: np.ndarray,
                y_train: np.ndarray,
                calib_method: str,
                n_splits: int):
    
    estimator, _ = get_classifier(estimator_name)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    calibrated_model = CalibratedClassifierCV(estimator,
                                              cv=cv,
                                              method=calib_method)
    estimator.set_params(**estimator_params)
    calibrated_model.fit(X_train, y_train)

    return calibrated_model
