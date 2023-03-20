import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from src.models.models import get_classifier

def probabilities_calibration(estimator: object,
                X_train: np.ndarray,
                y_train: np.ndarray,
                calib_method: str):
    
    calibrated_model = CalibratedClassifierCV(estimator, cv='prefit', method=calib_method)
    calibrated_model.fit(X_train, y_train)

    return calibrated_model
