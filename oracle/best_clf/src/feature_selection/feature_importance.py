import os
from sys import path
import numpy as np
from sklearn.ensemble import RandomForestClassifier

path.append("../")

from src.optimization import execute_optimization

RANDOM_STATE = 42

class FeatureSelector:
    
    def feature_importance(self, settings_path, X, y, n_feats=100, load_prep=True):
        
        base_dir = f"../../data/feature_selection/{settings_path}"
        fr_path = f"{base_dir}/feature_ranking"
        
        if load_prep and not os.path.exists(fr_path+".npy"):

            print("\tSELECTING FEATURES FROM SCRATCH.")

            model_path = f"{base_dir}/model.joblib"
        
            os.makedirs(base_dir, exist_ok=True)
            opt = execute_optimization(
                "random_forest",
                model_path,
                X,
                y,
                opt_n_jobs=10
            )

            bp = opt.best_params_
            { key.replace("classifier__", ''):bp[key] for key in bp }

            rf = RandomForestClassifier()
            for key in bp:
                setattr(rf, key, bp[key])
            
            rf.fit(X, y)

            fi = rf.feature_importances_
            sort = (-fi).argsort()

            np.save(fr_path, sort)
        else:
            
            print("\tLOADING PRE SELECTED FEATURES.")

            sort = np.load(fr_path+".npy")

        return sort[:n_feats]
