
import json

# Local libraries
from src.dist_mfs.dist import DistMFs

    
if __name__ == "__main__":


    with open("data/settings/settings.json", 'r') as fd:
        settings = json.load(fd)

    labels_dir = settings["LABELS_DIR"]
    output_dir = settings["OUTPUT_DIR"]
    datasets = settings["DATASETS"]
    n_folds = settings["N_FOLDS"]
    n_sub_folds = settings["N_SUB_FOLDS"]
    folds = settings["FOLDS"]

    mf = DistMFs()
    mf.build(labels_dir,
             output_dir,
             datasets,
             folds,
             n_folds,
             n_sub_folds)