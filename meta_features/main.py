
import json

# Local libraries
from src.dist_mfs.dist import DistMFs

    
if __name__ == "__main__":


    with open("data/settings/settings.json", 'r') as fd:
        settings = json.load(fd)

    reps_dir = settings["REPS_DIR"]
    input_rep = settings["INPUT_REP"]
    output_rep = settings["OUTPUT_REP"]
    datasets = settings["DATASETS"]
    n_sub_folds = settings["N_SUB_FOLDS"]
    

    mf = DistMFs()
    mf.build(reps_dir,
             input_rep,
             output_rep,
             datasets,
             n_sub_folds)