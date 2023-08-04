import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from src.aws.awsio import store_json_in_aws

def report_scoring(y_test: np.ndarray,
                   y_pred: np.ndarray,
                   output_dir: str,
                   fold: int = -1):

    scoring = {}
    scoring["macro"] = f1_score(y_test, y_pred, average="macro")
    scoring["micro"] = f1_score(y_test, y_pred, average="micro")
    scoring["accuracy"] = accuracy_score(y_test, y_pred)

    if fold == -1:
        print(f"\n\tMacro: {scoring['macro'] * 100}")
        print(f"\tMicro: {scoring['micro'] * 100}\n")
    else:
        print(f"\t\tFOLD {fold} - Macro: {scoring['macro']}")  # , end='')


    #with open(f"{output_dir}/scoring.json", "w") as fd:
    #    json.dump(scoring, fd)

    store_json_in_aws(f"{output_dir}/scoring.json",
                      scoring)
