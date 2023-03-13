import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def get_scores(y_test, y_pred):

    pc = precision_score(y_test, y_pred, pos_label=0) * 100
    rc = recall_score(y_test, y_pred, pos_label=0) * 100
    f1 = f1_score(y_test, y_pred, pos_label=0) * 100
    acc = accuracy_score(y_test, y_pred) * 100
    return pc, rc, f1, acc


def get_dict_score(pc, rc, f1, acc):

    d = {
        "f1_macro": f1,
        "precision": pc,
        "recall": rc,
        "accuracy": acc
    }
    return d


def save_scores(output_dir, scores):

    with open(f"{output_dir}/scoring.json", "w") as fd:
        json.dump(scores, fd)
