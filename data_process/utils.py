import io
import numpy as np

def read_texts(txt_file: str):
    with io.open(txt_file, errors='ignore', newline="\n") as fp:
        texts = fp.readlines()
    return texts

def fix_labels(labels):

    if np.min(labels) > 0:
        labels = labels - 1
        return labels
    return labels