import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class Parameters:

    def __init__(self,
        device = 0,
        buckets = 10,
        temperature = 1,
        label_smoothing = 0,
        do_train = False,
        do_evaluate = False,
        data_path = "/home/welton/data",
        logits_path = "logits/clfs_output/split_10/webkb/10_folds/rep_bert",
        base_output = "calibrated_probabilities/split_10/webkb/10_folds/rep_bert"
        ):

        self.device = device
        self.buckets = buckets
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.do_train = do_train
        self.do_evaluate = do_evaluate
        self.data_path = data_path
        self.logits_path = f"{data_path}/{logits_path}"
        self.base_output = f"{data_path}/{base_output}"


def load_output(logits_path, train_test):
    """Loads output file, wraps elements in tensor."""

    """
    with open(path) as f:
        elems = [json.loads(l.rstrip()) for l in f]
        for elem in elems:
            elem['true'] = torch.tensor(elem['true']).long()
            elem['logits'] = torch.tensor(elem['logits']).float()
        return elems
    """

    loader = np.load(logits_path)
    labels, logits = loader[f"y_{train_test}"], loader[f"X_{train_test}"]

    elem = {}
    elem["true"] = torch.from_numpy(labels)
    elem["logits"] = torch.from_numpy(logits)
    return elem


def get_bucket_scores(y_score):
    """
    Organizes real-valued posterior probabilities into buckets.
    For example, if we have 10 buckets, the probabilities 0.0, 0.1,
    0.2 are placed into buckets 0 (0.0 <= p < 0.1), 1 (0.1 <= p < 0.2),
    and 2 (0.2 <= p < 0.3), respectively.
    """

    bucket_values = [[] for _ in range(args.buckets)]
    bucket_indices = [[] for _ in range(args.buckets)]
    for i, score in enumerate(y_score):
        for j in range(args.buckets):
            if score < float((j + 1) / args.buckets):
                break
        bucket_values[j].append(score)
        bucket_indices[j].append(i)
    return (bucket_values, bucket_indices)


def get_bucket_confidence(bucket_values):
    """
    Computes average confidence for each bucket. If a bucket does
    not have predictions, returns -1.
    """

    return [
        np.mean(bucket)
        if len(bucket) > 0 else -1.
        for bucket in bucket_values
    ]


def get_bucket_accuracy(bucket_values, y_true, y_pred):
    """
    Computes accuracy for each bucket. If a bucket does
    not have predictions, returns -1.
    """

    per_bucket_correct = [
        [int(y_true[i] == y_pred[i]) for i in bucket]
        for bucket in bucket_values
    ]
    return [
        np.mean(bucket)
        if len(bucket) > 0 else -1.
        for bucket in per_bucket_correct
    ]


def calculate_error(n_samples, bucket_values, bucket_confidence, bucket_accuracy):
    """
    Computes several metrics used to measure calibration error:
        - Expected Calibration Error (ECE): \sum_k (b_k / n) |acc(k) - conf(k)|
        - Maximum Calibration Error (MCE): max_k |acc(k) - conf(k)|
        - Total Calibration Error (TCE): \sum_k |acc(k) - conf(k)|
    """

    assert len(bucket_values) == len(bucket_confidence) == len(bucket_accuracy)
    assert sum(map(len, bucket_values)) == n_samples

    expected_error, max_error, total_error = 0., 0., 0.
    for (bucket, accuracy, confidence) in zip(
        bucket_values, bucket_accuracy, bucket_confidence
    ):
        if len(bucket) > 0:
            delta = abs(accuracy - confidence)
            expected_error += (len(bucket) / n_samples) * delta
            max_error = max(max_error, delta)
            total_error += delta
    return (expected_error * 100., max_error * 100., total_error * 100.)


def create_one_hot(n_classes, args):
    """Creates one-hot label tensor."""

    #n_classes = shape[1]
    smoothing_value = args.label_smoothing / (n_classes - 1)
    one_hot = torch.full((n_classes,), smoothing_value).float()
    return one_hot


def cross_entropy(output, target, n_classes, args):
    """
    Computes cross-entropy with KL divergence from predicted distribution
    and true distribution, specifically, the predicted log probability
    vector and the true one-hot label vector.
    """
    
    model_prob = create_one_hot(n_classes, args)
    #model_prob = create_one_hot(tuple(output.shape), args)
    model_prob[target] = 1. - args.label_smoothing
    return F.kl_div(output, model_prob, reduction='sum').item()


def train(args, fold):
    
    logits_path = f"{args.logits_path}/{fold}/eval_logits.npz"
    elems = load_output(logits_path, "eval")

    #n_classes = len(elems[0]['logits'])
    n_classes = elems["logits"].shape[1]

    best_nll = float("inf")
    best_temp = -1

    temp_values = map(lambda x: round(x / 100 + 0.01, 2), range(1000))

    for temp in tqdm(temp_values, leave=False, desc='training'):
        nll = np.mean(    
            [ cross_entropy(
                    F.log_softmax(elems["logits"][idx] / temp, 0),
                    elems["true"][idx],
                    n_classes,
                    args
                )
                for idx in np.arange(elems['logits'].shape[0])
            ]
        )
        if nll < best_nll:
            best_nll = nll
            best_temp = temp

    args.temperature = best_temp

    output_dict = {'temperature': best_temp}

    print()
    print('*** training ***')
    for k, v in output_dict.items():
        print(f'{k} = {v}')


def evaluate(args, fold):

    logits_path = f"{args.logits_path}/{fold}/test_logits.npz"
    elems = load_output(logits_path, "test")

    n_classes = elems["logits"].shape[1]
    
    #labels = [elem['true'] for elem in elems]
    #preds = [elem['pred'] for elem in elems]
    labels = elems['true']
    preds = torch.argmax(elems['logits'], dim=1)

    #log_probs = [ F.log_softmax(elem['logits'] / args.temperature, 0) for elem in elems ]
    log_probs = [ F.log_softmax(logit / args.temperature, 0) for logit in elems["logits"] ]
    
    confs = [prob.exp().max().item() for prob in log_probs]
    
    """
    nll = [
        cross_entropy(log_prob, label, n_classes)
        for log_prob, label in zip(log_probs, labels)
    ]
    """
    nll = [ cross_entropy(log_probs[idx], labels[idx], n_classes, args) for idx in np.arange(labels.shape[0]) ]

    bucket_values, bucket_indices = get_bucket_scores(confs)
    bucket_confidence = get_bucket_confidence(bucket_values)
    bucket_accuracy = get_bucket_accuracy(bucket_indices, labels, preds)

    accuracy = accuracy_score(labels, preds) * 100.
    avg_conf = np.mean(confs) * 100.
    avg_nll = np.mean(nll)

    #expected_error, max_error, total_error = 0., 0., 0.
    
    """
    expected_error, max_error, total_error = calculate_error(
        len(elems), bucket_values, bucket_confidence, bucket_accuracy
    )
    """
    expected_error, max_error, total_error = calculate_error(
        labels.shape[0], bucket_values, bucket_confidence, bucket_accuracy
    )

    output_dict = {
        'accuracy': accuracy,
        'confidence': avg_conf,
        'temperature': args.temperature,
        'neg log likelihood': avg_nll,
        'expected error': expected_error,
        'max error': max_error,
        'total error': total_error,
    }

    print()
    print('*** evaluating ***')
    for k, v in output_dict.items():
        print(f'{k} = {v}')
    
    log_probs = [ F.softmax(logit / args.temperature, 0) for logit in elems["logits"] ]
    probs = np.vstack([ p.numpy() for p in log_probs ])
    return output_dict, probs

if __name__=="__main__":
    
    args = Parameters()

    for fold in np.arange(10):

        print(f"FOLD - {fold}")
        ## Estimating the temperature.
        #train(args, fold)
        ## Applying the temperature scaling on the probabilities.
        #output_dict, probs = evaluate(args, fold)
        ## Saving probabilities and logs.
        output = f"{args.base_output}/{fold}/"
        #os.makedirs(output, exist_ok=True)
        #np.savez(f"{output}/test", X_test=probs)
        #with open(f"{output}/calibration.json", 'w') as fd:
        #    json.dump(output_dict, fd)

        probs = []
        align = []
        # Applying calibration on the subfolds (train probabilities).
        for subfold in np.arange(4):
            logits_path = f"logits/clfs_output/split_10/webkb/10_folds/rep_bert/{fold}/sub_fold"
            sub_args = Parameters(logits_path=logits_path)
            print(f"\tSUB-FOLD - {subfold}")
            # Estimating the temperature.
            train(sub_args, subfold)
            # Applying the temperature scaling on the probabilities.
            output_dict, p = evaluate(sub_args, subfold)
            probs.append(p)
            # Saving probabilities and logs.
            sub_output = f"{output}/sub_fold/{subfold}"
            os.makedirs(sub_output, exist_ok=True)
            with open(f"{sub_output}/calibration.json", 'w') as fd:
                json.dump(output_dict, fd)
            # Loading fold ids alignments.
            align.append(np.load(f"{sub_args.logits_path}/{subfold}/align.npz")["align"])
        align = np.hstack(align)
        probs = np.vstack(probs)[align.argsort()]
        np.savez(f"{output}/train", X_train=probs)

