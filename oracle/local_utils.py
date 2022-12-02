import numpy as np

def build_clf_beans(clf_probas, label):
    predictions = clf_probas.argmax(axis=1)
    confidence_freq = {}
    hits = {}
    # For each prediction
    for idx, predicted_class in enumerate(predictions):
        
        # Getting the probability of the predicted class
        probability = clf_probas[idx][predicted_class] * 10
        bean = np.trunc(probability) / 10
        bean = 0.9 if bean >= 1 else bean
        # Adding the bean in confidence if is not there yet.
        if bean not in confidence_freq:
            confidence_freq[bean] = 0
        confidence_freq[bean] += 1
        # Veryfing if the predicted class was right.
        if predicted_class == label[idx]:
            if bean not in hits:
                hits[bean] = 0
            hits[bean] += 1
    return confidence_freq, hits

def get_miss_predictor(confidence_freq, hits, threshold=0.3):

    predictor = {}
    # For each confidence interval.
    for bean in hits:
        # Get the hit rate.
        hits_rate = hits[bean] / confidence_freq[bean]
        
        if hits_rate < threshold:
            predictor[bean] = True
    return predictor

def predict(X, estimator):
    
    estimates = []
    predictions = X.argmax(axis=1)
    # For each prediction.
    for idx, predicted_class in enumerate(predictions):
        probability = X[idx][predicted_class] * 10
        bean = np.trunc(probability) / 10
        bean = 0.9 if bean >= 1 else bean
        # If this confidence has a miss rate greater than THRESHOLD (wether it is in the dictionary or not)
        if bean in estimator:
            estimates.append(0)
        else:
            estimates.append(1)
    return np.array(estimates)