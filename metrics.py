import numpy as np

def accuracy_score(y_true,y_pred):


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    n_examples=len(y_true)
    correct_pred=np.sum(y_pred==y_true)

    return correct_pred/n_examples
    
    
def crossed_entropy(y_true, y_proba):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.ndim != 1 or y_proba.ndim != 1:
        raise ValueError("y_true and y_proba must be 1D arrays.")

    if y_true.shape != y_proba.shape:
        raise ValueError("y_true and y_proba must have the same length.")

    if y_true.size == 0:
        raise ValueError("Input arrays must not be empty.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")

    if not np.isfinite(y_proba).all():
        raise ValueError("Probabilities must be finite.")

    if np.any((y_proba < 0) | (y_proba > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    epsilon = 1e-15
    p = np.clip(y_proba, epsilon, 1 - epsilon)

    return -np.mean(
        y_true * np.log(p) + (1 - y_true) * np.log(1 - p)
    )
