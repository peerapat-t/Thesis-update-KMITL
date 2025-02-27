import numpy as np
import pandas as pd
import duckdb

def apk(actual, predicted, k=10):
    if not actual:
        return 0.0

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    return score / min(len(actual), k)

def mapk(val_array, top_k_preds, k=10):
    if val_array.shape[0] == 0:
        return 0.0
    actual_items = [list(np.nonzero(val_array[i])[0]) for i in range(val_array.shape[0])]

    return np.mean([apk(a, p, k) for a, p in zip(actual_items, top_k_preds)])

def get_top_k(pred_matrix, k):
    return np.argsort(-pred_matrix, axis=1)[:, :k]