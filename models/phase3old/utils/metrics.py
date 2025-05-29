# phase3/utils/metrics.py
import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mpe(y_true, y_pred):
    return np.mean((y_pred - y_true) / y_true) * 100

def quantile_pinball(y_true, y_pred, q):
    e = y_true - y_pred
    return np.mean(np.maximum(q * e, (q - 1) * e))