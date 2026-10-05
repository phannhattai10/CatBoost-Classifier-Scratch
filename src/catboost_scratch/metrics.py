"""
Tính Accuracy, AUC, F1, Logloss — dùng để (a) theo dõi early stopping trong 
boosting.py (so sánh metric trên val qua từng iteration), và (b) đánh giá model 
sau khi train xong trong model.py. Được gọi bởi cả boosting.py (trong lúc train) 
và model.py (hàm evaluate() sau khi train xong).
"""

import numpy as np

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def SSE(y):
    if len(y) == 0:
        return 0.0
    return np.sum((y - np.mean(y)) ** 2)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))