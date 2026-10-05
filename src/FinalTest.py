import sys
import os
sys.path.insert(0, os.path.abspath('src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import pickle
from preprocessing import engineer_features, load_params, transform

test_df = pd.read_csv('data/processed/test_raw.csv')

params = load_params('data/processed/preprocessing_params.pkl')

test_fe = engineer_features(test_df)
X_test, y_test = transform(test_fe, params)

with open(
    'data/processed/catboost_model.pkl',
    'rb'
) as f:
    model = pickle.load(f)

x_test_np = np.array(X_test)

y_prob = model.PredictProba(x_test_np)

y_pred = (y_prob >= 0.5).astype(int)

print(y_prob)
print(y_pred)

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

y_test_np = np.array(y_test)

print("Accuracy:",accuracy_score(y_test_np,y_pred))
print("Precision:",precision_score(y_test_np,y_pred))
print("Recall:",recall_score(y_test_np,y_pred))
print("F1:",f1_score(y_test_np,y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test_np,y_pred))