import sys
import os
sys.path.insert(0, os.path.abspath('src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import Model
from preprocessing import engineer_features, load_params, transform

# ── Load dữ liệu ──────────────────────────────────────────────
# Notebook 01 chỉ chia train/test (80/20).
# Val được tách từ train tại đây để tránh data leakage.
train_df_full = pd.read_csv('data/processed/train_raw.csv')
test_df       = pd.read_csv('data/processed/test_raw.csv')

# ── Split val từ train (stratified, 80/20 trong train) ────────
def stratified_split(df, target_col, train_ratio=0.80, test_ratio=0.20, random_state=42):
    """Chia train_df thành train/val giữ nguyên tỷ lệ class. Không dùng sklearn."""
    assert abs(train_ratio + test_ratio - 1.0) < 1e-9
    rng = np.random.default_rng(random_state)
    trains, vals = [], []
    for class_val in df[target_col].unique():
        subset = df[df[target_col] == class_val].copy()
        idx = rng.permutation(len(subset))
        subset = subset.iloc[idx].reset_index(drop=True)
        n = len(subset)
        n_train = int(n * train_ratio)
        trains.append(subset.iloc[:n_train])
        vals.append(subset.iloc[n_train:])
    train_df = pd.concat(trains).sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df   = pd.concat(vals).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train_df, val_df

train_df, val_df = stratified_split(train_df_full, target_col='Churn', random_state=42)
print(f'train: {len(train_df)} dòng | val: {len(val_df)} dòng | test: {len(test_df)} dòng')

params = load_params('data/processed/preprocessing_params.pkl')

train_fe = engineer_features(train_df)
val_fe   = engineer_features(val_df)
test_fe  = engineer_features(test_df)

X_train, y_train = transform(train_fe, params)
X_val,   y_val   = transform(val_fe,   params)
X_test,  y_test  = transform(test_fe,  params)
cat_cols = params['cat_cols']
cat_feature = [X_train.columns.get_loc(col) for col in cat_cols]
x_train_np = np.array(X_train)
y_train_np = np.array(y_train)

#load model
model = Model.CatBoostClassifier()
model.Fit(x_train_np,y_train_np,cat_feature)
x_val_np = np.array(X_val)

#predict
y_prob = model.PredictProba(x_val_np)
y_pred = (y_prob >= 0.5).astype(int)

print("Probability:")
print(y_prob)

print("Prediction:")
print(y_pred)

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

y_val_np = np.array(y_val)

print("Accuracy:",accuracy_score(y_val_np,y_pred))
print("Precision:",precision_score(y_val_np,y_pred))
print("Recall:",recall_score(y_val_np,y_pred))
print("F1:",f1_score(y_val_np,y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_val_np,y_pred))

import pickle

with open("data/processed/catboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

results = {
    "y_true": np.array(y_val),
    "y_pred": y_pred,
    "y_prob": y_prob
}

with open("data/processed/catboost_results.pkl", "wb") as f:
    pickle.dump(results, f)