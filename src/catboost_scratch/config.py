"""
Nơi duy nhất khai báo giá trị mặc định của mọi hyperparameter 
(iterations, learning_rate, depth, l2_leaf_reg, random_strength, 
early_stopping_rounds, số lượng permutation...).
"""

# Nơi quản lý toàn bộ hyperparameter mặc định của mô hình
DEFAULT_N_ESTIMATORS = 10
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MAX_DEPTH = 3
DEFAULT_ALPHA = 1.0
DEFAULT_RANDOM_STATE = 42
DEFAULT_CLASSIFIER_N_ESTIMATORS = 50