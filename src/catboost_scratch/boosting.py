"""
Vòng lặp chính của Ordered Boosting. Mỗi iteration: lấy permutation 
mới (permutation.py) → encode categorical theo permutation đó (target_statistics.py) 
→ tính gradient/residual (losses.py) → fit 1 cây mới trên residual (oblivious_tree.py) 
→ cập nhật prediction với learning_rate → lặp lại đến khi đủ iterations hoặc early 
stopping (dựa trên metrics.py đo trên val set). Gọi tất cả 4 file trên theo đúng thứ 
tự mỗi iteration. Được model.py gọi trong hàm fit().

"""

import numpy as np
from oblivious_tree import ObliviousTree
from permutation import PermutationGenerator
from metrics import sigmoid
from config import DEFAULT_N_ESTIMATORS, DEFAULT_LEARNING_RATE, DEFAULT_MAX_DEPTH, DEFAULT_RANDOM_STATE

class GradientBoosting:
    def __init__(self, n_estimators=DEFAULT_N_ESTIMATORS, learning_rate=DEFAULT_LEARNING_RATE, max_depth=DEFAULT_MAX_DEPTH):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.inital_y = None

    def Fit(self, x_train, y_train):
        self.inital_y = np.mean(y_train)
        prediction = np.full(len(x_train), self.inital_y)

        for _ in range(self.n_estimators):
            redisual = y_train - prediction
            tree = ObliviousTree(self.max_depth)
            tree.Fit(x_train, redisual)
            y_pred = tree.Predict(x_train)
            prediction += self.learning_rate * y_pred
            self.trees.append(tree)

        return self

    def Predict(self, x_test):
        prediction = np.full(len(x_test), self.inital_y)
        for tree in self.trees:
            tree_pred = tree.Predict(x_test)
            prediction += self.learning_rate * tree_pred
        return prediction


class OrderedBoosting:
    def __init__(self, n_estimators=DEFAULT_N_ESTIMATORS, learning_rate=DEFAULT_LEARNING_RATE, max_depth=DEFAULT_MAX_DEPTH, random_state=DEFAULT_RANDOM_STATE):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.initial_y = None
        self.permutation = None
        self.trees = []

    def Fit(self, x_train, y_train):
        n_samples = len(x_train)
        self.initial_y = np.mean(y_train)
        
        perm_gen = PermutationGenerator(self.random_state)
        self.permutation = perm_gen.generate(n_samples)

        ordered_prediction = np.full(n_samples, self.initial_y, dtype=float)
        prediction = np.full(n_samples, self.initial_y, dtype=float)

        self.trees = []
        for iteration in range(self.n_estimators):
            residual = (y_train - ordered_prediction)
            tree = ObliviousTree(max_depth=self.max_depth)
            tree.Fit(x_train, residual)
            tree_prediction = tree.Predict(x_train)
            self.trees.append(tree)
            prediction += (self.learning_rate * tree_prediction)

            new_ordered_prediction = np.zeros(n_samples, dtype=float)
            running_sum = 0.0
            running_count = 0

            for position in range(n_samples):
                idx = self.permutation[position]
                if running_count == 0:
                    new_ordered_prediction[idx] = self.initial_y
                else:
                    previous_tree_mean = (running_sum / running_count)
                    new_ordered_prediction[idx] = (self.initial_y + self.learning_rate * previous_tree_mean)
                running_sum += tree_prediction[idx]
                running_count += 1

            ordered_prediction = new_ordered_prediction
        return self

    def Predict(self, x_test):
        prediction = np.full(len(x_test), self.initial_y, dtype=float)
        for tree in self.trees:
            tree_prediction = tree.Predict(x_test)
            prediction += (self.learning_rate * tree_prediction)
        return prediction


class OrderedBoostingClassifier:
    def __init__(self, n_estimators=DEFAULT_N_ESTIMATORS, learning_rate=DEFAULT_LEARNING_RATE, 
                 max_depth=DEFAULT_MAX_DEPTH, random_state=DEFAULT_RANDOM_STATE):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.initial_y = None
        self.initial_probability = None
        self.permutation = None
        self.trees = []

    def Fit(self, x_train, y_train):
        n_samples = len(x_train)
        mean_y = np.mean(y_train)
        mean_y = np.clip(mean_y, 1e-15, 1 - 1e-15)

        self.initial_probability = mean_y
        self.initial_y = np.log(mean_y / (1 - mean_y))

        perm_gen = PermutationGenerator(self.random_state)
        self.permutation = perm_gen.generate(n_samples)

        raw_prediction = np.full(n_samples, self.initial_y, dtype=float)
        self.trees = []

        for iteration in range(self.n_estimators):
            probability = sigmoid(raw_prediction)
            gradient = y_train - probability

            tree = ObliviousTree(max_depth=self.max_depth)
            tree.Fit(x_train, gradient)
            tree_prediction = tree.Predict(x_train)

            self.trees.append(tree)
            ordered_prediction = np.zeros(n_samples, dtype=float)

            running_sum = 0.0
            running_count = 0

            for position in range(n_samples):
                idx = self.permutation[position]
                if running_count == 0:
                    ordered_prediction[idx] = self.initial_y
                else:
                    mean_prediction = running_sum / running_count
                    ordered_prediction[idx] = (self.initial_y + self.learning_rate * mean_prediction)

                running_sum += tree_prediction[idx]
                running_count += 1

            raw_prediction = ordered_prediction

        return self

    def PredictProba(self, x_test):
        raw_prediction = np.full(len(x_test), self.initial_y, dtype=float)
        for tree in self.trees:
            tree_prediction = tree.Predict(x_test)
            raw_prediction += (self.learning_rate * tree_prediction)

        return sigmoid(raw_prediction)

    def Predict(self, x_test, threshold=0.5):
        probability = self.PredictProba(x_test)
        return (probability >= threshold).astype(int)