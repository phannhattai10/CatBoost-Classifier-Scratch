"""
Tính Ordered Target Statistics — encode categorical feature thành số, 
nhưng với mỗi dòng chỉ dùng các dòng đứng trước nó trong permutation 
(không dùng toàn bộ dữ liệu tĩnh như Target Statistics thường). Đây là 
cơ chế cốt lõi chống Prediction Shift/Target Leakage. Nhận permutation 
từ permutation.py làm input. Được boosting.py gọi ở mỗi iteration (vì mỗi 
iteration dùng permutation khác nhau → TS cũng phải tính lại theo permutation đó) 
để chuyển categorical feature trong tập train thành số trước khi đưa vào oblivious_tree.py
"""

import numpy as np
from config import DEFAULT_ALPHA

class OrderedTargetStatistics:
    def __init__(self, alpha=DEFAULT_ALPHA):
        self.prior = None
        self.alpha = alpha
        self.permutation = None
        self.sum_target = {}
        self.count_target = {}

    def FitTransform(self, x, y):
        n_samples = len(x)
        self.permutation = np.random.permutation(len(x))
        self.prior = np.mean(y)

        result = np.zeros(n_samples, dtype="float")

        for i in self.permutation:
            val = x[i]
            sum_before = self.sum_target.get(val, 0.0)
            count_before = self.count_target.get(val, 0)

            result[i] = (sum_before + self.alpha * self.prior) / (count_before + self.alpha)

            self.sum_target[val] = sum_before + y[i]
            self.count_target[val] = count_before + 1
        return result

    def Transform(self, x):
        result = np.zeros(len(x), dtype=float)

        for i, val in enumerate(x):
            sum_val = self.sum_target.get(val, 0.0)
            count_val = self.count_target.get(val, 0)
            result[i] = (sum_val + self.alpha * self.prior) / (count_val + self.alpha)

        return result