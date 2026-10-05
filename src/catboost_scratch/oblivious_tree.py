"""
Xây 1 cây quyết định dạng symmetric/oblivious (toàn bộ node ở cùng 1 tầng 
dùng chung 1 điều kiện split) — nhận feature đã encode (từ target_statistics.py) 
và residual (từ losses.py) làm input, tìm split tốt nhất, trả về cấu trúc cây + giá 
trị lá. Không tự gọi file nào, chỉ nhận dữ liệu đầu vào đã được boosting.py chuẩn bị 
sẵn (từ target_statistics.py và losses.py). Được boosting.py gọi ở mỗi iteration để 
tạo ra 1 cây mới trong ensemble.
"""
import numpy as np
from metrics import SSE

class ObliviousTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.best_split = [] 
        self.leaf_value = [] 

    def Fit(self, x_train, y_train):
        n_samples = len(x_train)
        leaf_indices = np.zeros(n_samples, dtype="int")

        for depth in range(self.max_depth):
            min_val = float("inf")
            best_threshold = None
            best_feature = None

            for i in range(x_train.shape[1]):
                x_col = x_train[:, i]
                sort_value = np.unique(x_col)

                if len(sort_value) < 2:
                    continue

                thresholds = (sort_value[:-1] + sort_value[1:]) / 2.0
                for val in thresholds:
                    total_error = 0.0
                    num_leaf_at_depth = 2 ** depth

                    for k in range(num_leaf_at_depth):
                        in_leaf = (leaf_indices == k)
                        left_mask = in_leaf & (x_col <= val)
                        right_mask = in_leaf & (x_col > val)
                        total_error += SSE(y_train[left_mask])
                        total_error += SSE(y_train[right_mask])

                    if total_error < min_val:
                        min_val = total_error
                        best_threshold = val
                        best_feature = i

            self.best_split.append((best_feature, best_threshold))

            if best_feature is not None:
                is_right = x_train[:, best_feature] > best_threshold
                leaf_indices += is_right.astype(int) * (2 ** depth)

        max_number_leaf = 2 ** self.max_depth
        self.leaf_value = []
        for leaf in range(max_number_leaf):
            y_in_leaf = y_train[leaf_indices == leaf]
            if len(y_in_leaf) > 0:
                self.leaf_value.append(np.mean(y_in_leaf))
            else:
                self.leaf_value.append(0.0)

        return self

    def Predict(self, x_test):
        n_samples = len(x_test)
        leaf_indices = np.zeros(n_samples, dtype="int")

        for depth, (feature, threshold) in enumerate(self.best_split):
            if feature is not None:
                is_right = x_test[:, feature] > threshold
                leaf_indices += is_right.astype(int) * (2 ** depth)

        result = np.array([self.leaf_value[i] for i in leaf_indices])
        return result