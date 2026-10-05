"""
Sinh ra nhiều thứ tự ngẫu nhiên (random permutation) của tập 
train. Đây là nền tảng để "giả lập" việc dữ liệu đến tuần tự 
theo thời gian — nguyên liệu đầu vào cho cả Ordered Target 
Statistics lẫn Ordered Boosting.Được gọi bởi target_statistics.py 
(để biết thứ tự tính TS) và boosting.py (để mỗi iteration/mỗi 
cây dùng 1 permutation khác nhau, tránh model "nhìn thấy" chính dòng 
nó đang xử lý).
"""

import numpy as np

class PermutationGenerator:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def generate(self, n_samples):
        rng = np.random.default_rng(self.random_state)
        return rng.permutation(n_samples)