"""
Class CatBoostScratch — điểm vào duy nhất mà bên ngoài (notebook 04_train_model.ipynb) 
tương tác. Chứa __init__ (nhận hyperparameter, mặc định lấy từ config.py), fit() (gọi boosting.py 
chạy toàn bộ vòng lặp), predict() (dùng permutation cuối/toàn bộ train để encode categorical 
cho dữ liệu mới rồi chạy qua tất cả cây đã lưu), evaluate() (gọi metrics.py), get_feature_importance() 
(tổng hợp số lần mỗi feature được dùng để split qua tất cả cây trong oblivious_tree.py). Là file duy nhất 
biết và điều phối toàn bộ 7 file còn lại. Notebook ngoài chỉ cần from catboost_scratch.model import 
CatBoostScratch, không cần import trực tiếp các file con.
"""

from encoding import OrderedTargetStatistics
from boosting import OrderedBoosting, OrderedBoostingClassifier

class CatBoost:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, alpha=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.alpha = alpha
        self.random_state = random_state
        self.cat_encoders = {}
        self.model = None
        self.cat_features = []

    def preprocessFit(self, x_train, y_train, cat_features):
        x_encode = x_train.copy()
        for col in cat_features:
            encoder = OrderedTargetStatistics(self.alpha)
            x_encode[:, col] = encoder.FitTransform(x_encode[:, col], y_train)
            self.cat_encoders[col] = encoder
        return x_encode.astype(float)

    def preprocessPredict(self, x_test):
        x_encode = x_test.copy()
        for col in self.cat_features:
            encoder = self.cat_encoders[col]
            x_encode[:, col] = encoder.Transform(x_encode[:, col])
        return x_encode.astype(float)

    def Fit(self, x_train, y_train, cat_features=[]):
        self.cat_features = list(cat_features)
        x_encoded = self.preprocessFit(x_train, y_train, self.cat_features)
        self.model = OrderedBoosting(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.Fit(x_encoded, y_train)
        return self

    def Predict(self, x_test):
        x_encoded = self.preprocessPredict(x_test)
        return self.model.Predict(x_encoded)


class CatBoostClassifier:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, alpha=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.alpha = alpha
        self.random_state = random_state
        self.cat_encoders = {}
        self.model = None
        self.cat_features = []

    def preprocessFit(self, x_train, y_train, cat_features):
        x_encode = x_train.copy()
        for col in cat_features:
            encoder = OrderedTargetStatistics(self.alpha)
            x_encode[:, col] = encoder.FitTransform(x_encode[:, col], y_train)
            self.cat_encoders[col] = encoder
        return x_encode.astype(float)

    def preprocessPredict(self, x_test):
        x_encode = x_test.copy()
        for col in self.cat_features:
            encoder = self.cat_encoders[col]
            x_encode[:, col] = encoder.Transform(x_encode[:, col])
        return x_encode.astype(float)

    def Fit(self, x_train, y_train, cat_features=[]):
        self.cat_features = list(cat_features)
        x_encoded = self.preprocessFit(x_train, y_train, self.cat_features)
        self.model = OrderedBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.Fit(x_encoded, y_train)
        return self

    def PredictProba(self, x_test):
        x_encoded = self.preprocessPredict(x_test)
        return self.model.PredictProba(x_encoded)

    def Predict(self, x_test, threshold=0.5):
        return (self.PredictProba(x_test) >= threshold).astype(int)