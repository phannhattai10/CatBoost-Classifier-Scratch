import numpy as np
import pandas as pd



def MSE(y_true, y_pred):
    n = len(y_true)
    return np.sum((y_true - y_pred) ** 2) / n

def SSE(y):
    if len(y) == 0:
        return 0
    mean = np.mean(y)
    return np.sum((y - mean) ** 2)

#base learner cho GradientBoosting
class ObliviousTree:
    def __init__(self, max_depth = 3, max_borders = 32):
        self.max_depth = max_depth
        self.max_borders = max_borders

        self.best_split = [] #save best split after train
        self.leaf_value = [] #save value of leaf

    def Fit(self,x_train, y_train):
        n_samples = len(x_train)
        leaf_indices = np.zeros(n_samples,dtype="int") #use to save which value at which leaf

        for depth in range(self.max_depth):
            min_val = float("inf")
            best_threshold = None
            best_feature = None
            num_leaf_at_depth = 2**depth #number of leaf at depth
            leaf_masks = [leaf_indices == k for k in range(num_leaf_at_depth)]

            for i in range(x_train.shape[1]):
                x_col = x_train[:,i]
                sort_value = np.unique(x_col)

                if(len(sort_value) < 2): #the col has less than 2 samples
                    continue

                #create threshold (sử dụng quantiles tương tự CatBoost border_count khi số giá trị lớn)
                if self.max_borders is not None and len(sort_value) > self.max_borders:
                    quantiles = np.linspace(0, 1, self.max_borders + 2)[1:-1]
                    thresholds = np.unique(np.quantile(sort_value, quantiles))
                else:
                    thresholds = (sort_value[:-1]+sort_value[1:])/2.0

                for val in thresholds:
                    total_error = 0.0

                    for k in range(num_leaf_at_depth): #calculate all the error of leaf at the depth
                        in_leaf = leaf_masks[k]
                        left_mask = in_leaf & (x_col <= val)
                        right_mask = in_leaf & (x_col > val)
                        total_error += SSE(y_train[left_mask])
                        total_error += SSE(y_train[right_mask])

                    if total_error < min_val:
                        min_val = total_error
                        best_threshold = val
                        best_feature = i
            self.best_split.append((best_feature,best_threshold))

            if best_feature is not None: #update leaf_indices
                is_right = x_train[:,best_feature] > best_threshold
                leaf_indices += is_right.astype(int)* 2**(depth)

        #after have done all depth, calculate the mean prediction of each leaf
        max_number_leaf = 2**self.max_depth
        for leaf in range(max_number_leaf):
            y_in_leaf = y_train[leaf_indices == leaf]
            if len(y_in_leaf) > 0:
                self.leaf_value.append(np.mean(y_in_leaf))
            else:
                self.leaf_value.append(0.0)

        return self

    def Predict(self,x_test):
        n_samples = len(x_test)
        leaf_indices = np.zeros(n_samples,dtype="int")

        for depth, (feature, threshold) in enumerate(self.best_split):
            if feature is not None:
                is_right = x_test[:,feature] > threshold
                leaf_indices+= is_right.astype(int)* (2**depth)

        result = np.array([self.leaf_value[i] for i in leaf_indices])
        return result

class OrderedTargetStatistics:
    def __init__(self, alpha=1.0):
        self.prior = None
        self.alpha = alpha
        self.permutation = None
        self.sum_target = {}
        self.count_target = {}

    def FitTransform(self, x, y): #x input is a category
        n_samples = len(x)
        self.permutation = np.random.permutation(len(x))
        self.prior = np.mean(y)

        result = np.zeros(n_samples,dtype="float")

        for i in self.permutation:
            val = x[i] #get category

            sum_before = self.sum_target.get(val,0.0)
            count_before = self.count_target.get(val,0)

            result[i] = (sum_before+self.alpha*self.prior)/(count_before+self.alpha)

            #after calculate the val result, add into sum_target and count
            self.sum_target[val] = sum_before+y[i]
            self.count_target[val] = count_before+1
        return result

    def Transform(self, x):
        result = np.zeros(len(x),dtype=float)

        for i, val in enumerate(x):
            sum_val = self.sum_target.get(val,0.0)
            count_val = self.count_target.get(val,0)
            result[i] = (sum_val+ self.alpha * self.prior) / (count_val+ self.alpha)

        return result



class GradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.trees = []
        self.inital_y = None

    def Fit(self, x_train,y_train):
        self.inital_y = np.mean(y_train)
        prediction = np.full(len(x_train),self.inital_y)

        for _ in range(self.n_estimators):
            redisual = y_train - prediction

            #create tree
            tree = ObliviousTree(self.max_depth)

            #have tree study the redisual
            tree.Fit(x_train,redisual)

            y_pred = tree.Predict(x_train)
            prediction +=self.learning_rate*y_pred
            self.trees.append(tree)

        return self

    def Predict(self, x_test):
        prediction = np.full(len(x_test),self.inital_y)

        for tree in self.trees:
            tree_pred = tree.Predict(x_test)
            prediction+= self.learning_rate*tree_pred
        return prediction



#AI Generated ChatGPT: Now you are an AI Engineering, Based on my previous code help me write OrderedBoosting without changing it without making it too difficult to understand.
class OrderedBoosting:

    def __init__( self,n_estimators=10,learning_rate=0.1,max_depth=3,random_state=42):
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
        rng = np.random.default_rng(self.random_state)

        self.permutation = rng.permutation(n_samples)

        ordered_prediction = np.full(n_samples,self.initial_y,dtype=float)

        # Prediction cuối cùng của model
        prediction = np.full(n_samples,self.initial_y,dtype=float)

        self.trees = []
        for iteration in range(self.n_estimators):
            residual = (y_train - ordered_prediction)
            tree = ObliviousTree(max_depth=self.max_depth)
            tree.Fit(x_train,residual)
            tree_prediction = tree.Predict(x_train)
            self.trees.append(tree)
            prediction += (self.learning_rate* tree_prediction)

            new_ordered_prediction = np.zeros(n_samples,dtype=float)

            running_sum = 0.0
            running_count = 0

            for position in range(n_samples):
                idx = self.permutation[position]
                if running_count == 0:
                    new_ordered_prediction[idx] = (self.initial_y)
                else:
                    previous_tree_mean = (running_sum/ running_count)
                    new_ordered_prediction[idx] = (self.initial_y+self.learning_rate
                                                   *previous_tree_mean)
                running_sum += tree_prediction[idx]
                running_count += 1

            ordered_prediction = (new_ordered_prediction)
        return self

    def Predict(self, x_test):

        prediction = np.full(len(x_test),self.initial_y,dtype=float)
        for tree in self.trees:
            tree_prediction = tree.Predict(x_test)
            prediction += (self.learning_rate* tree_prediction)
        return prediction


#final model
class CatBoost:

    def __init__(self,n_estimators=10,learning_rate=0.1,max_depth=3,alpha=1.0,random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.alpha = alpha
        self.random_state = random_state

        self.cat_encoders = {} 
        self.model = None
        self.cat_features = []

    def preprocessFit(self,x_train,y_train,cat_features):
        x_encode = x_train.copy()
        for col in cat_features:
            encoder = OrderedTargetStatistics(self.alpha)
            x_encode[:,col] = encoder.FitTransform(x_encode[:,col],y_train)
            self.cat_encoders[col] = encoder
        return x_encode.astype(float)

    def preprocessPredict(self,x):
        x_encode = x.copy()
        for col in self.cat_features:
            encoder = self.cat_encoders[col]
            x_encode[:,col] = encoder.Transform(x_encode[:,col])
        return x_encode.astype(float)

    def Fit(self,x_train,y_train,cat_features=[]):
        #save category feature column
        self.cat_features = cat_features
       
        x_encoded = self.preprocessFit(x_train,y_train,cat_features)
        self.model = OrderedBoosting( n_estimators=self.n_estimators,learning_rate=self.learning_rate,
            max_depth=self.max_depth,random_state=self.random_state)

        self.model.Fit( x_encoded,y_train)
        return self

   
    def Predict(self, x_test):
        #encode 
        x_encoded = self.preprocessPredict(x_test)

        #predict with OrderedBoosting
        prediction = self.model.Predict(x_encoded)
        return prediction


#Based on my code, help me code OrderedBoostingClassifier
class OrderedBoostingClassifier:
    def __init__(self,n_estimators=10,learning_rate=0.1,max_depth=3,random_state=42,pos_weight=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.pos_weight = pos_weight  # weight cho class minority (Churn=Yes)

        self.initial_y = None
        self.initial_probability = None
        self.permutation = None
        self.trees = []

    def sigmoid(self,x):
        x = np.clip(x,-500,500)
        return 1.0/(1.0+np.exp(-x))

    def Fit(self,x_train,y_train):
        n_samples = len(x_train)

        mean_y = np.mean(y_train)
        mean_y = np.clip(mean_y,1e-15,1-1e-15)

        self.initial_probability = mean_y
        self.initial_y = np.log(mean_y/(1-mean_y))

        rng = np.random.default_rng(self.random_state)
        self.permutation = rng.permutation(n_samples)

        raw_prediction = np.full(
            n_samples,
            self.initial_y,
            dtype=float
        )

        self.trees = []

        for iteration in range(self.n_estimators):

            probability = self.sigmoid(raw_prediction)

            gradient = y_train - probability

            # Áp dụng pos_weight: nhân gradient của class minority lên để
            # model chú ý nhiều hơn vào Churn=Yes khi dữ liệu mất cân bằng
            if self.pos_weight != 1.0:
                weighted_gradient = gradient.copy()
                weighted_gradient[y_train == 1] *= self.pos_weight
            else:
                weighted_gradient = gradient

            tree = ObliviousTree(
                max_depth=self.max_depth
            )

            tree.Fit(
                x_train,
                weighted_gradient
            )

            tree_prediction = tree.Predict(x_train)

            self.trees.append(tree)

            ordered_prediction = np.zeros(
                n_samples,
                dtype=float
            )

            running_sum = 0.0
            running_count = 0

            for position in range(n_samples):

                idx = self.permutation[position]

                if running_count == 0:
                    ordered_prediction[idx] = self.initial_y
                else:
                    mean_prediction = running_sum / running_count

                    ordered_prediction[idx] = (
                        self.initial_y
                        + self.learning_rate * mean_prediction
                    )

                running_sum += tree_prediction[idx]
                running_count += 1

            raw_prediction = ordered_prediction

        return self

    def PredictProba(self,x_test):

        raw_prediction = np.full(
            len(x_test),
            self.initial_y,
            dtype=float
        )

        for tree in self.trees:

            tree_prediction = tree.Predict(x_test)

            raw_prediction += (
                self.learning_rate *
                tree_prediction
            )

        return self.sigmoid(raw_prediction)

    def Predict(self,x_test,threshold=0.5):

        probability = self.PredictProba(x_test)

        return (
            probability >= threshold
        ).astype(int)


#Based on my code, help me code CatBoostClassifier
class CatBoostClassifier:
    def __init__(self,n_estimators=50,learning_rate=0.1,max_depth=3,alpha=1.0,random_state=42,pos_weight='balanced'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.alpha = alpha
        self.random_state = random_state
        # pos_weight: 'balanced' = tự tính từ dữ liệu (n_neg/n_pos), hoặc float cụ thể
        self.pos_weight = pos_weight

        self.cat_encoders = {}
        self.model = None
        self.cat_features = []

    def preprocessFit(self,x_train,y_train,cat_features):

        x_encode = x_train.copy()

        for col in cat_features:

            encoder = OrderedTargetStatistics(
                self.alpha
            )

            x_encode[:,col] = encoder.FitTransform(
                x_encode[:,col],
                y_train
            )

            self.cat_encoders[col] = encoder

        return x_encode.astype(float)

    def preprocessPredict(self,x_test):

        x_encode = x_test.copy()

        for col in self.cat_features:

            encoder = self.cat_encoders[col]

            x_encode[:,col] = encoder.Transform(
                x_encode[:,col]
            )

        return x_encode.astype(float)

    def Fit(self,x_train,y_train,cat_features=[]):

        self.cat_features = list(cat_features)

        x_encoded = self.preprocessFit(
            x_train,
            y_train,
            self.cat_features
        )

        # Tính pos_weight thực tế từ dữ liệu nếu dùng 'balanced'
        if self.pos_weight == 'balanced':
            n_neg = float((y_train == 0).sum())
            n_pos = float((y_train == 1).sum())
            computed_weight = n_neg / n_pos if n_pos > 0 else 1.0
        else:
            computed_weight = float(self.pos_weight)

        self.model = OrderedBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            pos_weight=computed_weight
        )

        self.model.Fit(
            x_encoded,
            y_train
        )

        return self

    def PredictProba(self,x_test):

        x_encoded = self.preprocessPredict(
            x_test
        )

        return self.model.PredictProba(
            x_encoded
        )

    def Predict(self,x_test,threshold=0.5):

        probability = self.PredictProba(
            x_test
        )

        return (
            probability >= threshold
        ).astype(int)






            


        