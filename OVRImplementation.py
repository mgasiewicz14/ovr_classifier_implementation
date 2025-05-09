import numpy as np
from sklearn.svm import SVC


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X ,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = np.where(linear_output >= 0, 1, -1)
                update = self.learning_rate * (y_[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict_proba(self, X):
        return np.dot(X, self.weights) + self.bias
            
    def predict(self, X):
        linear_output = self.predict_proba(X)
        y_pred = np.where(linear_output >= 0, 1, 0)
        return y_pred
    
class OVRPerceptron:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.classifiers = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            ovr_y = np.where(y == cls, 1, 0)
            clf = Perceptron(**self.kwargs)
            clf.fit(X, ovr_y)
            self.classifiers[cls] = clf
    
    def predict(self, X):
        scores = {cls: clf.predict_proba(X) for cls, clf in self.classifiers.items()}
        scores_matrix = np.column_stack([scores[cls] for cls in self.classes])
        predictions = np.argmax(scores_matrix, axis=1)
        return predictions
    
class OVRSVC:
    def __init__(self, **svc_kwargs):
        self.svc_kwargs = svc_kwargs
        self.classifiers = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            ovr_y = np.where(y == cls, 1 ,0)
            clf = SVC(**self.svc_kwargs)
            clf.fit(X, ovr_y)
            self.classifiers[cls] = clf

    def predict(self, X):
        scores = {cls: clf.decision_function(X) for cls, clf in self.classifiers.items()}
        scores_matrix = np.column_stack([scores[cls] for cls in self.classes])
        predictions = np.argmax(scores_matrix, axis=1)
        return predictions
    
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, average='macro'):
    classes = np.unique(y_true)

    if average == 'micro':
        TP = 0
        FP = 0
        for cls in classes:
            TP += np.sum((y_true == cls) & (y_pred == cls))
            FP += np.sum((y_true != cls) & (y_pred == cls))
        return TP / (TP + FP) if (TP + FP) > 0 else 0
    
    elif average == 'macro':
        precisions = []
        for cls in classes:
            TP = np.sum((y_true == cls) & (y_pred == cls))
            FP = np.sum((y_true != cls) & (y_pred == cls))
            p = TP / (TP + FP) if (TP + FP) > 0 else 0
            precisions.append(p)
        return np.mean(precisions)
    
def recall(y_true, y_pred, average='macro'):
    classes = np.unique(y_true)

    if average == 'micro':
        TP = 0
        FN = 0
        for cls in classes:
            TP += np.sum((y_true == cls) & (y_pred == cls))
            FN += np.sum((y_true == cls) & (y_pred != cls))
        return TP / (TP + FN) if (TP + FN) > 0 else 0
    
    elif average == 'macro':
        recalls = []
        for cls in classes:
            TP = np.sum((y_true == cls) & (y_pred == cls))
            FN = np.sum((y_true == cls) & (y_pred != cls))
            r = TP / (TP + FN) if (TP + FN) > 0 else 0
            recalls.append(r)
            return np.mean(recalls)
        
def f1_score(y_true, y_pred, average='macro'):
    prec = precision(y_true, y_pred, average=average)
    rec = recall(y_true, y_pred, average=average)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

