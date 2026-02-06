import numpy as np
import statistics

class KNNBase:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train:np.ndarray, y_train:np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
    
    def _get_neighbours(self, test_point):
        distances = []
        for i, train_point in enumerate(self.X_train):
            dist = np.linalg.norm(train_point-test_point)
            distances.append((self.y_train[i], dist))

        distances.sort(key=lambda x:x[1])
        neighbours = [distances[j][0] for j in range(self.k)]
        return neighbours

    def predict(self, X_test):
        raise NotImplementedError("Subclasses should implement this method")

class KNNClassifier(KNNBase):
    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            neighbours = self._get_neighbours(test_point)
            prediction = statistics.mode(neighbours)
            predictions.append(prediction)
        return predictions
    
class KNNRegression(KNNBase):
    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            neighbours = self._get_neighbours(test_point)
            prediction = np.mean(neighbours)
            predictions.append(prediction)
        return predictions