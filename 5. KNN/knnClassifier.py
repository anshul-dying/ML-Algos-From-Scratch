import numpy as np
import statistics

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train:np.ndarray, y_train:np.ndarray):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = []
            for i, train_point in enumerate(self.X_train):
                dist = np.linalg.norm(train_point-test_point)
                distances.append((self.y_train[i], dist))

            distances.sort(key=lambda x:x[1])

            neighbours = [distances[j][0] for j in range(self.k)]

            prediction = statistics.mode(neighbours)
            predictions.append(prediction)
        return predictions