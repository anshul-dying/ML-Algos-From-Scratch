import numpy  as np
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations    = iterations
    

    def fit(self, feature: np.array, target: np.array) -> list | str | int:
        if(feature.shape != target.shape):
            return f"No. of rows are not equal for X({feature.shape}) and y({target.shape})"
    
        def h_theta(x, weights:list):
            return weights[0] + weights[1]*x

        def derivative(X, y, idx):
            derivative = 0
            for i in range(X.shape[0]):
                if idx == 0:
                    derivative += (h_theta(X[i], self.weights) - y[i])
                else:
                    derivative += (h_theta(X[i], self.weights) - y[i])*X[i]
            return derivative / X.shape[0]

        def gradientDescent(weights: list, X, y):
            new_weights = weights.copy()
            for i in range(len(weights)):
                new_weights[i] -= self.learning_rate * derivative(X, y, i)
            
            return new_weights
        
        self.weights = np.zeros(feature.ndim+1)
        for i in range(self.iterations):
            self.weights = gradientDescent(self.weights, feature, target)
        
        return self.weights
    
    def predict(self, X: np.array) -> np.array:
        pred = []

        for i in range(X.shape[0]):
            pred.append(self.weights[0]+self.weights[1]*X[i])
        
        return np.array(pred)

