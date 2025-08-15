import numpy  as np
import pandas as pd

class LinearBase:
    """
    Base class for linear regression models using gradient descent.

    Attributes:
        learning_rate (float): The step size for gradient descent.
        iterations (int): The number of passes over the training data.
        weights (np.ndarray): The learned weights for the model, including bias.
        alpha (float): The regularization strength (used in child classes).
    """
    def __init__(self, learning_rate=0.01, iterations=1000, alpha=1):
        self.learning_rate = learning_rate
        self.iterations    = iterations
        self.alpha         = alpha

    def fit(self, feature: np.ndarray, target: np.ndarray) -> list | str | int:
        self.m, self.n = feature.shape
        self.X = feature
        self.y = target
        if(feature.shape[0] != target.shape[0]):
            return f"No. of rows are not equal for X({feature.shape}) and y({target.shape})"
        
        self.weights = np.zeros(self.n+1)
        for i in range(self.iterations):
            self._gradientDescent(feature, target)
        
        return self.weights

    def _h_theta(self, x):
        x_with_bias = np.insert(x, 0, 1)
        return self.weights@x_with_bias

    def _derivative(self, idx):
        derivative = 0 
        for i in range(self.m):
            if idx == 0:
                derivative += (self._h_theta(self.X[i]) - self.y[i])
            else:
                derivative += (self._h_theta(self.X[i]) - self.y[i])*self.X[i, idx-1]
        return derivative / self.m

    def _gradientDescent(self, X, y):
        for j in range(len(self.weights)):
            self.weights[j] -= self.learning_rate * self._derivative(j)

    
    def predict(self, X: np.ndarray) -> np.array:
        pred = []
        for i in range(X.shape[0]):
            pred.append(self._h_theta(X[i]))
        return np.array(pred)

class LinearRegression(LinearBase):
    """
    Standard Linear Regression using Ordinary Least Squares.
    No regularization is applied.
    """
    def __init__(self, learning_rate=0.01, iterations=1000):
        super().__init__(learning_rate, iterations)

    def fit(self, feature, target):
        return super().fit(feature, target)
        
    def predict(self, X):
        return super().predict(X)
    
class RidgeRegression(LinearBase):
    """
    Linear Regression with L2 (Ridge) regularization.
    
    The regularization term helps to prevent overfitting by penalizing large weights.
    """
    def __init__(self, learning_rate=0.01, iterations=1000, alpha=1):
        super().__init__(learning_rate, iterations, alpha)
    
    def _derivative(self, idx):
        derivative = 0
        for i in range(self.m):
            if idx == 0:
                derivative += (self._h_theta(self.X[i])-self.y[i])
            else:
                derivative += (self._h_theta(self.X[i])-self.y[i])*self.X[i, idx-1]

        derivative = derivative / self.m
        if idx != 0:
            derivative += 2*(self.alpha/self.m)*(self.weights[idx]) 
        return derivative


# TODO -> Implement Lasso Regression
class LassoRegression(LinearBase):
    pass