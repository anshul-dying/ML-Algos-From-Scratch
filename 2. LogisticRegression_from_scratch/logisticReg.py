import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations    = iterations

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
        z = self.weights@x_with_bias
        return 1/(1+np.exp(-z))

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

    
    # def predict(self, X: np.ndarray) -> np.array:
    #     pred = []
    #     for i in range(X.shape[0]):
    #         pred.append(self._h_theta(X[i]))
    #     return np.array(pred)

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            res = self._h_theta(X[i])
            if(res >= 0.5):
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_pred
        

# Synthetic data for X and y
np.random.seed(42)
n_samples = 200
X = np.random.uniform(0, 10, n_samples)
X = X.reshape(-1, 1)
true_t0 = -5
true_t1 = 1
probs = 1 / (1 + np.exp(-(true_t0 + true_t1 * X)))
y = np.random.binomial(1, probs)

print(f"Proportion of class 1: {y.mean():.2f}")

reg = LogisticRegression(0.01, 1000)
reg.fit(X, y)
t0, t1 = reg.weights[0], reg.weights[1]
x_test = np.linspace(0, 10, 100)
y_prob = 1 / (1 + np.exp(-(t0 + t1 * x_test)))

print("Accuracy: ", accuracy_score(y, reg.predict(X)))

plt.scatter(X, y, alpha=0.3, label='Data')
plt.plot(x_test, y_prob, color='green', label='Learned Sigmoid')
plt.plot(x_test, 1 / (1 + np.exp(-(true_t0 + true_t1 * x_test))),
         color='red', linestyle='--', label='True Sigmoid')
plt.xlabel('X')
plt.ylabel('Probability')
plt.title('Logistic Regression Fit')
plt.legend()
plt.show()