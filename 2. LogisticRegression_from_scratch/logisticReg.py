import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, lr: float, iter: int):
        self.learning_rate = lr
        self.iterations    = iter
        self.weights       = np.zeros(2)

    def h_theta(self, x):
        t0, t1 = self.weights[0], self.weights[1]
        return 1 / (1 + np.exp(-(t0 + t1 * x)))
    
    def derivative(self ,X, y, idx):
        der = 0
        for i in range(X.shape[0]):
            if idx == 0:
                der += self.h_theta(X[i]) - y[i]
            else:
                der += (self.h_theta(X[i]) - y[i])*X[i]
        return der
         
    def gradient_descent(self, weights: list, X, y):
        new_weights = weights.copy()
        for i in range(len(new_weights)):
            new_weights[i] -= self.learning_rate * self.derivative(X, y, i)
        return new_weights
    
    def fit(self, feature: np.array, target: np.array):
        if(feature.shape != target.shape):
            return f"No. of rows are not equal for X({feature.shape}) and y({target.shape})"

        for i in range(self.iterations):
            self.weights = self.gradient_descent(self.weights, feature, target)

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            res = self.h_theta(X[i])
            if(res >= 0.5):
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_pred
        

# Synthetic data for X and y
np.random.seed(42)
n_samples = 200
X = np.random.uniform(0, 10, n_samples)
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

plt.scatter(X, y, alpha=0.3, label='Data')
plt.plot(x_test, y_prob, color='green', label='Learned Sigmoid')
plt.plot(x_test, 1 / (1 + np.exp(-(true_t0 + true_t1 * x_test))),
         color='red', linestyle='--', label='True Sigmoid')
plt.xlabel('X')
plt.ylabel('Probability')
plt.title('Logistic Regression Fit')
plt.legend()
plt.show()