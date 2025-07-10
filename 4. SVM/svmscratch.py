import numpy as np

class SVM:
    def __init__(self, learning_rate = 0.001, lambda_param=0.01, n_iters=1000):
        self.lr             = learning_rate
        self.lambda_param   = lambda_param
        self.n_iters        = n_iters
        self.w              = None
        self.b              = None

    def fit(self, X:np.ndarray, y:np.ndarray):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0

        y_ = np.where(y<=0,-1,1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(self.w, x_i) - self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_[idx]*x_i)
                    self.b -= self.lr * y_[idx]

    def predict(self, X: np.ndarray):
        approx = X @ self.w + self.b
        return np.sign(approx)