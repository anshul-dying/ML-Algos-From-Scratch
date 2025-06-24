from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * ((x_i @ self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2*self.lambda_param*self.w)
                else:
                    self.w -= self.lr * (2*self.lambda_param*self.w - y_[idx] * x_i)
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        approx = (X @ self.w) + self.b
        return np.sign(approx)
    


X, y = make_blobs(n_samples=100, centers=2, random_state=6)
y = np.where(y == 0, -1, 1)

svm = SVM()
svm.fit(X, y)
preds = svm.predict(X)

accuracy = np.mean(preds == y)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

def plot_hyperplane(w, b, X):
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_vals = np.linspace(xlim[0], xlim[1], 200)
    y_vals = lambda x, offset: -(w[0] * x + b + offset) / w[1]

    plt.plot(x_vals, y_vals(x_vals, 0), 'k-', label='Decision boundary (w.x + b = 0)')
    plt.plot(x_vals, y_vals(x_vals, 1), 'r--', label='Margin +1')
    plt.plot(x_vals, y_vals(x_vals, -1), 'b--', label='Margin -1')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(X[:, 0], X[:, 1], c=preds, cmap='bwr', alpha=0.7)
plot_hyperplane(svm.w, svm.b, X)
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Decision Boundary and Margins")
plt.show()
