import numpy as np
from scipy.stats import multivariate_normal

class GDA:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        m = y_train.shape[0]
        X_train = X_train.reshape(m,-1)
        input_feature = X_train.shape[1]
        class_label = len(np.unique(y_train.reshape(-1)))

        self.mu    = np.zeros((class_label, input_feature))     # Rn
        self.sigma = np.zeros((class_label, input_feature, input_feature)) # Rnxn
        self.phi   = np.zeros(class_label) # R

        for label in range(class_label):
            indices = (y_train == label)

            self.phi[label] = float(np.sum(indices)) / m
            self.mu[label] = np.mean(X_train[indices, :], axis=0)
            self.sigma[label] = np.cov(X_train[indices, :], rowvar=0)

    def predict(self, X_test):
        X_test = X_test.reshape(X_test.shape[0], -1)
        class_label = self.mu.shape[0]
        scores = np.zeros((X_test.shape[0], class_label))

        for label in range(class_label):
            normal_dist_prob = multivariate_normal(mean=self.mu[label], cov=self.sigma[label])
            for i, x_test in enumerate(X_test):
                scores[i, label] = np.log(self.phi[label]) + normal_dist_prob.logpdf(x_test)
        predictions = np.argmax(scores, axis=1)
        return predictions