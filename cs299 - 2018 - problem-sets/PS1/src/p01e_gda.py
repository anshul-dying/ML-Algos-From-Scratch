import numpy as np
import util
from scipy.stats import multivariate_normal
import os

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_test, y_test = util.load_dataset(eval_path, add_intercept=False)
    gda = GDA()
    gda.fit(x_train, y_train)
    y_pred = gda.predict(x_test)
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)

    for ys in y_pred:
        with open(pred_path, 'a') as file:
            file.write(f"{ys}\n")
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def fit(self, x: np.ndarray, y:np.ndarray):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        self.m, self.n = x.shape
        self.class_label = len(np.unique(y))

        self.mu = np.zeros((self.class_label, self.n))
        self.sigma = [None] * self.class_label
        self.phi = np.zeros(self.class_label)

        for label in range(self.class_label):
            indices = (y==label)
            self.phi[label] = np.sum(indices)/self.m
            self.mu[label] = np.mean(x[indices,:], axis=0)
            self.sigma[label] = np.cov(x[indices, :], rowvar=0)
        # *** END CODE HERE ***

    def predict(self, x:np.ndarray):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, _ = x.shape
        scores = np.zeros((m, self.class_label))
        for label in range(self.class_label):
            normal_prob_dist = multivariate_normal(mean=self.mu[label], cov=self.sigma[label])

            for i, x_i in enumerate(x):
                scores[i, label] = np.log(self.phi[label]) + normal_prob_dist.logpdf(x_i)
        predictions = np.argmax(scores, axis=1)
        return predictions
        # *** END CODE HERE


if __name__ == "__main__":
    main()