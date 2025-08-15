import numpy as np
import util
import os

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)

    logReg = LogisticRegression()

    logReg.fit(x_train, y_train)

    y_pred = logReg.predict(x_test)
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)

    for ys in y_pred:
        with open(pred_path, "a") as file:
            file.write(f"{ys}\n")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def _sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def _h_theta(self, x):
        z = np.dot(x, self.theta)
        return self._sigmoid(z)
    
    def _gradient_log_likelihood(self, X, y):
        return X.T@(y-self._h_theta(X))
    
    def _hessian_log_likelihood(self, X):
        h = self._h_theta(X)
        R = np.diag(h*(1-h))
        return -X.T@R@X

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros(x.shape[1])

        for i in range(self.max_iter):
            self.theta -= np.linalg.solve(self._hessian_log_likelihood(x), self._gradient_log_likelihood(x,y))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        y_pred = []
        for i in range(x.shape[0]):
            if self._h_theta(x[i]) > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
        
        return y_pred
        # *** END CODE HERE ***
