import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')


    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    _, t_train = util.load_dataset(train_path, label_col='t')
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    _, t_valid = util.load_dataset(valid_path, label_col='t')
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    _, t_test = util.load_dataset(test_path, label_col='t')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    log_reg = LogisticRegression()
    log_reg.fit(x_train, t_train)
    pred_c = log_reg.predict(x_test)
    util.plot(x_test, t_test, log_reg.theta, 'output/p02c.png')
    with open(pred_path_c, 'w') as f:
        for label in pred_c:
            f.write(f"{label}\n")


    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    pred_d = log_reg.predict(x_train)
    util.plot(x_test, t_test, log_reg.theta, 'output/p02d.png')
    with open(pred_path_d, 'w') as f:
        for label in pred_d:
            f.write(f"{label}\n")


    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _h_theta(theta, x):
        z = np.dot(x, theta)
        return _sigmoid(z)
    
    v_plus = x_valid[y_valid == 1]
    alpha = _h_theta(log_reg.theta, v_plus).mean()
    
    def predict(theta, x):
        return _h_theta(theta, x) / alpha >= 5
    
    theta_prime = log_reg.theta + np.log(2/alpha - 1) * np.array([1,0,0])
    util.plot(x_test, t_test, theta_prime, 'output/p02e.png')
    pred_e = predict(theta_prime, x_train)
    with open(pred_path_e, 'w') as f:
        for label in pred_e:
            f.write(f"{label}\n")
    # *** END CODE HERE
