import numpy as np


def sigmoid(x):
    """Computes sigmoid of x

    Parameters
    ----------
    x : np.ndarray
        Vector for sigmoid computation

    Returns
    -------
    np.ndarray
        Sigmoid function of x
    """
    return 1 / (1 + np.exp(-x))


def mean_squared_loss(y, y_predicted):
    """Counts mean squared loss of the model

    Parameters
    ----------
    y : np.ndarray
        Class labels
    y_predicted : np.ndarray
        Values predicted by the model

    Returns
    -------
    float
        Mean squared loss of the model
    """
    N: int = y.shape[0]
    y_difference: np.ndarray = y - y_predicted

    return y_difference.dot(y_difference) / (2 * N)


def logistic_loss(y, y_predicted):
    """Counts logistic loss of Logistic regression model

    Parameters
    ----------
    y : np.ndarray
        Class labels
    y_predicted : np.ndarray
        Values predicted by the model

    Returns
    -------
    float
        Logistic loss of the regression model
    """
    N: int = y.shape[0]
    return (1 / N) * np.sum(np.log(1 + np.exp(y_predicted)) - y * y_predicted)


def compute_gradient(y, y_predicted, tx, N=1, regularization=0):
    """Computes gradient of a linear regression model with squared loss function

    Parameters
    ----------
    y : np.ndarray
        Class labels
    y_predicted : np.ndarray
        Labels predicted by model
    tx : np.ndarray
        Data
    N : int
        The size of the dataset
    regularization : int or np.ndarray

    Returns
    -------
    np.ndarray
        Gradient of a mean square loss for linear model
    """
    return -tx.T.dot(y - y_predicted) / N + regularization


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent

    Parameters
    ----------
    y : np.ndarray
        Class labels
    tx : np.ndarray
        Train data
    initial_w : np.ndarray
        Initial weights of a linear model
    max_iters : int
        Maximal number of iterations
    gamma : float
        Learning rate

    Returns
    -------
    w : np.ndarray
        The parameters of the linear model trained by GD
    float
        The mean squared loss of the model
    """
    w: np.ndarray = initial_w
    N: int = tx.shape[1]

    for _ in range(max_iters):
        w -= gamma * compute_gradient(y, tx.dot(w), tx, N)

    return w, mean_squared_loss(y, tx.dot(w))


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent

    Parameters
    ----------
    y : np.ndarray
        Class labels
    tx : np.ndarray
        Train data
    initial_w : np.ndarray
        Initial weights of the linear model
    max_iters : int
        Maximal number of iterations
    gamma : float
        Learning rate

    Returns
    -------
    w : np.ndarray
        The parameters of the linear model trained by SGD
    float
        The mean squared loss of the final model
    """
    w: np.ndarray = initial_w
    N: int = tx.shape[1]
    random_samples: np.ndarray = np.random.randint(N, size=max_iters)

    for sample in random_samples:
        w -= gamma * compute_gradient(y[sample], tx[sample, :].dot(w), tx[sample, :])

    return w, mean_squared_loss(y, tx.dot(w))


def least_squares(y, tx):
    """Least squares regression using normal equations

    Parameters
    ----------
    y : np.ndarray
        Class labels
    tx : np.ndarray
        Train data

    Returns
    -------
    np.ndarray
        The parameters of a model
    float
        The mean squared loss of the final model
    """
    return ridge_regression(y, tx, 0)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations

    Parameters
    ----------
    y : np.ndarray
        Class labels
    tx : np.ndarray
        Train data
    lambda_ : float
        The penalty term

    Returns
    -------
    w : np.ndarray
        The parameters of the linear model trained by SGD
    float
        The mean squared loss of the final model
    """
    penalization_matrix = lambda_ * np.identity(tx.shape[1])
    txT_tx_inv = np.linalg.pinv(np.dot(tx.T, tx) - penalization_matrix)
    txT_tx_inv_txT = np.dot(txT_tx_inv, tx.T)
    w = np.dot(txT_tx_inv_txT, y)

    return w, mean_squared_loss(y, tx.dot(w))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent

    Parameters
    ----------
    y : np.ndarray
        Class labels
    tx : np.ndarray
        Train data
    initial_w : np.ndarray
        Initial weights
    max_iters : int
        Maximum number of iteration for training
    gamma : float
        Learning rate

    Returns
    -------
    w : np.ndarray
        The parameters of the linear model trained by LR
    float
        The logistic loss of the final model
    """
    w: np.ndarray = initial_w
    N: int = tx.shape[1]

    for _ in range(max_iters):
        y_predicted = sigmoid(tx.dot(w))
        w -= gamma * compute_gradient(y, y_predicted, tx, N)

    return w, logistic_loss(y.T, y_predicted)


def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
    """Regulated logistic regression using gradient descent

    Parameters
    ----------
    y : np.ndarray
        Class labels
    tx : np.ndarray
        Train data
    initial_w : np.ndarray
        Initial weights
    max_iters : int
        Maximum number of iteration for training
    gamma : float
        Learning rate
    lambda_ : float
        Penalty for complexity

    Returns
    -------
    w : np.ndarray
        The parameters of the linear model trained by L2 logistic regression
    float
        The logistic loss with the penalization of the final model
    """
    w: np.ndarray = initial_w
    N: int = tx.shape[1]

    for _ in range(max_iters):
        y_predicted = sigmoid(tx.dot(w))
        reg_parameter = lambda_ * w
        w -= gamma * compute_gradient(y, y_predicted, tx, N, reg_parameter)

    return w, logistic_loss(y.T, y_predicted) + lambda_ / (2 * N * w.T.dot(w))
