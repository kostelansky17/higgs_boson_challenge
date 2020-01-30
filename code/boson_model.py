import numpy as np

from implementations import compute_gradient, sigmoid
from proj1_helpers import predict_labels

NUMBER_OF_FOLDS = 10
TRAINING_ITERATIONS = 1500
LAMBDA = 0.02


def train_jet_models(jets):
    """Trains model for each jet group 

    Parameters
    ----------
    jets : List of Jet classes
        The data divided into groups by jets

    Returns
    -------
    np.array
        Trained weights for each Jet group
    """

    jets_weights = []
    i = 0
    for jet in jets:
        print(f"Training Jet Model {i}")
        i += 1

        w = train_boson_jet_model(
            jet.y, jet.tx_select, TRAINING_ITERATIONS, LAMBDA, jet.rho_select, True
        )
        jets_weights.append(w)

    return np.array(jets_weights)


def train_boson_jet_model(y, tx, max_iters, lambda_, rho, log=False):
    """Regulated logistic regression using gradient descent and HKK regularization

    Parameters
    ----------
    y : np.ndarray
        Class labels
    tx : np.ndarray
        Training data
    max_iters : int
        Maximum number of iteration for training
    gamma : float
        Learning rate
    lambda_ : float
        Penalty for complexity
    rho : float
        Penalty for complexity
    log : bool, optional
        If True Test and Validation error is being printed

    Returns
    -------
    w : np.ndarray
        The weights of the model
    float
        The logistic loss with the penalization of the final model
    """
    w = np.zeros(tx.shape[1])
    N: int = tx.shape[0]
    valid_ids, train_ids = split_indexes(N, NUMBER_OF_FOLDS)
    iteration = 1

    for i in range(NUMBER_OF_FOLDS):
        fold_tx_train = tx[train_ids[i]]
        fold_tx_valid = tx[valid_ids[i]]

        fold_y_train = y[train_ids[i]]
        fold_y_valid = y[valid_ids[i]]

        for _ in range(max_iters):
            y_predicted = sigmoid(fold_tx_train.dot(w))
            reg_parameter = w * lambda_ / (1 - rho)
            w -= (1 / iteration) * compute_gradient(
                fold_y_train, y_predicted, fold_tx_train, N, reg_parameter
            )
            iteration += 1

        if log:
            print(
                f"Training accuracy: {accuracy_score(predict_labels(w, fold_tx_train), fold_y_train)}"
            )
            print(
                f"Validation accuracy: {accuracy_score(predict_labels(w, fold_tx_valid), fold_y_valid)}"
            )

    return w


def split_indexes(data_size, k, seed=1):
    """Given the size of data and the number of folds, returns
    test and train indices for each fold

    Parameters
    ----------
    data_size : integer
        Number of training instances
    k : integer
        Number of folds
    seed : integer
        Seed for index randomization

    Returns
    -------
    test_indices : np.array
        List of indices for each test fold
    train_indices : np.array
        List of indices for each train fold
    """
    np.random.seed(seed)
    indexes = np.random.choice(data_size, data_size, replace=False)

    average_size = data_size / float(k)
    fold_indexes = []
    last = 0.0

    while last < len(indexes):
        fold_indexes.append(indexes[int(last) : int(last + average_size)])
        last += average_size

    train_indices = []
    test_indices = []

    for i in range(k):
        test_indices.append(fold_indexes[i])
        train_indices_intermediate = []
        for j in range(k - 1):
            train_indices_intermediate.append(fold_indexes[(i + j + 1) % k])
        train_indices.append(np.concatenate(train_indices_intermediate))

    return np.array(test_indices), np.array(train_indices)


def accuracy_score(y_predicted, y):
    """Computes accuracy of the predictions

    Parameters
    ----------
    y_predicted : np.ndarray
        Predicted labels
    y : np.ndarray
        True labels

    Returns
    -------
    float
        Accuracy of the model
    """
    return np.sum(y_predicted == y) / y.shape[0]
