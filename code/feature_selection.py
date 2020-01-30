import numpy as np

from boson_model import train_boson_jet_model
from preprocessing import jet_normalization, z_score_normalization

FEATURE_EXP_MAX_POWER = 5
FS_TRAINING_ITERS = 100


def select_features(jets):
    """Selects the best feature expansion for every Jet group, for further
    information please read paragraph 'Evolutionary feature selection' in
    section 'Feature engineering' in the report

    Parameters
    ----------
    jets : list of Jet classes
        The data divided into groups by jets

    Returns
    -------
    jets : list of Jet classes
        The jet groups with configuration of best feature expansion
    """
    i = 0

    for jet in jets:
        print(f"Feature selection for jet group {i}...")
        i += 1

        tx_select = jet_normalization(jet.tx)
        features_select = create_feature_selection(jet.number_of_features + 1, 1)
        rho_select = np.hstack((jet.rho, [0.0]))

        for power in range(2, FEATURE_EXP_MAX_POWER):
            tx_exp, features_exp, rho_exp = expand_features(
                tx_select, features_select, rho_select, jet, power
            )

            w = train_boson_jet_model(jet.y, tx_exp, FS_TRAINING_ITERS, 0, rho_exp)

            tx_select, features_select, rho_select = select_best_features(
                tx_exp, features_exp, rho_exp, w, jet.number_of_features + 1
            )

        jet.add_selection(tx_select, rho_select, features_select)

    return jets


def create_feature_selection(number_of_features, power):
    """Creates the information matrix about the features
    expansion, to each feature is assigned its current power

    Parameters
    ----------
    number_of_features : int
        Number of features in the Jet class
    power : int
        Power of the features

    Returns
    -------
    np.ndarray
        Feature selection information matrix
    """
    features_indexes = np.arange(number_of_features)
    powers = np.full((1, number_of_features), power)

    return np.vstack((features_indexes, powers))


def select_best_features(tx, selected_features, rho_exp, w, number_of_select):
    """Selects features by the highest value of the weights 

    Parameters
    ----------
    tx : np.ndarray
        Original features
    selected_features : [(int, int)]
        Best features from previous iteration
    rho_exp : np.ndarray
        Trained paremeters
    w : np.ndarray
        The weights of the model
    number_of_select : int

    Returns
    -------
    np.ndarray
        Selected features
    np.ndarray
        Selected features selection
    np.ndarray
        Selected rho values
    """
    select = np.argpartition(np.abs(w), -number_of_select)[-number_of_select:]

    return tx[:, select], selected_features[:, select], rho_exp[select]


def expand_features(tx_select, features_select, rho_select, jet, power):
    """Stacks the selected features and the original features
    powered to a number of a current iteration of the selection
    algorithm

    Parameters
    ----------
    tx_select : np.ndarray
        Features selected from previous iteration
    features_select : np.ndarray
        Features selection selected from previous iteration
    rho_select : np.ndarray
        Rho of features selected from previous iteration
    jet : Jet
        Currently processed Jet class
    power : int
        Power to expand features

    Returns
    -------
    np.ndarray
        Expanded features
    np.ndarray
        Expanded features selection
    np.ndarray
        Expanded rho (the percentage of missing values in each feature)
    """
    tx_power = z_score_normalization(np.power(jet.tx, power))
    new_features = create_feature_selection(jet.number_of_features, power)

    return (
        np.hstack((tx_select, tx_power)),
        np.hstack((features_select, new_features)),
        np.hstack((rho_select, jet.rho)),
    )
