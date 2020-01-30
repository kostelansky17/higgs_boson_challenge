import numpy as np

JET_NUMBER_FEATURE = 22
NUMBER_OF_JETS = 4
NUMBER_OF_FEATURES = 30


class Jet(object):
    """Handy storage of data for Jet group

    ...

    Attributes
    ----------
    tx_raw : np.ndarray
        Not preprocessed data
    y_raw : np.ndarray
        Not preprocessed labels
    ids : np.ndarray
        Ids of data

    Methods
    -------
    add_selection(tx_select: np.ndarray, rho_select: ndarray)
        Represent the photo in the given colorspace
    """

    def __init__(self, tx_raw: np.ndarray, y_raw: np.ndarray, ids: np.ndarray):
        self.tx, self.rho, self.features_ids = preprocess_jet(tx_raw)
        self.y = substitute_values(y_raw, -1.0, 0.0)
        self.number_of_features = self.tx.shape[1]
        self.ids = ids
        self.tx_select = None
        self.rho_select = None
        self.features_configuration = None

    def add_selection(self, tx_select, rho_select, features_cofiguration):
        self.tx_select = tx_select
        self.rho_select = rho_select
        self.features_configuration = features_cofiguration


def preprocess_jet(tx_raw):
    """Clears data from columns with only NaN or 0.0 values, the rest
    of the NaN fields is substituted by the mean of the particular feature,
    deletes the column describing number of jets

    Parameters
    ----------
    tx_raw : np.ndarray
        Raw data to process

    Returns
    -------
    tx : np.ndarray
        Preprocessed data
    rho :  np.ndarray
        The percentage of missing values in each feature
    """
    features_ids = np.arange(NUMBER_OF_FEATURES)
    tx_raw = np.delete(tx_raw, JET_NUMBER_FEATURE, axis=1)
    features_ids = np.delete(features_ids, JET_NUMBER_FEATURE)

    NaN_features = np.sum(tx_raw == -999.0, axis=0) / tx_raw.shape[0] == 1.0
    tx = tx_raw[:, ~NaN_features]
    features_ids = features_ids[~NaN_features]

    empty_features = tx == 0.0
    tx = tx[:, ~empty_features.all(axis=0)]
    features_ids = features_ids[~empty_features.all(axis=0)]

    rho = np.sum(tx == -999.0, axis=0) / tx.shape[0]

    return missing_to_mean(tx), rho, features_ids


def missing_to_mean(tx):
    """Substitutes missing values of features by their mean

    Parameters
    ----------
    tx : np.ndarray
        Data with missing values

    Returns
    -------
    tx : np.ndarray
        Data with substituted missing values
    """
    tx = substitute_values(tx, -999.0, np.nan)
    NaN_positions = np.where(np.isnan(tx))
    tx_mean = np.nanmean(tx, axis=0)
    tx[NaN_positions] = np.take(tx_mean, NaN_positions[1])

    return tx


def divide_into_jets_groups(tx, y, ids):
    """Divide the data into four Classes by number of jets 

    Parameters
    ----------
    tx : np.ndarray
        Data to divide into jets group
    y : np.ndarray
        Labels to divide into jets group
    ids : np.ndarray
        Ids to divide into jets group

    Returns
    -------
    jets: list of Jet classes
        Divided data into Jet classes
    """
    jets = []

    for i in range(NUMBER_OF_JETS):
        jet_selection = tx[:, JET_NUMBER_FEATURE] == i
        jet = Jet(tx[jet_selection], y[jet_selection], ids[jet_selection])
        jets.append(jet)

    return jets


def preprocess_test_data(tx, ids, jets):
    """Divides test data into the Jet groups, and preprocess
    them to format selected by evolutionary features selection

    Parameters
    ----------
    tx : np.ndarray
        Data to divide into jets group
    ids : np.ndarray
        Labels to divide into jets group
    jets : List of Jet classes
        List of jet classes with selection of features expansion

    Returns
    -------
    np.ndarray
        Preprocessed test data divided into the Jet Groups
    np.ndarray
        Test data ids divided into the Jet groups
    """
    jets_tx = []
    jets_ids = []

    for i in range(NUMBER_OF_JETS):
        jet_select = tx[:, JET_NUMBER_FEATURE] == i

        tx_jet = tx[jet_select]
        ids_jet = ids[jet_select]

        tx_jet = tx_jet[:, jets[i].features_ids]
        tx_jet = missing_to_mean(tx_jet)
        ones = np.ones((tx_jet.shape[0], 1))

        tx_jet = np.hstack((tx_jet, ones))
        tx_exp = np.zeros(tx_jet.shape)

        for j in range(jets[i].tx_select.shape[1]):
            tx_exp[:, j] = np.power(
                tx_jet[:, jets[i].features_configuration[0, j]],
                jets[i].features_configuration[1, j],
            )

        tx_exp[:, ~np.all(tx_exp == 1, axis=0)] = z_score_normalization(
            tx_exp[:, ~np.all(tx_exp == 1, axis=0)]
        )
        jets_tx.append(tx_exp)
        jets_ids.append(ids_jet)

    return np.array(jets_tx), np.array(jets_ids)


def substitute_values(data, substitute_from=-999, substitute_to=0):
    """Substitutes values in data with specified value

    Parameters
    ----------
    data : np.ndarray
        Data to process
    substitute_from : float, optional
        Value to be substituted
    substitute_to : float, optional
        Value used in substitution

    Returns
    -------
    data: np.ndarray
        Data with substituted values
    """
    missing_position = data == substitute_from
    data[missing_position] = substitute_to

    return data


def jet_normalization(data):
    """Normalizes Jet data and add column of ones

    Parameters
    ----------
    data : np.ndarray
        Data to process

    Returns
    -------
    data: np.ndarray
        Normalized data with added column of ones
    """
    data_normalized = z_score_normalization(data)
    ones = np.ones((data.shape[0], 1))
    return np.hstack((data_normalized, ones))


def z_score_normalization(data):
    """Applies z-sore normalization on data

    Parameters
    ----------
    data : np.ndarray
        Data to normalize 

    Returns
    -------
    np.ndarray
        Normalized data
    """
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    return (data - data_mean) / data_std
