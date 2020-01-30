import numpy as np
import sys

from boson_model import train_jet_models
from feature_selection import select_features
from preprocessing import divide_into_jets_groups, preprocess_test_data
from proj1_helpers import create_csv_submission, load_csv_data, predict_labels


def main(train_file, test_file):
    """

    Parameters
    ----------
    train_file : str
        Path to train file
    test_file : str
        Path to test file
    """
    print("Loading the training data...")
    y, tx_raw, ids = load_csv_data(train_file)

    print("Dividing training data into the jet groups...")
    jets = divide_into_jets_groups(tx_raw, y, ids)

    print("Selecting best features expansion for each jet group...")
    jets = select_features(jets)

    print("Train model for every jet group...")
    jets_weights = train_jet_models(jets)

    print("Loading the test data...")
    _, tx_raw_test, ids_test = load_csv_data(test_file)

    print("Dividing testing data into the jet groups...")
    jets_tx, jets_ids = preprocess_test_data(tx_raw_test, ids_test, jets)

    y_res = []
    y_ids = []

    print("Classifying the test data...")
    for i in range(4):
        y_pred = predict_labels(jets_weights[i], jets_tx[i])
        y_res.append(y_pred)
        y_ids.append(jets_ids[i])

    y_test_final = np.concatenate(y_res, axis=0)
    y_test_ids = np.concatenate(y_ids, axis=0)

    y_test_final[y_test_final == 0] = -1

    create_csv_submission(y_test_ids, y_test_final, "submission.csv")
    print("Done, the results have been saved to the file: submission.csv")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Invalid number of parameters.")
