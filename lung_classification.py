from numpy.core.multiarray import ndarray
import numpy as np

from set_mlp import SET_MLP
from nn_functions import Relu, Sigmoid, MSE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import logging

import scipy.io as sio

DATA_PATH = 'data/'
FILE_NAME = 'lung.mat'

TEST_SIZE = 1 / 3


def load_lung_data() -> (ndarray, ndarray):
    """
    The lung.mat data set is from: http://featureselection.asu.edu/

    Dataset characteristics:

    few sample: ()
    medium # features:

    To expect: high variance

    """
    lung_data = sio.loadmat(DATA_PATH + FILE_NAME)

    X_: ndarray = lung_data['X']
    y_: ndarray = lung_data['Y']

    enc = OneHotEncoder().fit(y_)
    y_ = enc.transform(y_).astype('uint8').toarray()

    logging.info("Available output categories:")
    logging.info(enc.categories_)

    return X_, y_


def train_test_split_normalize(X_: ndarray, y_: ndarray, test_size=TEST_SIZE, random_state=42) \
        -> (ndarray, ndarray, ndarray, ndarray):
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=test_size, random_state=random_state)

    normalize = StandardScaler().fit(X_train_)
    X_train_ = normalize.transform(X_train_)
    X_test_ = normalize.transform(X_test_)
    return X_train_, X_test_, y_train_, y_test_


def set_mlp_feature_selection(X_train_, X_test_, y_train_, y_test_, set_params_, sparsity_=0.4) -> ndarray:
    n_hidden_neurons_layer = set_params_['n_hidden_neurons_layer']
    epochs = set_params_['epochs']
    epsilon = set_params_['epsilon']
    zeta = set_params_['zeta']
    batch_size = set_params_['batch_size']
    dropout_rate = set_params_['dropout_rate']
    learning_rate = set_params_['learning_rate']
    momentum = set_params_['momentum']
    weight_decay = set_params_['weight_decay']

    set_mlp = SET_MLP(
        (X_train_.shape[1], n_hidden_neurons_layer, n_hidden_neurons_layer, n_hidden_neurons_layer, y_train_.shape[1]),
        (Relu, Relu, Relu, Sigmoid), epsilon=epsilon)

    set_mlp.fit(X_train_, y_train_, X_test_, y_test_, loss=MSE, epochs=epochs, zeta=zeta, batch_size=batch_size,
                dropout_rate=dropout_rate, learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay,
                testing=True, run_id=run_id)

    return set_mlp.feature_selection_mean(sparsity=sparsity_)


if __name__ == '__main__':
    run_id = 0

    # SET model parameters
    set_params = {'n_hidden_neurons_layer': 3000,
                  'epochs': 100,
                  'epsilon': 20,  # set the sparsity level
                  'zeta': 0.3,  # in [0..1]. Percentage of unimportant connections to be removed and replaced
                  'batch_size': 2, 'dropout_rate': 0, 'learning_rate': 0.01, 'momentum': 0.9, 'weight_decay': 0.0002}

    X, y = load_lung_data()

    X_train, X_test, y_train, y_test = train_test_split_normalize(X, y, test_size=TEST_SIZE, random_state=run_id)

    feature_selection = set_mlp_feature_selection(X_train, X_test, y_train, y_test, set_params)

    X_train = X_train[:, feature_selection]
    X_test = X_test[:, feature_selection]
