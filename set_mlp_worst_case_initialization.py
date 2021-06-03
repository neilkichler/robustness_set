import os
import copy
import pickle
import time
import datetime
import logging

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from scipy.sparse import lil_matrix

from set_mlp import SET_MLP, Relu, Softmax, load_fashion_mnist_data, CrossEntropy

from multiprocessing import Pool

from utils.monitor import Monitor
import matplotlib.pyplot as plt
import psutil

FOLDER = "benchmarks"
FOLDER_NEW = ""
SUB_FOLDER = "set_worst_case"
SUB_FOLDER_PRETRAINED_WEIGHTS = "benchmark"
RUN_PREFIX = "set_mlp_run"
EXTENSION = ".pickle"

# Here you can select the time when the benchmark was made
PRETRAINED_FOLDER_TIME = "22_05_2021_13_59_10"

logging.basicConfig(filename=f'{__file__}.log', level=logging.INFO, format='%(asctime)s %(message)s', filemode='w')
log = logging.getLogger()

def vis_feature_selection(feature_selection, epoch=0, sparsity=0.5, id=0):
    image_dim = (28, 28)
    f_data = np.reshape(feature_selection, image_dim)

    plt.imshow(f_data, vmin=0, vmax=1, cmap="gray_r", interpolation=None)
    plt.title(f"epoch: {epoch}, Sparsity: {sparsity}, id: {id}")
    plt.show()


def feature_selection_mean(sparsity=0.4, weights=None):
    means = np.asarray(np.mean(np.abs(weights), axis=1)).flatten()
    means_sorted = np.sort(means)
    threshold_idx = int(means.size * sparsity)

    n = len(means)
    if threshold_idx == n:
        return np.ones(n, dtype=bool)

    means_threshold = means_sorted[threshold_idx]

    feature_selection = means >= means_threshold

    return feature_selection


def single_run_worst_case(run_id, n_training_epochs):
    log.info(f"[run={run_id}] Job started")

    # load data
    n_training_samples = 5000  # max 60000 for Fashion MNIST
    n_testing_samples  = 1000  # max 10000 for Fashion MNIST
    n_features         =  784  # Fashion MNIST has 28*28=784 pixels as features

    fname = f"{FOLDER}/{SUB_FOLDER_PRETRAINED_WEIGHTS}_{PRETRAINED_FOLDER_TIME}/{RUN_PREFIX}_{run_id}{EXTENSION}"
    with open(fname, "rb") as h:
        prev_results = pickle.load(h)
        evolved_weights = prev_results['evolved_weights'][-1]
        # Take the parameters from the prev. trained model
        set_params = prev_results['set_params']

        # SET model parameters
        n_hidden_neurons_layer = set_params['n_hidden_neurons_layer']
        epsilon = set_params['epsilon']
        zeta = set_params['zeta']
        batch_size = set_params['batch_size']
        dropout_rate = set_params['dropout_rate']
        learning_rate = set_params['learning_rate']
        momentum = set_params['momentum']
        weight_decay = set_params['weight_decay']

    np.random.seed(run_id)

    x_train, y_train, x_test, y_test = load_fashion_mnist_data(n_training_samples, n_testing_samples, run_id)

    # create SET-MLP (Multilayer Perceptron w/ adaptive sparse connectivity trained & Sparse Evolutionary Training)
    set_mlp = SET_MLP((x_train.shape[1], n_hidden_neurons_layer, n_hidden_neurons_layer, n_hidden_neurons_layer,
               y_train.shape[1]),
              (Relu, Relu, Relu, Softmax), epsilon=epsilon)

    # Set the weights initially to the opposite of the previously learned weights
    for k, v in evolved_weights.items():
        # currently we only edit the first layer
        # we might want to change more than that in the future
        if k > 1:
            break

        features_old = feature_selection_mean(sparsity=0.7, weights=v)

        # new_mask = abs(v) > 0.02 # some kind of a threshold -> use the zeta perhaps?
        # new_mask = new_mask.todense()

        n_rows, n_cols = v.shape
        new_mask = np.zeros((n_rows, n_cols), dtype=bool)
        new_mask[features_old] = np.ones(n_cols, dtype=bool)


        limit = np.sqrt(6. / float(n_rows))
        epsilon = 13
        mask_weights = np.random.rand(n_rows, n_cols)

        # set all prev. important weights to zero
        mask_weights[new_mask] = 0.0

        n_deselected = np.count_nonzero(new_mask)
        # Adjust prob to account for dropped values
        # prob = 1 - ((1 + (n_deselected / (n_rows * n_cols))) * epsilon * (n_rows + n_cols)) / (n_rows * n_cols)  # normal to have 8x connections
        prob = 1 - (epsilon * (n_rows + n_cols)) / (n_rows * n_cols)  # normal to have 8x connections

        # Adjust prob to get to the same density as before
        # Here we might do something more clever later on
        while np.count_nonzero(mask_weights[mask_weights >= prob]) < v.getnnz():
            prob -= 0.001

        # generate an Erdos Renyi sparse weights mask
        n_params = np.count_nonzero(mask_weights[mask_weights >= prob])
        weights = lil_matrix((n_rows, n_cols))

        weights[mask_weights >= prob] = np.random.uniform(-limit, limit, n_params)
        log.info("Updated initial weight layer based on worst features")

        log.info(f"Create sparse matrix with {weights.getnnz()} connections and {(weights.getnnz() / (n_rows * n_cols)) * 100} % density level")
        weights = weights.tocsr()
        set_mlp.w[k] = weights

        # Some debug visualizations
        # features = feature_selection_mean(sparsity=0.7, weights=weights)
        # vis_feature_selection(features, sparsity=0.7, id=run_id)
        # vis_feature_selection(features_old, epoch=400, sparsity=0.7, id=run_id)

    start_time = datetime.datetime.now()

    # train SET-MLP to find important features
    set_metrics = set_mlp.fit(x_train, y_train, x_test, y_test, loss=CrossEntropy, epochs=n_training_epochs,
                              batch_size=batch_size, learning_rate=learning_rate,
                              run_id=run_id, momentum=momentum, weight_decay=weight_decay, zeta=zeta, dropout_rate=dropout_rate, testing=True,
                              save_filename="Pretrained_Worst_results/set_mlp_" + str(
                    n_training_samples) + "_training_samples_e" + str(epsilon) + "_rand" + str(run_id), monitor=False)

    # After every epoch we store all weight layers to do feature selection and topology comparison
    evolved_weights = set_mlp.weights_evolution

    dt = datetime.datetime.now() - start_time

    step_time = datetime.datetime.now() - start_time
    log.info(f"\n[run_id={run_id}]Total training time: {step_time}")

    result = {'run_id': run_id, 'set_params': set_params, 'set_metrics':
            set_metrics, 'evolved_weights': evolved_weights, 'training_time':
            dt}

    with open(f"{FOLDER_NEW}/{RUN_PREFIX}_{run_id}{EXTENSION}", "wb") as h:
        pickle.dump(result, h)

    log.info(f"-------Finished testing run: {run_id}")


def test_SET_fmnist_worst_case(n_runs=10, n_training_epochs=100, use_logical_cores=True, use_pretrained=False):

    start_test = datetime.datetime.now()
    n_cores = psutil.cpu_count(logical=use_logical_cores)
    with Pool(processes=n_cores) as pool:

        futures = [pool.apply_async(single_run_worst_case, (i,
            n_training_epochs)) for i in range(n_runs)]

        for i, future in enumerate(futures):
            log.info(f'[run={i}] Starting job')
            future.get()
            log.info(f'-----------------------------[run={i}] Finished job')

    delta_time = datetime.datetime.now() - start_test

    log.info("-" * 30)
    log.info(f"Finished the entire process after: {delta_time.seconds}s")


if __name__ == "__main__":

    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    flist = os.listdir(f"{FOLDER}/{SUB_FOLDER_PRETRAINED_WEIGHTS}_{PRETRAINED_FOLDER_TIME}")

    n_runs = len(flist)
    n_training_epochs = 400
    use_logical_cores = False

    date_format = "%d_%m_%Y_%H_%M_%S"
    FOLDER_NEW = f"{FOLDER}/{SUB_FOLDER}_{datetime.datetime.now().strftime(date_format)}"
    os.makedirs(FOLDER_NEW)

    print(f"See {__file__}.log and set_mlp.log for current status of the log")

    log.info(f"Start {__file__}")

    test_SET_fmnist_worst_case(n_runs=n_runs, n_training_epochs=n_training_epochs,
            use_logical_cores=use_logical_cores)
