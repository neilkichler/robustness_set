import copy
import os
import datetime
import pickle
import bz2
from multiprocessing import Pool
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from fmnist_data import load_fashion_mnist_data
from feature_selection import feature_selection_mean
from train_utils import load_set_trained

import psutil

FOLDER = "benchmarks/"
RESULTS_FOLDER = "benchmarks/"
RESULT_PREFIX = "fmnist_results_"
RESULTS_EXTENSION = ".pickle"

BENCHMARK_PREFIX = "benchmark_"
BENCHMARK_TIME = "02_06_2021_18_27_06"
RUN_PREFIX = "set_mlp_density_run_"
RUN_EXTENSION = ".pickle"

BENCHMARK_PATH = FOLDER + BENCHMARK_PREFIX + BENCHMARK_TIME

DEBUG = False


def classify(model, data) -> (datetime.timedelta, float):
    def index_one(x):
        return np.where(x == 1)

    X_train, X_test, y_train, y_test = data
    current_y_train = y_train
    current_y_test = y_test

    # undo one-hot encoding for SVC
    if isinstance(model, SVC):
        current_y_train = np.apply_along_axis(index_one, 1, current_y_train).flatten()
        current_y_test = np.apply_along_axis(index_one, 1, current_y_test).flatten()

    start_time = datetime.datetime.now()

    model.fit(X_train, current_y_train)

    elapsed_time = datetime.datetime.now() - start_time

    score = model.score(X_test, current_y_test)

    return elapsed_time, score


def classify_differen_epochs_and_sparisties(run_id, fname, data, models, sample_epochs, sparseness_levels):
    print(f"[run={run_id}] Job started")

    n_models = len(models)
    X_train, X_test, y_train, y_test = data

    n_features = X_train.shape[1]

    set_pretrained_samples = {}
    n_sample_epochs = len(sample_epochs)

    set_pretrained_samples = load_set_trained(fname, sample_epochs)
    density_levels = []

    runs = set_pretrained_samples['runs']
    for run in runs:
        density_levels.append(run['set_sparsity'])

    n_density_levels = len(density_levels)

    n_sparseness_levels = len(sparseness_levels)

    dimensions = (n_density_levels, n_sample_epochs, n_sparseness_levels, n_models)
    scores = np.zeros(dimensions)
    times = np.zeros(dimensions)

    selected_features = np.zeros((n_density_levels, n_sample_epochs, n_sparseness_levels, n_features))

    for s, run in enumerate(runs):
        set_sparsity = run['set_sparsity']
        run = run['run']

        evolved_weights = run['evolved_weights']
        run_metrics = run['set_metrics']

        for i, epoch in enumerate(sample_epochs):
            current_weights = evolved_weights[i]  # we already pre-filtered
            first_layer = current_weights[1]

            for j, sparsity in enumerate(sparseness_levels):
                selected_indices = feature_selection_mean(sparsity=sparsity, weights=first_layer)

                selected_features[s][i][j] = selected_indices

                for k, model in enumerate(models):
                    elapsed_time, score = classify(model, (
                        X_train[:, selected_indices], X_test[:, selected_indices], y_train, y_test))

                    print(
                        "[run_id={:<3}|set_epsilon={:<6}|weights_epoch={:<3}|feature_sparseness={:<6}|model={:<20}] Finished fitting w/ accuracy={:>3}".format(
                            run_id, set_sparsity, epoch, sparsity, type(model).__name__, score))

                    times[s][i][j][k] = elapsed_time.microseconds
                    scores[s][i][j][k] = score

        results = {'set': set_pretrained_samples, 'sparseness_levels': sparseness_levels, 'models': models,
                   'scores': scores,
                   'times': times,
                   'stats': [], 'sample_epochs': sample_epochs, 'dimensions': dimensions,
                   'selected_features': selected_features}

        # save a compressed version
        print("saving compressed")
        with bz2.BZ2File(f"{RESULTS_FOLDER}/{RESULT_PREFIX}{run_id}{RESULTS_EXTENSION}.pbz2", "w") as h:
            pickle.dump(results, h)
        print("finished saving")

    print(f"-------Finished testing run: {run_id}")


def fmnist_feature_select_and_classify(runs, models, sample_epochs, sparseness_levels, use_logical_cores=False):
    start_test = datetime.datetime.now()
    n_cores = psutil.cpu_count(logical=use_logical_cores)

    n_training_samples = 5000  # max 60000 for Fashion MNIST
    n_testing_samples = 1000  # max 10000 for Fashion MNIST

    if DEBUG:
        # 25, 30
        i = 19
        fname = BENCHMARK_PATH + "/" + RUN_PREFIX + str(i) + RUN_EXTENSION
        x_train, y_train, x_test, y_test = load_fashion_mnist_data(n_training_samples, n_testing_samples, i)
        data = [x_train, x_test, y_train, y_test]
        classify_differen_epochs_and_sparisties(i, fname, data, models, sample_epochs, sparseness_levels)
        return

    with Pool(processes=n_cores) as pool:
        futures = []
        for i in range(runs):
            fname = BENCHMARK_PATH + "/" + RUN_PREFIX + str(i) + RUN_EXTENSION
            # TODO(Neil): Make loading pattern consistent
            x_train, y_train, x_test, y_test = load_fashion_mnist_data(n_training_samples, n_testing_samples, i)
            data = [x_train, x_test, y_train, y_test]
            futures.append(pool.apply_async(classify_differen_epochs_and_sparisties,
                                            (i, fname, data, models, sample_epochs, sparseness_levels)))

        for i, future in enumerate(futures):
            print(f'[run={i}] Starting job')
            future.get()
            print(f'-----------------------------[run={i}] Finished job')

    delta_time = datetime.datetime.now() - start_test

    print("-" * 30)
    print(f"Finished the entire process after: {delta_time.seconds}s")


if __name__ == "__main__":
    sub_folder = "fmnist_results"
    date_format = "%d_%m_%Y_%H_%M_%S"
    RESULTS_FOLDER = f"{FOLDER}/{sub_folder}_{datetime.datetime.now().strftime(date_format)}"
    os.makedirs(RESULTS_FOLDER)

    models = [SVC(C=1, kernel='linear', gamma='auto'),
              KNeighborsClassifier(n_neighbors=3),
              ExtraTreesClassifier(n_estimators=50, n_jobs=1)]

    flist = os.listdir(BENCHMARK_PATH)

    # NOTE (Neil): Assumes that all files in directory are benchmark files
    runs = len(flist)

    sample_epochs = [0, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 399]

    sparseness_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999]

    fmnist_feature_select_and_classify(runs, models, sample_epochs, sparseness_levels)
