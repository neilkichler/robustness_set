import copy
import os
import datetime
import pickle
from multiprocessing import Pool
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from madelon_data import load_madelon_npz, train_test_split_normalize
from feature_selection import feature_selection_mean
from lung_data import load_lung_data, train_test_split_normalize

import psutil

FOLDER = "benchmarks/"
RESULTS_FOLDER = "benchmarks/"
RESULT_PREFIX = "madelon_results_"
RESULTS_EXTENSION = ".pickle"

BENCHMARK_PREFIX = "benchmark_madelon_"
BENCHMARK_TIME = "05_06_2021_01_59_55"
RUN_PREFIX = "set_mlp_density_run_"
RUN_EXTENSION = ".pickle"

BENCHMARK_PATH = FOLDER + BENCHMARK_PREFIX + BENCHMARK_TIME

DEBUG = 0


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


def classify_differen_epochs_and_sparisties(run_id, fname, data, models,
        sample_epochs, sparseness_levels, save_folder):
    print(f"[run={run_id}] Job started")

    n_models = len(models)
    X_train, X_test, y_train, y_test = data

    n_features = X_train.shape[1]

    with open(fname, "rb") as h:
        set_pretrained = pickle.load(h)

    density_levels = set_pretrained['density_levels']
    n_density_levels = len(density_levels)
    runs = set_pretrained['runs']

    n_sample_epochs = len(sample_epochs)
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
            current_weights = evolved_weights[i]
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

        results = {'set': set_pretrained, 'sparseness_levels': sparseness_levels, 'models': models, 'scores': scores,
                   'times': times,
                   'stats': [], 'sample_epochs': sample_epochs, 'dimensions': dimensions,
                   'selected_features': selected_features}

        with open(f"{save_folder}/{RESULT_PREFIX}{run_id}{RESULTS_EXTENSION}", "wb") as h:
            pickle.dump(results, h)

    print(f"-------Finished testing run: {run_id}")


def madelon_feature_select_and_classify(runs, models, sample_epochs,
        sparseness_levels, save_folder,  use_logical_cores=False):
    start_test = datetime.datetime.now()
    n_cores = psutil.cpu_count(logical=use_logical_cores)

    X, y = load_madelon_npz()

    if DEBUG:
        i = 0
        fname = BENCHMARK_PATH + "/" + RUN_PREFIX + str(i) + RUN_EXTENSION
        data = train_test_split_normalize(X, y, random_state=i)
        classify_differen_epochs_and_sparisties(i, fname, data, models, sample_epochs, sparseness_levels)
        return

    with Pool(processes=n_cores) as pool:
        futures = []
        for i in range(runs):
            fname = BENCHMARK_PATH + "/" + RUN_PREFIX + str(i) + RUN_EXTENSION
            data = train_test_split_normalize(X, y, random_state=i)
            futures.append(pool.apply_async(classify_differen_epochs_and_sparisties,
                                            (i, fname, data, models,
                                                sample_epochs,
                                                sparseness_levels, save_folder)))

        for i, future in enumerate(futures):
            print(f'[run={i}] Starting job')
            future.get()
            print(f'-----------------------------[run={i}] Finished job')

    delta_time = datetime.datetime.now() - start_test

    print("-" * 30)
    print(f"Finished the entire process after: {delta_time.seconds}s")


if __name__ == "__main__":

    sub_folder = "madelon_results"
    date_format = "%d_%m_%Y_%H_%M_%S"
    RESULTS_FOLDER = f"{FOLDER}/{sub_folder}_{datetime.datetime.now().strftime(date_format)}"
    os.makedirs(RESULTS_FOLDER)

    models = [SVC(C=1, kernel='linear', gamma='auto'),
              KNeighborsClassifier(n_neighbors=3),
              ExtraTreesClassifier(n_estimators=50, n_jobs=1)]

    flist = os.listdir(BENCHMARK_PATH)

    # NOTE (Neil): Assumes that all files in directory are benchmark files
    runs = len(flist)

    sample_epochs = [0, 5, 10, 20, 30, 40, 50, 75, 99]

    sparseness_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95,
                         0.975, 0.99, 0.995, 0.999]

    madelon_feature_select_and_classify(runs, models, sample_epochs,
            sparseness_levels, RESULTS_FOLDER)
