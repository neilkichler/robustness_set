import copy
import datetime
import logging
import os
import pickle
from multiprocessing import Pool
import bz2

import matplotlib.pyplot as plt
import numpy as np
import psutil
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from fmnist_data import load_fashion_mnist_data
from set_mlp import SET_MLP, Relu, Softmax , CrossEntropy
from utils.monitor import Monitor

from train_utils import sample_weights_and_metrics

FOLDER = "benchmarks"


def vis_feature_selection(feature_selection, epoch=0, sparsity=0.5, id=0):
    image_dim = (28, 28)
    f_data = np.reshape(feature_selection, image_dim)

    plt.imshow(f_data, vmin=0, vmax=1, cmap="gray_r", interpolation=None)
    plt.title(f"epoch: {epoch}, Sparsity: {sparsity}, id: {id}")
    plt.show()


# return the index of where a one is inside an array
def index_one(x):
    return np.where(x == 1)


def single_run_density(run_id, set_params, density_levels, n_training_epochs,
                       fname="", save_compressed=True):
    """
    the density levels are the set epsilon sparsity levels
    """
    print(f"[run={run_id}] Job started")
    n_training_samples = 5000  # max 60000 for Fashion MNIST
    n_testing_samples = 1000  # max 10000 for Fashion MNIST
    n_features = 784  # Fashion MNIST has 28*28=784 pixels as features

    # SET model parameters
    n_hidden_neurons_layer = set_params['n_hidden_neurons_layer']
    zeta = set_params['zeta']
    batch_size = set_params['batch_size']
    dropout_rate = set_params['dropout_rate']
    learning_rate = set_params['learning_rate']
    momentum = set_params['momentum']
    weight_decay = set_params['weight_decay']

    sum_training_time = 0

    np.random.seed(run_id)

    x_train, y_train, x_test, y_test = load_fashion_mnist_data(n_training_samples, n_testing_samples, run_id)

    if os.path.isfile(fname):
        with open(fname, "rb") as h:
            results = pickle.load(h)
    else:
        results = {'density_levels': density_levels, 'runs': []}

    for epsilon in density_levels:
        logging.info(f"[run_id={run_id}] Starting SET-Sparsity: epsilon={epsilon}")
        set_params['epsilon'] = epsilon
        # create SET-MLP (Multilayer Perceptron w/ adaptive sparse connectivity trained & Sparse Evolutionary Training)

        set_mlp = SET_MLP((x_train.shape[1], n_hidden_neurons_layer, n_hidden_neurons_layer, n_hidden_neurons_layer,
                           y_train.shape[1]), (Relu, Relu, Relu, Softmax), epsilon=epsilon)

        start_time = datetime.datetime.now()
        # train SET-MLP to find important features
        set_metrics = set_mlp.fit(x_train, y_train, x_test, y_test, loss=CrossEntropy, epochs=n_training_epochs,
                              batch_size=batch_size, learning_rate=learning_rate,
                              momentum=momentum, weight_decay=weight_decay, zeta=zeta, dropout_rate=dropout_rate,
                              testing=True, run_id=run_id,
                              save_filename="", monitor=False)

        dt = datetime.datetime.now() - start_time

        sample_epochs = [0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 199]
        print("HELLO")
        print(len(set_mlp.weights_evolution))
        evolved_weights, set_metrics = sample_weights_and_metrics(set_mlp.weights_evolution, set_metrics, sample_epochs)

        run_result = {'run_id': run_id, 'set_params': copy.deepcopy(set_params), 'set_metrics': set_metrics,
                      'evolved_weights': evolved_weights, 'training_time': dt}

        results['runs'].append({'set_sparsity': epsilon, 'run': run_result})

        fname = f"{FOLDER}/set_mlp_density_run_{run_id}.pickle"
        # save preliminary results
        if save_compressed:
            with bz2.BZ2File(f"{fname}.pbz2", "w") as h:
                pickle.dump(results, h)
        else:
            with open(fname, "wb") as h:
                pickle.dump(results, h)


def single_run(run_id, set_params, models, sample_epochs,
               sparseness_levels, n_training_epochs, use_pretrained=False):
    # instead of returning just save everything directly inside here??
    print(f"[run={run_id}] Job started")

    n_models = len(models)
    # TODO(Neil): Maybe make these global?
    # load data
    n_training_samples = 5000  # max 60000 for Fashion MNIST
    n_testing_samples = 1000  # max 10000 for Fashion MNIST
    n_features = 784  # Fashion MNIST has 28*28=784 pixels as features

    # SET model parameters
    n_hidden_neurons_layer = set_params['n_hidden_neurons_layer']
    epsilon = set_params['epsilon']
    zeta = set_params['zeta']
    batch_size = set_params['batch_size']
    dropout_rate = set_params['dropout_rate']
    learning_rate = set_params['learning_rate']
    momentum = set_params['momentum']
    weight_decay = set_params['weight_decay']

    sum_training_time = 0

    np.random.seed(run_id)

    x_train, y_train, x_test, y_test = load_fashion_mnist_data(n_training_samples, n_testing_samples, run_id)

    if not use_pretrained:
        # create SET-MLP (Multilayer Perceptron w/ adaptive sparse connectivity trained & Sparse Evolutionary Training)

        set_mlp = SET_MLP((x_train.shape[1], n_hidden_neurons_layer, n_hidden_neurons_layer, n_hidden_neurons_layer,
                           y_train.shape[1]), (Relu, Relu, Relu, Softmax), epsilon=epsilon)

        start_time = datetime.datetime.now()

        # train SET-MLP to find important features
        set_metrics = set_mlp.fit(x_train, y_train, x_test, y_test, loss=CrossEntropy, epochs=n_training_epochs,
                                  batch_size=batch_size, learning_rate=learning_rate,
                                  momentum=momentum, weight_decay=weight_decay, zeta=zeta, dropout_rate=dropout_rate,
                                  testing=True,
                                  save_filename="", monitor=False)
        # save_filename="Pretrained_results/set_mlp_" + str(
        #     n_training_samples) + "_training_samples_e" + str(epsilon) + "_rand" + str(run_id), monitor=True)

        # After every epoch we store all weight layers to do feature selection and topology comparison
        evolved_weights = set_mlp.weights_evolution

        dt = datetime.datetime.now() - start_time

        step_time = datetime.datetime.now() - start_time
        print("\nTotal training time: ", step_time)
        sum_training_time += step_time

        result = {'run_id': run_id, 'set_params': set_params, 'set_metrics':
            set_metrics, 'evolved_weights': evolved_weights, 'training_time':
                      dt}

        with open(f"{FOLDER}/set_mlp_run_{run_id}.pickle", "wb") as h:
            pickle.dump(result, h)

    set_pretrained = None
    # TODO(Neil): hardcoded path

    fname = f"benchmarks/benchmark_22_05_2021_13_59_10/set_mlp_run_{run_id}.pickle"
    if os.path.isfile(fname):
        try:
            with open(fname, "rb") as h:
                set_pretrained = pickle.load(h)
        except EOFError:
            return
    else:
        return

    evolved_weights = set_pretrained['evolved_weights']
    n_evolutions = len(evolved_weights)
    n_sparseness_levels = len(sparseness_levels)
    selected_features = np.zeros((n_evolutions, n_sparseness_levels, n_features))

    # TODO(Neil): Make a pandas dataframe instead?
    dimensions = (n_evolutions, n_sparseness_levels, n_models)
    scores = np.empty(dimensions)
    times = np.empty(dimensions)

    zero_array = [None] * n_models
    stats = [zero_array[:] for _ in range(n_sparseness_levels)]
    monitor = Monitor()

    for i, epoch in enumerate(sample_epochs):
        for j, sparsity in enumerate(sparseness_levels):
            first_layer = evolved_weights[epoch][1]
            selected_indices = feature_selection_mean(sparsity, weights=first_layer)  # alternative
            # selected_indices = set_mlp.feature_selection(fullness, weights=weights[1])

            # vis_feature_selection(selected_indices, epoch=epoch, sparsity=sparsity, id=run_id)

            selected_features[i][j] = selected_indices
            # continue

            selected_x_train = x_train[:, selected_indices]
            selected_x_test = x_test[:, selected_indices]

            for k, model in enumerate(models):
                current_y_train = y_train
                current_y_test = y_test

                # undo one-hot encoding for SVC
                if isinstance(model, SVC):
                    current_y_train = np.apply_along_axis(index_one, 1, current_y_train).flatten()
                    current_y_test = np.apply_along_axis(index_one, 1, current_y_test).flatten()

                start_time = datetime.datetime.now()
                # wrapper(model, selected_x_train, current_y_train)
                monitor.start_monitor()

                model.fit(selected_x_train, current_y_train)

                monitor.stop_monitor()
                elapsed_time = datetime.datetime.now() - start_time

                score = model.score(selected_x_test, current_y_test)

                print(
                    "[run_id={:<3}|weights_epoch={:<3}|sparseness={:<6}|model={:<20}] Finished fitting w/ accuracy={:>3}".format(
                        run_id, epoch, sparsity, type(model).__name__, score))

                times[i][j][k] = elapsed_time.microseconds
                scores[i][j][k] = score
                if i == len(evolved_weights) - 1:
                    stats[j][k] = monitor.get_stats()

    results = {'set': set_pretrained, 'sparseness_levels': sparseness_levels, 'models': models, 'scores': scores,
               'times': times,
               'stats': stats, 'sample_epochs': sample_epochs, 'dimensions': dimensions,
               'selected_features': selected_features}

    with open(f"{FOLDER}/fmnist_results_{run_id}.pickle", "wb") as h:
        pickle.dump(results, h)

    print(f"-------Finished testing run: {run_id}")


def feature_selection_mean(sparsity=0.4, weights=None):
    # TODO(Neil): explain why we choose only the first layer
    # the main reason is that this first layer will already have
    # most of the important information in it, given that everything
    # gets backpropageted

    means = np.asarray(np.mean(np.abs(weights), axis=1)).flatten()
    means_sorted = np.sort(means)
    threshold_idx = int(means.size * sparsity)

    n = len(means)
    if threshold_idx == n:
        return np.ones(n, dtype=bool)

    means_threshold = means_sorted[threshold_idx]

    feature_selection = means >= means_threshold

    return feature_selection


def test_fmnist_pretrained_set(evolved_weights, runs=10, n_training_epochs=100, sample_epochs=None,
                               sparseness_levels=None, use_logical_cores=True):
    results = {}
    max_finished = 0

    sparseness = 0.7

    # evolved_weights = evolved_weights[sample_epochs]

    for epochs in evolved_weights:
        for i, epoch in enumerate(epochs):
            if (i % 50 == 0):
                feature_selection = feature_selection_mean(epoch[1], sparseness)
                vis_feature_selection(feature_selection, epoch=i, sparsity=sparseness)


def test_fmnist(runs=10, n_training_epochs=100, sample_epochs=None,
                sparseness_levels=None, use_logical_cores=True, use_pretrained=False):
    # use some default values if none are given
    if sparseness_levels is None:
        sparseness_levels = [0.1, 0.5, 0.9]

    if sample_epochs is None:
        sample_epochs = [10, 50, 100, 200]

    results = {}
    max_finished = 0

    # Test selection on multiple classifiers
    models = [SVC(C=1, kernel='linear', gamma='auto'),
              KNeighborsClassifier(n_neighbors=3),
              ExtraTreesClassifier(n_estimators=50, n_jobs=1)]

    if use_pretrained:
        print("Skip training SET")
        print("Do feature selection and fitting of:")
    else:
        print('Starting SET training for feature selection, followed by fitting:')

    for model in models:
        print(f'{model}')

    n_sparseness_levels = len(sparseness_levels)
    n_models = len(models)

    info = {'runs': runs, 'n_training_epochs': n_training_epochs, 'sparseness_levels': sparseness_levels,
            'n_models': n_models}

    dimensions = (runs, n_training_epochs, n_sparseness_levels, n_models)
    scores = np.zeros(dimensions)
    times = np.zeros(dimensions)
    stats_ = []  # np.empty((runs, len(models), n_sparseness_levels))
    selected_features_per_run = []
    all_evolved_weights = []
    set_metrics_per_run = np.empty((runs, n_training_epochs, 4))

    # SET model parameters
    set_params = {'n_hidden_neurons_layer': 3000,
                  'epsilon': 13,  # set the sparsity level
                  'zeta': 0.3,  # in [0..1]. Percentage of unimportant connections to be removed and replaced
                  'batch_size': 40, 'dropout_rate': 0, 'learning_rate': 0.05, 'momentum': 0.9, 'weight_decay': 0.0002}

    start_test = datetime.datetime.now()
    n_cores = psutil.cpu_count(logical=use_logical_cores)
    with Pool(processes=n_cores) as pool:

        futures = [pool.apply_async(single_run, (i, set_params, models,
                                                 sample_epochs, sparseness_levels, n_training_epochs, use_pretrained))
                   for i in range(runs)]

        for i, future in enumerate(futures):
            print(f'[run={i}] Starting job')
            # s, t, stats, selected_features, evolved_weights, set_metrics = future.get()
            future.get()
            print(f'-----------------------------[run={i}] Finished job')

    delta_time = datetime.datetime.now() - start_test

    print("-" * 30)
    print(f"Finished the entire process after: {delta_time.seconds}s")

    return results


def fmnist_train_set_differnt_densities(runs=10, n_training_epochs=100, set_sparsity_levels=None, use_logical_cores=True):
    # SET model parameters
    set_params = {'n_hidden_neurons_layer': 3000,
                  'epsilon': 13,  # set the sparsity level
                  'zeta': 0.3,  # in [0..1]. Percentage of unimportant connections to be removed and replaced
                  'batch_size': 40, 'dropout_rate': 0, 'learning_rate': 0.05, 'momentum': 0.9, 'weight_decay': 0.0002}

    start_test = datetime.datetime.now()
    n_cores = psutil.cpu_count(logical=use_logical_cores)
    with Pool(processes=n_cores) as pool:
        futures = []
        for i in range(runs):
            fname = f"{FOLDER}/set_mlp_density_run_{i}.pickle"

            # check if results already exist
            # if os.path.isfile(fname):
            #     with open(fname, "rb") as h:
            #         result = pickle.load(h)
            #         for el in result['runs']:
            #             remaining_density_levels.remove(el['set_sparsity'])

            futures.append(pool.apply_async(single_run_density, (i, set_params,
                set_sparsity_levels, n_training_epochs, fname)))

        for i, future in enumerate(futures):
            print(f'[run={i}] Starting job')
            future.get()
            print(f'-----------------------------[run={i}] Finished job')

    delta_time = datetime.datetime.now() - start_test

    print("-" * 30)
    print(f"Finished the entire process after: {delta_time.seconds}s")



def fmnist_train_set_differnt_densities_sequential(runs=10, n_training_epochs=100, set_sparsity_levels=None, use_logical_cores=True):
    # SET model parameters
    set_params = {'n_hidden_neurons_layer': 3000,
                  'epsilon': 13,  # set the sparsity level
                  'zeta': 0.3,  # in [0..1]. Percentage of unimportant connections to be removed and replaced
                  'batch_size': 40, 'dropout_rate': 0, 'learning_rate': 0.05, 'momentum': 0.9, 'weight_decay': 0.0002}

    start_test = datetime.datetime.now()
    n_cores = psutil.cpu_count(logical=use_logical_cores)
    for i in range(runs):
        fname = f"{FOLDER}/set_mlp_density_run_{i}.pickle"
        print(f'[run={i}] Starting job')
        single_run_density(i, set_params, set_sparsity_levels, n_training_epochs, fname)
        print(f'-----------------------------[run={i}] Finished job')

    delta_time = datetime.datetime.now() - start_test

    print("-" * 30)
    print(f"Finished the entire process after: {delta_time.seconds}s")



if __name__ == "__main__":
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    sub_folder = "benchmark"
    date_format = "%d_%m_%Y_%H_%M_%S"
    FOLDER = f"{FOLDER}/{sub_folder}_{datetime.datetime.now().strftime(date_format)}"
    os.makedirs(FOLDER)

    evolved_weigths = None
    # fname = "E:/research/robustness_set/benchmarks/benchmark_18_05_2021_21_57_42/benchmark_upto_run_43_1621438657.8744707.pickle"

    # fname = "E:/research/robustness_set/benchmarks/benchmark_20_05_2021_19_12_14/benchmark_completed_1621574438.7710762.pickle"
    fname = None

    test_density = True

    if test_density:
        runs = 8
        n_training_epochs = 200
        # set_sparsity_levels = [1, 2, 3, 4, 5, 6, 13, 32, 64, 128, 256] # , 512, 1024]
        set_sparsity_levels = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31] # , 512, 1024]
        # the levels are chosen to have [0.16, 0.5, 1, 2, 5, 10, 20, 40, 80, 100] % density in the first layer
        use_logical_cores = False
        # FOLDER = "benchmarks/benchmark_02_06_2021_13_02_23"

        # fmnist_train_set_differnt_densities(runs, n_training_epochs, set_sparsity_levels, use_logical_cores=use_logical_cores)
        fmnist_train_set_differnt_densities_sequential(runs, n_training_epochs, set_sparsity_levels, use_logical_cores=use_logical_cores)

    else:
        benchmark = None
        if fname:
            with open(fname, "rb") as handle:
                pre_trained_set = pickle.load(handle)
                evolved_weights = pre_trained_set['evolved_weights']

                info = pre_trained_set['info']
                runs = info['runs']
                # sample_epochs = info['sample_weights']  # TODO: Name changed inside info
                sparseness_levels = info['sparseness_levels']

                use_logical_cores = True

                test_fmnist_pretrained_set(evolved_weights=evolved_weights, runs=runs)

        else:
            # all have dimensions (runs, models, n_sparseness_levels)
            runs = 32
            n_training_epochs = 400

            sample_epochs = [0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 399]

            sparseness_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 0.925, 0.95,
                                 0.975, 0.99, 0.995, 0.999]

            use_logical_cores = True
            use_pretrained = True

            benchmark = test_fmnist(runs=runs, sample_epochs=sample_epochs, n_training_epochs=n_training_epochs,
                                    sparseness_levels=sparseness_levels,
                                    use_logical_cores=use_logical_cores, use_pretrained=use_pretrained)

            '''
        print("Finished benchmark. Saving final results to disk")
        with open(f"{FOLDER}/benchmark_completed_{time.time()}.pickle", "wb") as handle:
            pickle.dump(benchmark, handle)
            '''

        # with open("benchmark.pickle", "rb") as handle:
        #    b2 = pickle.load(handle)

        # print(benchmark == b2)
