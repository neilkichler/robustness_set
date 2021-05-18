import os
import pickle
import inspect

import matplotlib.pyplot as plt
import numpy as np

FOLDER = "RobustnessResults"
BENCHMARK_FOLDER = "benchmarks/"


def current_method_name():
    # [0] is this method's frame, [1] is the parent's frame - which we want
    return inspect.stack()[1].function


def get_model_names(data):

    models = data['models']
    model_names = [type(m).__name__ for m in models]
    return model_names


def plot_sparsity_vs_accuracy(data, save_plot=False, show_plot=False):

    fig = plt.figure()

    info = data['info']

    n_sparseness_level = info['n_sparseness_level']
    scores = data['scores']
    score0 = scores[0]
    model_names = get_model_names(data)

    percent = 100

    # This assumes that the runs are normally distributed!
    # For n > 30 or more we should be fine
    #last_epoch = -1
    #scores = scores[:, last_epoch, :]
    means = np.mean(scores, axis=0)
    std = np.std(scores, axis=0)

    # for score in scores:

    colors = ["green", "red", "blue"]

    for i, model in enumerate(model_names):
        m_means = means[:, i]
        m_std = std[:, i]
        color = colors[i]
        plt.fill_between(np.linspace(0.1, 1, len(means[:, -1])) * percent, (m_means - m_std) * percent,
                         (m_means + m_std) * percent, alpha=0.1,
                             color=color)

        plt.plot(np.linspace(0.1, 1, 10) * percent, m_means * percent, color=color, label=model)

    plt.title("Sparsity vs Accuracy")
    plt.xlabel("features selected [%]")
    plt.ylabel("Accuracy [%]")
    plt.legend()

    # TODO(Neil): Saving the plot is duplicated across the functions. Maybe make a master_function to do this for us
    #             maybe even with a decorator?
    if save_plot:
        fig.savefig(f"{FOLDER}/{current_method_name()}.pdf", bbox_inches='tight')

    if show_plot:
        plt.show()


def plot_sparsity_vs_time(data, save_plot=False, show_plot=False):

    fig = plt.figure()

    times = data['times']
    model_names = get_model_names(data)
    percent = 100

    means = np.mean(times, axis=0)
    std = np.std(times, axis=0)

    # maybe add a dict with model to color?
    colors = ["green", "red", "blue"]

    for i, model in enumerate(model_names):
        m_means = means[:, i]
        m_std = std[:, i]

        color = colors[i]
        plt.fill_between(np.linspace(0.1, 1, 10) * percent, (m_means - m_std) * percent,
                         (m_means + m_std) * percent, alpha=0.1,
                         color=color)

        plt.plot(np.linspace(0.1, 1, 10) * percent, m_means * percent, color=color, label=model)

    plt.title("Sparsity vs Time")
    plt.xlabel("features selected [%]")
    plt.ylabel("Training time [ms]")
    plt.legend()

    if save_plot:
        fig.savefig(f"{FOLDER}/{current_method_name()}.pdf", bbox_inches='tight')

    if show_plot:
        plt.show()


def plot_feature_selection_per_run(data):

    percent = 100
    image_dim = (28, 28)
    features = data['selected_features']

    # let's look at only one run at a time
    for i, f in enumerate(features):
        fig, ax = plt.subplots(3, 3)
        fig.suptitle(f'Feature Selection FMNIST Run: {i}')
        for i in range(3):
            for j in range(3):
                k = 3 * i + j
                f_data = np.reshape(f[k], image_dim)
                current_ax = ax[i, j]
                current_ax.imshow(f_data, vmin=0, vmax=1, cmap="gray_r", interpolation=None)
                current_ax.set_title(f"Sparse: {10 + k * 10} %")
                # current_ax.set_xlabel("12")

        plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.3)
        plt.show()


def plot_feature_selection_aggregate(data, show_plot=False, show_cbar=False, save_plot=False):
    image_dim = (28, 28)
    features = data['selected_features']

    # plot the aggregate
    fig, axs = plt.subplots(nrows=3, ncols=3)
    fig.suptitle(f'Feature Selection Aggregate FMNIST (6 Runs avg.)')

    # save image handle for color-bar
    im = None

    for i, ax in enumerate(axs.flat):
        agg = np.zeros(784)

        for f in features:
            agg += f[i]

        f_data = np.reshape(agg, image_dim)
        im = ax.imshow(f_data, vmin=0, vmax=np.max(f_data), cmap="Greys", interpolation=None)
        ax.set_title(f"Sparse: {10 + i * 10} %")

    if show_cbar:
        cbar = fig.colorbar(im, ax=axs)  # .ravel().tolist())
        cbar.set_label("Avg. Feature Strength")
    else:
        plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.3)

    if save_plot:
        fig.savefig(f"{FOLDER}/{current_method_name()}.pdf", bbox_inches='tight')

    if show_plot:
        plt.show()


if __name__ == "__main__":

    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    benchmark_prefix = "benchmark_"

    benchmark_folders = os.listdir(BENCHMARK_FOLDER)
    # pick the most recent benchmark
    latest_benchmark_folder = sorted(filter(lambda x: x.startswith(benchmark_prefix), benchmark_folders))[-1]

    full_folder_name = BENCHMARK_FOLDER + latest_benchmark_folder
    flist = os.listdir(full_folder_name)

    completed_list = sorted(filter(lambda x: x.startswith(benchmark_prefix + "completed"), flist))

    if completed_list:
        flist = completed_list
    else:  # only partially completed benchmarks exist
        flist = sorted(filter(lambda x: x.startswith(benchmark_prefix), flist))

    # always use the most complete/latest benchmark by default
    fname = full_folder_name + '/' + flist[-1]

    # fname = "benchmark_1621249372.1620104.pickle"

    with open(fname, "rb") as handle:
        benchmark = pickle.load(handle)

        print(benchmark["models"])

        plot_sparsity_vs_accuracy(benchmark, show_plot=True, save_plot=True)
        plot_sparsity_vs_time(benchmark, save_plot=True)
        plot_feature_selection_aggregate(benchmark, show_plot=False, save_plot=True)
        plt.show()
