import re
import os
import pickle
import inspect

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import numpy as np

FOLDER = "RobustnessResults/new_result"
BENCHMARK_FOLDER = "benchmarks/"
EXTENSION = ".png"
DPI = 300


def current_method_name():
    # [0] is this method's frame, [1] is the parent's frame - which we want
    return inspect.stack()[1].function


def get_model_names(data):

    models = data['models']
    model_names = [type(m).__name__ for m in models]
    return model_names



def plot_sparsity_vs_accuracy_single(data, save_plot=False, show_plot=False):

    fig = plt.figure()


    sparseness_levels = np.array(data['sparseness_levels'])
    n_sparseness_level = len(sparseness_levels)
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

    sparseness_levels_percent = sparseness_levels * percent

    for i, model in enumerate(model_names):
        m_means = means[100, :, i]
        m_std = std[100, :, i]
        color = colors[i]
        plt.fill_between(sparseness_levels_percent, (m_means - m_std) * percent,
                         (m_means + m_std) * percent, alpha=0.1,
                             color=color)

        plt.plot(sparseness_levels_percent, m_means * percent, color=color, label=model)

    plt.title("Sparsity vs Accuracy")
    plt.xlabel("Sparsity [%]")
    plt.ylabel("Accuracy [%]")
    plt.legend()

    # TODO(Neil): Saving the plot is duplicated across the functions. Maybe make a master_function to do this for us
    #             maybe even with a decorator?
    if save_plot:
        fig.savefig(f"{FOLDER}/{current_method_name()}{EXTENSION}", bbox_inches='tight')

    if show_plot:
        plt.show()



def plot_sparsity_vs_accuracy_all_runs(data, save_plot=False, show_plot=False):

    fig = plt.figure()

    sparseness_levels = np.array(data[0]['sparseness_levels'])
    n_sparseness_level = len(sparseness_levels)

    last_epoch = -1

    scores = np.array([d['scores'][last_epoch] for d in data])

    # score0 = scores[0]
    model_names = get_model_names(data[-1])

    percent = 100

    # This assumes that the runs are normally distributed!
    # For n > 30 or more we should be fine
    #scores = scores[:, last_epoch, :]
    means = np.mean(scores, axis=0)
    std = np.std(scores, axis=0)

    # for score in scores:

    colors = ["green", "red", "blue"]

    sparseness_levels_percent = sparseness_levels * percent

    for i, model in enumerate(model_names):
        m_means = means[:, i]
        m_std = std[:, i]
        color = colors[i]
        plt.fill_between(sparseness_levels_percent, (m_means - m_std) * percent,
                         (m_means + m_std) * percent, alpha=0.1,
                             color=color)

        plt.plot(sparseness_levels_percent, m_means * percent, color=color, label=model)
        plt.scatter(sparseness_levels_percent, m_means * percent, color=color)

    plt.title("FMNIST Feature Selection Sparsity vs Accuracy")
    plt.xlabel("Sparsity [%]")
    plt.ylabel("Accuracy [%]")
    plt.legend()

    # TODO(Neil): Saving the plot is duplicated across the functions. Maybe make a master_function to do this for us
    #             maybe even with a decorator?
    if save_plot:
        fig.savefig(f"{FOLDER}/{current_method_name()}{EXTENSION}", bbox_inches='tight', dpi=DPI)

    if show_plot:
        plt.show()



def plot_sparsity_vs_accuracy(data, save_plot=False, show_plot=False):

    fig = plt.figure()

    info = data['info']

    sparseness_levels = np.array(data['sparseness_levels'])
    n_sparseness_level = len(sparseness_levels)
    n_runs = info['runs']
    scores = data['scores'][0:n_runs, :, :]

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

    sparseness_levels_percent = sparseness_levels * percent

    for i, model in enumerate(model_names):
        m_means = means[100, :, i]
        m_std = std[100, :, i]
        color = colors[i]
        plt.fill_between(sparseness_levels_percent, (m_means - m_std) * percent,
                         (m_means + m_std) * percent, alpha=0.1,
                             color=color)

        plt.plot(sparseness_levels_percent, m_means * percent, color=color, label=model)

    plt.title("Sparsity vs Accuracy")
    plt.xlabel("Sparsity [%]")
    plt.ylabel("Accuracy [%]")
    plt.legend()

    # TODO(Neil): Saving the plot is duplicated across the functions. Maybe make a master_function to do this for us
    #             maybe even with a decorator?
    if save_plot:
        fig.savefig(f"{FOLDER}/{current_method_name()}{EXTENSION}", bbox_inches='tight')

    if show_plot:
        plt.show()

def plot_sparsity_vs_time_all_runs(data, save_plot=False, show_plot=False):

    fig = plt.figure()

    # TODO(Neil): Time is actually in micro not milliseconds. Divide by another 1000!

    percent = 100

    sparseness_levels = np.array(data[0]['sparseness_levels'])
    n_sparseness_level = len(sparseness_levels)

    last_epoch = -1

    times = np.array([d['times'][last_epoch] for d in data]) / (1000 * 1000)

    # score0 = scores[0]
    model_names = get_model_names(data[-1])


    sparseness_levels_percent = sparseness_levels * percent
    means = np.mean(times, axis=0)
    std = np.std(times, axis=0)

    # maybe add a dict with model to color?
    colors = ["green", "red", "blue"]

    for i, model in enumerate(model_names):
        m_means = means[:, i]
        m_std = std[:, i]

        color = colors[i]
        plt.fill_between(sparseness_levels_percent, (m_means - m_std) * percent,
                         (m_means + m_std) * percent, alpha=0.1,
                         color=color)

        plt.plot(sparseness_levels_percent, m_means * percent, color=color, label=model)

    plt.title("Sparsity vs Time")
    plt.xlabel("Sparsity [%]")
    plt.ylabel("Training time [ms]")
    plt.legend()

    if save_plot:
        fig.savefig(f"{FOLDER}/{current_method_name()}{EXTENSION}", bbox_inches='tight', dpi=DPI)

    if show_plot:
        plt.show()


def plot_sparsity_vs_time(data, save_plot=False, show_plot=False):

    fig = plt.figure()

    info = data['info']
    n_runs = info['runs']

    # TODO(Neil): Time is actually in micro not milliseconds. Divide by another 1000!

    times = data['times'][0:n_runs, :, :]
    model_names = get_model_names(data)
    percent = 100

    means = np.mean(times, axis=0)
    std = np.std(times, axis=0)

    sparseness_levels = np.array(info['sparseness_levels'])
    sparseness_levels_percent = sparseness_levels * percent

    # maybe add a dict with model to color?
    colors = ["green", "red", "blue"]

    for i, model in enumerate(model_names):
        m_means = means[10, :, i]
        m_std = std[10, :, i]

        color = colors[i]
        plt.fill_between(sparseness_levels_percent, (m_means - m_std) * percent,
                         (m_means + m_std) * percent, alpha=0.1,
                         color=color)

        plt.plot(sparseness_levels_percent, m_means * percent, color=color, label=model)

    plt.title("Sparsity vs Time")
    plt.xlabel("Sparsity [%]")
    plt.ylabel("Training time [ms]")
    plt.legend()

    if save_plot:
        fig.savefig(f"{FOLDER}/{current_method_name()}{EXTENSION}", bbox_inches='tight')

    if show_plot:
        plt.show()


def plot_feature_selection_single_instance(data):
    percent = 100
    image_dim = (28, 28)
    features = data['selected_features']

    sparseness_levels = data['sparseness_levels']

    selected_sparseness = np.linspace(0.1, 1, 10)

    # let's look at only one run at a time
    # for i, f in enumerate(features):

    f = features[5]  # 00, :, :]
    # f = f[50]

    # fig.suptitle(f'Feature Selection FMNIST Run:')  # {i}')
    for i, s in enumerate(sparseness_levels):
        if s not in selected_sparseness:
            continue
        # k = 3 * i + j
        f_data = np.reshape(f[i], image_dim)
        # current_ax = a  # x[i, j]
        plt.imshow(f_data, vmin=0, vmax=1, cmap="gray_r", interpolation=None)
        # current_ax.set_title(f"Sparse: {sparseness_levels[i]} %")
        # current_ax.set_xlabel("12")
        plt.show()



def plot_epoch_vs_accuracy_all_runs(data, save_plot=False, show_plot=False):
    percent = 100

    sparseness_levels = np.array(data[0]['sparseness_levels'])
    sample_epochs = np.array(data[0]['sample_epochs'])
    n_sample_epochs = len(sample_epochs)
    n_sparseness_level = len(sparseness_levels)

    last_epoch = -1

    model_names = get_model_names(data[-1])

    sparseness_levels_percent = sparseness_levels * percent

    # maybe add a dict with model to color?
    colors = ["green", "red", "blue"]

    for i, model in enumerate(model_names):
        fig = plt.figure()
        for j in range(n_sample_epochs - 4):
            scores = np.array([d['scores'][j] for d in data])
            means = np.mean(scores, axis=0)
            std = np.std(scores, axis=0)


            m_means = means[0:10, i]
            m_std = std[0:10, i]

            color = colors[i]

            types = [str(j)] * 11
            # plt.scatter(sparseness_levels_percent[0:10], m_means * percent, color=color, alpha=0.1*(j + 1), label=str(sample_epochs[j]))
            yerr_low = (m_means - m_std)
            yerr_high = (m_means + m_std)
            plt.errorbar(sparseness_levels_percent[0:10], m_means * percent,
                         yerr=[yerr_low, yerr_high], #color=color,
                         fmt='--o', # alpha=0.1*(j + 1),
                         label=str(sample_epochs[j]))
                         # , alpha=0.1, color=color, fmt='o')

        plt.title(f"{model}: Sparsity vs Accuracy (per epoch)")
        plt.xlabel("Sparsity [%]")
        plt.ylabel("Accuracy [%]")
        plt.legend()

        if save_plot:
            fig.savefig(f"{FOLDER}/{model}_{current_method_name()}{EXTENSION}", bbox_inches='tight', dpi=300)

        if show_plot:
            plt.show()


def plot_feature_selection_aggregate_per_epoch(data, title=False, show_plot=False, show_cbar=False, save_plot=False):
    percent = 100
    image_dim = (28, 28)


    sample_epochs = data[0]['sample_epochs']

    sparseness_levels = data[0]['sparseness_levels']
    sparseness_level = sparseness_levels.index(0.7)

    fig = plt.figure(dpi=300.0)

    sparseness_levels = data[0]['sparseness_levels']
    selected_sparseness_idx = [1, 2, 3, 4, 5, 6, 7, 8, 10]

    if title:
        fig.suptitle(f'Feature Selection FMNIST ({len(data)} runs, 70% sparsity)')

    # NOTE(Neil): could now also be -1
    LAST_EPOCH = 12

    aggs = np.array([d['selected_features'][:, sparseness_level, :] for d in data]).sum(axis=0)

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(3, 3),
                    axes_pad=0.25,
                    share_all=True,
                    label_mode="1",
                    cbar_location="right",
                    cbar_mode="single",
                    )

    for i in range(len(grid)):
        agg = aggs[i]
        f_data = np.reshape(agg, image_dim)
        im = grid[i].imshow(f_data, vmin=0, vmax=np.max(f_data), cmap="gray_r", interpolation=None)

        grid[i].set_title(f"Epoch: {sample_epochs[i]}")


    cbar = grid.cbar_axes[0].colorbar(im)


    cbar.set_label_text("prevalence") # of features")

    for cax in grid.cbar_axes:
        cax.toggle_label(True)

    # This affects all axes because we set share_all = True.
    grid.axes_llc.set_xticks([])
    grid.axes_llc.set_yticks([])

    if save_plot:
        fig.savefig(f"{FOLDER}/{current_method_name()}{EXTENSION}", bbox_inches='tight', dpi=DPI)

    if show_plot:
        plt.show()






def plot_feature_selection_per_epoch(data):

    percent = 100
    image_dim = (28, 28)
    features = data['selected_features']
    sample_epochs = data['sample_epochs']

    sparseness_levels = data['sparseness_levels']
    sparseness_level = sparseness_levels.index(0.7)

    f = features[:, sparseness_level, :]

    fig, ax = plt.subplots(3, 3)
    # fig.suptitle(f'Evolution Feature Selection FMNIST:')  # {i}')
    for i, a in enumerate(ax.flat):
        f_data = np.reshape(f[i], image_dim)
        a.imshow(f_data, vmin=0, vmax=1, cmap="gray_r", interpolation=None)
        a.set_title(f"Epoch: {sample_epochs[i]}")

    # plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.3)
    plt.tight_layout()
    plt.show()





def plot_feature_selection_per_run(data):

    percent = 100
    image_dim = (28, 28)
    features = data['selected_features']

    sparseness_levels = data['sparseness_levels']
    selected_sparseness_idx = [1, 2, 3, 4, 5, 6, 7, 8, 10]

    f = features[12]
    # second plot
    # f = features[:, 4, :]

    # f = features[0]  # 00, :, :]
    # f = f[0]

    fig, ax = plt.subplots(3, 3)
    # fig.suptitle(f'Feature Selection FMNIST Run:')  # {i}')
    for i, a in enumerate(ax.flat):
        idx = selected_sparseness_idx[i]
        s = sparseness_levels[idx]

        f_data = np.reshape(f[idx], image_dim)
        current_ax = a  # x[i, j]
        current_ax.imshow(f_data, vmin=0, vmax=1, cmap="gray_r", interpolation=None)
        current_ax.set_title(f"Sparse: {int(s*percent)} %")
        # current_ax.set_xlabel("12")

    # plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_selection_aggregate_separate_runs(data, title=False, show_plot=False, show_cbar=False, save_plot=False):
    percent = 100
    image_dim = (28, 28)
    fig = plt.figure(dpi=300.0)

    sparseness_levels = data[0]['sparseness_levels']
    selected_sparseness_idx = [1, 2, 3, 4, 5, 6, 7, 8, 10]

    if title:
        fig.suptitle(f'Feature Selection FMNIST ({len(data)} runs, sparsity=0.1,...,0.9)')

    # NOTE(Neil): could now also be -1
    LAST_EPOCH = 12

    aggs = np.array([d['selected_features'][LAST_EPOCH] for d in data]).sum(axis=0)

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(3, 3),
                    axes_pad=0.1,
                    share_all=True,
                    label_mode="1",
                    cbar_location="right",
                    cbar_mode="single",
                    )

    for i in range(len(grid)):
        idx = selected_sparseness_idx[i]
        s = sparseness_levels[idx]

        agg = aggs[idx]
        f_data = np.reshape(agg, image_dim)

        im = grid[i].imshow(f_data, vmin=0, vmax=np.max(f_data), cmap="Greys", interpolation=None)

    # plt.colorbar(im, cax = grid.cbar_axes[0])
    cbar = grid.cbar_axes[0].colorbar(im)


    cbar.set_label_text("prevalence") # of features")

    for cax in grid.cbar_axes:
        cax.toggle_label(True)

    # This affects all axes because we set share_all = True.
    grid.axes_llc.set_xticks([])
    grid.axes_llc.set_yticks([])

    if save_plot:
        fig.savefig(f"{FOLDER}/{current_method_name()}{EXTENSION}", bbox_inches='tight', dpi=DPI)

    if show_plot:
        plt.show()







def plot_feature_selection_aggregate(data, show_plot=False, show_cbar=False, save_plot=False):
    percent = 100
    image_dim = (28, 28)
    features = data['selected_features']

    features = features[100, :, :]

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
        fig.savefig(f"{FOLDER}/{current_method_name()}{EXTENSION}", bbox_inches='tight')

    if show_plot:
        plt.show()


if __name__ == "__main__":

    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    benchmark_prefix = "benchmark_"
    benchmark_run_prefix = "fmnist_"

    benchmark_folders = os.listdir(BENCHMARK_FOLDER)
    # pick the most recent benchmark
    latest_benchmark_folder = sorted(filter(lambda x: x.startswith(benchmark_prefix), benchmark_folders))[-1]

    full_folder_name = BENCHMARK_FOLDER + latest_benchmark_folder
    flist = os.listdir(full_folder_name)

    completed_list = sorted(filter(lambda x: x.startswith(benchmark_prefix + "completed"), flist))

    if completed_list:
        flist = completed_list
    else:  # only partially completed benchmarks exist
        # flist = sorted(filter(lambda x: x.startswith(benchmark_prefix), flist))
        flist = sorted(filter(lambda x: x.startswith(benchmark_run_prefix), flist))

    sample_epochs = [0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 399]

    FID = 5
    # fixup
    fixup = False
    if fixup:
        for f in flist:
        # f = flist[FID]
            fname = full_folder_name + '/' + f
            benchmark = None
            with open(fname, "rb") as h:
                benchmark = pickle.load(h)
                benchmark['sample_epochs'] = sample_epochs
                a = benchmark['dimensions']
                n_sample_epochs = len(sample_epochs)
                benchmark['dimensions'] = (n_sample_epochs, a[1], a[2])
                benchmark['scores'] = benchmark['scores'][0:n_sample_epochs, :, :]
                benchmark['times'] = benchmark['times'][0:n_sample_epochs, :, :]
                benchmark['selected_features'] = benchmark['selected_features'][0:n_sample_epochs]

            with open(fname, "wb") as h:
                pickle.dump(benchmark, h)

            print(f"finished with {f}")

    # Note(Neil): You can also overwrite the fname here
    # fname = "benchmark_1621249372.1620104.pickle"

    save_all_runs_no_weights = False
    fname_all_runs = full_folder_name + "/fmnist_all_runs_no_weights.pickle"

    if save_all_runs_no_weights:
        all_runs = []
        for frun in flist:
            fname = full_folder_name + '/' + frun

            with open(fname, "rb") as handle:
                benchmark = pickle.load(handle)
                benchmark['set']['evolved_weights'] = []
                all_runs.append(benchmark)

            '''
            # This info should have been in the benchmark
            # TODO(Neil): really not needed anymore
            if not completed_list:
                n_runs = int(re.search(r'\d+', flist[-1]).group())
                benchmark['info']['runs'] = n_runs
            else:
                benchmark['info']['runs'] += 1

            print(benchmark["models"])
            '''
        with open(fname_all_runs, "wb") as h:
            pickle.dump(all_runs, h)

    all_runs = []

    with open(fname_all_runs, "rb") as h:
        all_runs = pickle.load(h)


    # TODO(Neil): time plot seems very noisy? Maybe due to parralel execution?
    #             it isn't very interesting for our study anyways
    plot_sparsity_vs_time_all_runs(all_runs, show_plot=True, save_plot=True)
    plot_feature_selection_aggregate_separate_runs(all_runs, title=True, show_plot=True, save_plot=True)
    plot_sparsity_vs_accuracy_all_runs(all_runs, show_plot=True, save_plot=True)
    plot_epoch_vs_accuracy_all_runs(all_runs, show_plot=True, save_plot=True)
    plot_feature_selection_aggregate_per_epoch(all_runs, show_plot=True,
            save_plot=True, title=True)
    # plt.show()
