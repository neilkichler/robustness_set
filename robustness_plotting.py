import argparse
import bz2
import os
import pickle
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import psutil
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.sparse import save_npz

from utils_plotting import current_method_name, get_model_names

# NOTE(Neil): To quickly use a specific data folder change this variable
# DEFAULT_FOLDER = "benchmark_23_05_2021_13_38_50"
# DEFAULT_FOLDER = "benchmark_07_06_2021_22_27_28/"
# DEFAULT_FOLDER = "benchmark_08_06_2021_11_47_48"
DEFAULT_FOLDER = "fmnist_results_05_06_2021_02_00_15"

# FILE = "set_mlp_density_run_0.pickle"
# FILE = "set_mlp_density_run_0.pickle.pbz2"
# FILE = "fmnist_all_runs_no_weights.pickle"
# FILE = ""
FILE = "fmnist_results_0.pickle"
FOLDER = "RobustnessResults/new_result"
BENCHMARK_FOLDER = "benchmarks/"
BENCHMARK_PREFIX = "benchmark_"
# BENCHMARK_RUN_PREFIX = "fmnist_"
BENCHMARK_RUN_PREFIX = "set_mlp_density_run"
TOPOLOGY_FOLDER = "topo/"
DPI = 300
DPI_LIST = [300, 600, 1200]


def save_figs(fig, name):
    fig.savefig(f"{FOLDER}/{name}.png", bbox_inches='tight', dpi=DPI)

    for dpi in DPI_LIST:
        fig.savefig(f"{FOLDER}/{name}_dpi{dpi}.pdf", bbox_inches='tight', dpi=dpi)


def clear_console():
    cmd = 'clear'
    if os.name in ('nt', 'dos'):  # Windows detected
        cmd = 'cls'
    os.system(cmd)


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
    # last_epoch = -1
    # scores = scores[:, last_epoch, :]
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
    plt.xlabel(r"Sparsity [\%]")
    plt.ylabel(r"Accuracy [\%]")
    plt.legend()

    # TODO(Neil): Saving the plot is duplicated across the functions. Maybe make a master_function to do this for us
    #             maybe even with a decorator?
    if save_plot:
        save_figs(fig, current_method_name())

    if show_plot:
        plt.show()


def plot_sparsity_vs_accuracy_all_runs_base(scores, sparseness_levels, model_names, save_plot=False, show_plot=False,
                                            fname_prefix="", title=False):
    fig = plt.figure()
    percent = 100

    # This assumes that the runs are normally distributed!
    # For n > 30 or more we should be fine
    # scores = scores[:, last_epoch, :]
    means = np.mean(scores, axis=0)
    std = np.std(scores, axis=0)

    # for score in scores:

    colors = ["green", "red", "blue"]

    sparseness_levels_percent = np.asarray(sparseness_levels) * percent

    for i, model in enumerate(model_names):
        m_means = means[:, i]
        m_std = std[:, i]
        color = colors[i]
        # plt.fill_between(sparseness_levels_percent, (m_means - m_std) * percent,
        #                  (m_means + m_std) * percent, alpha=0.1,
        #                  color=color)

        plt.plot(sparseness_levels_percent, m_means * percent, color=color, label=model)
        plt.scatter(sparseness_levels_percent, m_means * percent, color=color)

    plt.title(fname_prefix)

    if title:
        plt.title("FMNIST Feature Selection Sparsity vs Accuracy")
    plt.xlabel(r"Dropped Features [\%]")
    plt.ylabel(r"Accuracy [\%]")
    plt.grid(linestyle="--")
    plt.legend()

    # TODO(Neil): Saving the plot is duplicated across the functions. Maybe make a master_function to do this for us
    #             maybe even with a decorator?
    if save_plot:
        save_figs(fig, fname_prefix + current_method_name())

    if show_plot:
        plt.show()


def plot_sparsity_vs_accuracy_all_runs(data, save_plot=False, show_plot=False, title=False):
    sparseness_levels = np.array(data[0]['sparseness_levels'])

    last_epoch = -1

    scores = np.array([d['scores'][last_epoch] for d in data])

    model_names = get_model_names(data[-1])
    plot_sparsity_vs_accuracy_all_runs_base(scores, sparseness_levels, model_names, save_plot=save_plot,
                                            show_plot=show_plot,
                                            title=title)


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
    # last_epoch = -1
    # scores = scores[:, last_epoch, :]
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
    plt.grid(linestyle='--')
    plt.legend()

    # TODO(Neil): Saving the plot is duplicated across the functions. Maybe make a master_function to do this for us
    #             maybe even with a decorator?
    if save_plot:
        save_figs(fig, current_method_name())

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
        save_figs(fig, current_method_name())

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
        save_figs(fig, current_method_name())

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


def plot_epoch_vs_accuracy_all_runs(data, save_plot=False, show_plot=False, title=False):
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
                         yerr=[yerr_low, yerr_high],  # color=color,
                         fmt='--o',  # alpha=0.1*(j + 1),
                         label=str(sample_epochs[j]))
            # , alpha=0.1, color=color, fmt='o')

        if title:
            plt.title(f"{model}: Features Dropped vs Accuracy (per epoch)")
        plt.xlabel(r"Features Dropped [\%]")
        plt.ylabel(r"Accuracy [\%]")
        plt.legend(fontsize=12, loc="lower left", ncol=3)
        plt.grid(linestyle='--')

        if save_plot:
            save_figs(fig, model + "_" + current_method_name())

        if show_plot:
            plt.show()


def plot_feature_selection_aggregate_per_epoch(data, title=False, show_plot=False, show_cbar=False, save_plot=False):
    percent = 100
    image_dim = (28, 28)

    sample_epochs = data[0]['sample_epochs']

    sparseness_levels = data[0]['sparseness_levels']
    sparseness_level = sparseness_levels.index(0.7)

    fig = plt.figure(dpi=300)

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

        grid[i].set_title(f"Epoch: {sample_epochs[i]}", fontsize=10)

    cbar = grid.cbar_axes[0].colorbar(im)

    cbar.set_label_text("prevalence")  # of features")

    for cax in grid.cbar_axes:
        cax.toggle_label(True)

    # This affects all axes because we set share_all = True.
    grid.axes_llc.set_xticks([])
    grid.axes_llc.set_yticks([])

    if save_plot:
        save_figs(fig, current_method_name())

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
        current_ax.set_title(f"Sparse: {int(s * percent)} %")
        # current_ax.set_xlabel("12")

    # plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_selection_aggregate_separate_runs(data, title=False, show_plot=False, show_cbar=False,
                                                   save_plot=False):
    percent = 100
    image_dim = (28, 28)
    fig = plt.figure(dpi=300)

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
    cbar.set_label_text("prevalence")  # of features")

    for cax in grid.cbar_axes:
        cax.toggle_label(True)

    # This affects all axes because we set share_all = True.
    grid.axes_llc.set_xticks([])
    grid.axes_llc.set_yticks([])

    if save_plot:
        save_figs(fig, current_method_name())

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
        save_figs(fig, current_method_name())

    if show_plot:
        plt.show()


# TODO: we may want to work with kwargs instead of passing all the options separately
def plot_accuracy_different_densities(data, show_plot=False, save_plot=False, title=False):
    info = data['set']
    density_levels = info['density_levels']
    scores = data['scores']
    sparseness_levels = data['sparseness_levels']
    models = get_model_names(data)

    fig = plt.figure(dpi=300)

    if title:
        fig.suptitle(f'Feature Selection Densities FMNIST')

    for density_level in range(scores.shape[0]):
        # for now we only have a single value:
        score = [scores[density_level][-1]]
        plot_sparsity_vs_accuracy_all_runs_base(score, sparseness_levels, models, title=title, show_plot=show_plot,
                                                fname_prefix=f"{density_levels[density_level]}", save_plot=save_plot)

    if save_plot:
        save_figs(fig, current_method_name())

    if show_plot:
        plt.show()


def save_topology_single_run(folder, f):
    frun_name = f.partition(".pickle")[0]
    fname = folder + "/" + f
    with open(fname, "rb") as handle:
        benchmark = pickle.load(handle)
        # weight_layers = benchmark['set']['evolved_weights']
        weight_layers = benchmark['evolved_weights']
        sample_epochs = benchmark['sample_epochs']

        for sample in sample_epochs:

            sample_folder = f"{full_folder_name}/{TOPOLOGY_FOLDER}{frun_name}/epoch_%.3d" % sample
            if not os.path.exists(sample_folder):
                os.makedirs(sample_folder)

            for i, layer in enumerate(weight_layers[sample].values()):
                topo_fname = f"{sample_folder}/weight_l{i}"
                save_npz(topo_fname, layer)


def save_topology(folder, flist, use_logical_cores=False, debug_info=True):
    print(f"Starting to save topologies of {len(flist)} files")
    n_cores = psutil.cpu_count(logical=use_logical_cores)
    with Pool(processes=n_cores) as pool:

        futures = [pool.apply_async(save_topology_single_run, (folder, frun)) for frun in flist]

        n_futures = len(futures)
        for i, future in enumerate(futures):
            future.get()
            if debug_info:
                print(f"Finished saving weight-layers: ({i + 1}/{n_futures})")


def default_args_parser():
    def add_bool_arg(parse, name, default=False, help_msg=''):
        group = parse.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true', help='Enable: ' + help_msg)
        group.add_argument('--no-' + name, dest=name, action='store_false', help='Disable: ' + help_msg)
        parse.set_defaults(**{name: default})

    p = argparse.ArgumentParser(description='Plotting tool')

    add_bool_arg(p, 'plotting', default=True, help_msg='plotting')
    add_bool_arg(p, 'title', default=False, help_msg='Add title to figure')
    add_bool_arg(p, 'show_plots', default=True, help_msg='Show plots while running')
    add_bool_arg(p, 'save_plots', default=False, help_msg='Save all plots')
    add_bool_arg(p, 'fixup', help_msg='Try to fixup data before plotting')
    add_bool_arg(p, 'save_topo', help_msg='Save the topology of all runs')
    add_bool_arg(p, 'save_no_weights',
                 help_msg='Save all runs without the SET weights. Reduces memory footprint (default: "").')

    p.add_argument('--nprocessor', default=psutil.cpu_count(logical=False), type=int,
                   help='# processors for calculation')

    p.add_argument('--data_path', metavar='DIR', default=BENCHMARK_FOLDER,
                   help=f'path to data (default: {BENCHMARK_FOLDER}/)')

    p.add_argument('--folder_prefix', metavar='DIR', default=BENCHMARK_PREFIX,
                   help=f"Only search folders with specific prefix for data (default: {BENCHMARK_PREFIX}).")

    p.add_argument('--file_prefix', metavar='DIR', default=BENCHMARK_RUN_PREFIX,
                   help=f"Only search files with specific prefix (default: {BENCHMARK_RUN_PREFIX}).")

    p.add_argument('--file', metavar='fname', default=FILE, help=f"Use a specific file (default: {FILE}).")

    p.add_argument('--folder_path', metavar='DIR', default=DEFAULT_FOLDER,
                   help=f"select a specific folder to run on. "
                        "If not set, the latest created folder is taken (default: {DEFAULT_FOLDER}).")

    p.add_argument('--topo_path', metavar='DIR', default=TOPOLOGY_FOLDER,
                   help=f'path to topo folder (default: {TOPOLOGY_FOLDER})')
    p.add_argument('--save_path', metavar='DIR', default=FOLDER, help=f'path to save plots in (default: {FOLDER})')
    return p


def load_file(fname, save_uncompressed=False):
    if fname.endswith(".pbz2"):
        with bz2.BZ2File(fname, 'r') as h:
            data = pickle.load(h)

        if save_uncompressed:
            with open(fname[:-5], "wb") as h:
                pickle.dump(all_runs, h)
    else:
        with open(fname, "rb") as h:
            data = pickle.load(h)

    return data


if __name__ == "__main__":

    parser = default_args_parser()
    args = parser.parse_args()

    save_all_runs_no_weights = args.save_no_weights
    save_topology_all_runs = args.save_topo
    plotting = args.plotting
    save_plots = args.save_plots
    show_plots = args.show_plots
    topo_path = args.topo_path
    title = args.title
    fixup = args.fixup
    data_path = args.data_path
    default_folder = args.folder_path

    if not default_folder.endswith("/"):
        default_folder += "/"
    benchmark_prefix = args.folder_prefix
    benchmark_run_prefix = args.file_prefix

    if not os.path.exists(data_path):
        raise ValueError(f"The specified data path does not exist: {data_path}")

    if default_folder:
        latest_benchmark_folder = default_folder
    else:
        benchmark_folders = os.listdir(data_path)
        # pick the most recent benchmark
        latest_benchmark_folder = sorted(filter(lambda x: x.startswith(benchmark_prefix), benchmark_folders))[-1]

    full_folder_name = data_path + latest_benchmark_folder

    # TODO(Neil): Only use this if it actually exists
    flist = []

    if args.file:
        fname = full_folder_name + args.file
    else:
        flist = os.listdir(full_folder_name)

        completed_list = sorted(filter(lambda x: x.startswith(benchmark_prefix + "completed"), flist))

        if completed_list:
            flist = completed_list
        else:  # only partially completed benchmarks exist
            # flist = sorted(filter(lambda x: x.startswith(benchmark_prefix), flist))
            flist = sorted(filter(lambda x: x.startswith(benchmark_run_prefix), flist))
        fname = flist[-1]

    sample_epochs = [0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 399]

    # fixup
    if fixup:
        for f in flist:
            fname = full_folder_name + '/' + f
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

    if save_all_runs_no_weights:
        all_runs = []
        for frun in flist:
            fname = full_folder_name + frun

            run = load_file(fname)
            run['set']['evolved_weights'] = []
            all_runs.append(run)

        with open(full_folder_name + topo_path + save_all_runs_no_weights, "wb") as h:
            pickle.dump(all_runs, h)

    if save_topology_all_runs:
        if not os.path.exists(full_folder_name + topo_path):
            os.makedirs(full_folder_name + topo_path)

        save_topology(full_folder_name, flist[1:])

    if plotting:
        all_runs = load_file(fname)

        plt.rc('font', size=22)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plot_accuracy_different_densities(all_runs, title=title, show_plot=show_plots, save_plot=save_plots)
        # plot_feature_selection_aggregate_separate_runs(all_runs, title=title, show_plot=show_plots,
        #                                                save_plot=save_plots)
        # plot_sparsity_vs_accuracy_all_runs(all_runs, show_plot=save_plots, save_plot=save_plots)
        # plot_epoch_vs_accuracy_all_runs(all_runs, show_plot=save_plots, save_plot=save_plots, title=title)
        # plot_feature_selection_aggregate_per_epoch(all_runs, show_plot=show_plots, save_plot=save_plots, title=title)

        # plt.rcParams['font.family'] = 'serif'
        # plt.rcParams['font.serif'] = 'Computer Modern Roman'
        # plt.figure()
        # plt.title("Testing")
        # plt.show()
