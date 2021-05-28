import os
import pickle

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from utils_plotting import current_method_name, feature_selection_mean

FOLDER = "benchmarks"
FOLDER_NEW = ""
SUB_FOLDER = "set_worst_case"
SUB_FOLDER_PRETRAINED_WEIGHTS = "benchmark"
RUN_PREFIX = "set_mlp_run"
EXTENSION = ".pickle"

# Figure options
FIG_EXTENSION = ".png"
FIG_FOLDER = "RobustnessResults/new_result"
DPI = 300

# Here you can select the time when the benchmark was made
# PRETRAINED_FOLDER_TIME = "26_05_2021_01_44_48"
PRETRAINED_FOLDER_TIME = "27_05_2021_00_06_58"


def plot_feature_selection_worst_case_per_epoch(aggs, n_runs, sample_epochs, title=False, show_plot=False, show_cbar=False, save_plot=False):
    percent = 100
    image_dim = (28, 28)

    fig = plt.figure(dpi=DPI)

    if title:
        fig.suptitle(f'Feature Selection Worst Case FMNIST ({n_runs} runs, 70% sparsity)')

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(3, 3),
                    axes_pad=0.25,
                    share_all=True,
                    label_mode="1",
                    cbar_location="right",
                    cbar_mode="single",
                    )

    im = None

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
        fig.savefig(f"{FIG_FOLDER}/{current_method_name()}{FIG_EXTENSION}", bbox_inches='tight', dpi=DPI)

    if show_plot:
        plt.show()


def perform_feature_selection(fname, benchmark, sample_epochs, sparseness_levels):
    n_sparseness_levels = len(sparseness_levels)

    evolved_weights = benchmark['evolved_weights']
    n_features = evolved_weights[0][1].shape[0]
    n_sample_epochs = len(sample_epochs)
    selected_features = np.zeros((n_sample_epochs, n_sparseness_levels, n_features))

    for i, epoch in enumerate(sample_epochs):
        for j, sparsity in enumerate(sparseness_levels):
            first_layer = evolved_weights[epoch][1]
            selected_indices = feature_selection_mean(weights=first_layer, sparsity=sparsity)
            selected_features[i][j] = selected_indices

    benchmark['selected_features'] = selected_features

    with open(fname, "wb") as h:
        pickle.dump(benchmark, h)


if __name__ == "__main__":

    full_folder_name = f"{FOLDER}/{SUB_FOLDER}_{PRETRAINED_FOLDER_TIME}"
    flist = os.listdir(full_folder_name)

    n_runs = len(flist)
    n_training_epochs = 400
    use_logical_cores = True
    should_perform_feature_selection = True

    sample_epochs = [0, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 399]

    sparseness_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999]
    sparseness_level = sparseness_levels.index(0.7)
    aggs = None
    for frun in flist:
        if not frun.startswith(RUN_PREFIX):
            continue
        fname = full_folder_name + '/' + frun

        with open(fname, "rb") as handle:
            benchmark = pickle.load(handle)
            benchmark['sample_epochs'] = sample_epochs
            benchmark['sparseness_levels'] = sparseness_levels

            if should_perform_feature_selection:
                perform_feature_selection(fname, benchmark, sample_epochs, sparseness_levels)

        # load data once again after feature selection
        with open(fname, "rb") as h:
            benchmark = pickle.load(h)
            if aggs is None:
                aggs = benchmark['selected_features'][:, sparseness_level, :]
            else:
                aggs += benchmark['selected_features'][:, sparseness_level, :]

    plot_feature_selection_worst_case_per_epoch(aggs, n_runs, sample_epochs, show_plot=True, save_plot=True, title=True)
