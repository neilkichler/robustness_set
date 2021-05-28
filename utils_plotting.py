import inspect
import numpy as np


def current_method_name():
    # [0] is this method's frame, [1] is the parent's frame - which we want
    return inspect.stack()[1].function


def get_model_names(data):

    models = data['models']
    model_names = [type(m).__name__ for m in models]
    return model_names


def feature_selection_mean(weights=None, sparsity=0.4):
    means = np.asarray(np.mean(np.abs(weights), axis=1)).flatten()
    means_sorted = np.sort(means)
    threshold_idx = int(means.size * sparsity)

    n = len(means)
    if threshold_idx == n:
        return np.ones(n, dtype=bool)

    means_threshold = means_sorted[threshold_idx]

    feature_selection = means >= means_threshold

    return feature_selection
