import numpy as np

def nonzero_mean(x):
    return np.mean(x[x.nonzero()])

# Linear bins, quantilized on nonzero values, with 0 as first bin
def f0_quantilize(x, n_bins=5):
    bins = np.concatenate(([0], np.quantile(x[x.nonzero()], np.linspace(0, 1, n_bins))))
    return np.digitize(x, bins)