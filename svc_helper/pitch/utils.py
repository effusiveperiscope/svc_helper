import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, make_smoothing_spline

def nonzero_mean(x):
    return np.mean(x[x.nonzero()])

# Linear bins, quantilized on nonzero values, with 0 as first bin
def f0_quantilize(x, n_bins=5):
    bins = np.concatenate(([0], np.quantile(x[x.nonzero()], np.linspace(0, 1, n_bins))))
    return np.digitize(x, bins)

def smooth_pitch(pitch, lam=0.4):
    """
    Pitch smoothing function that preserves on/offsets
    """
    nonzero_indices = np.nonzero(pitch)[0]
    nonzero_values = pitch[nonzero_indices]

    if len(nonzero_values) == 0:
        return pitch
    
    # Use nearest neighbor interpolation of nonzero regions to avoid artifacting at onsets
    interpolator = interp1d(nonzero_indices, nonzero_values, kind='nearest',
        bounds_error=False, fill_value=(nonzero_values[0], nonzero_values[-1]))
    interpolated = interpolator(np.arange(0, pitch.shape[0]))
    smoothed_curve = make_smoothing_spline(np.arange(0, pitch.shape[0]), 
        interpolated, lam=lam)

    # Then mask to preserve onsets
    mask = (pitch != 0).astype(np.float32)
    return smoothed_curve(np.arange(0, pitch.shape[0])) * mask