import numpy as np
import torch
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

def f0_to_coarse(pitch: np.ndarray, # Coarse pitch from RVC
        f0_min = 50,
        f0_max = 1100,
    ) -> torch.Tensor:
    """Converts f0 to coarse representation."""
    if type(pitch) is torch.Tensor:
        pitch = pitch.detach().cpu().numpy()
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + pitch / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int32)
    return torch.from_numpy(f0_coarse).unsqueeze(0)