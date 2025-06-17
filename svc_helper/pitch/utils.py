import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage
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

def discretize_f0_log(f0, n_voiced_bins=4, hold_length=None):
    """
    Discretize F0 values into bins based on logarithmic pitch perception.
    
    Args:
        f0: F0 array where 0 indicates unvoiced frames
        n_voiced_bins: Number of bins for voiced frames (total bins = n_voiced_bins + 1)
        hold_length: Optional sample-and-hold length for smoothing voiced segments.
                    If specified, brief pitch changes shorter than hold_length frames
                    will be smoothed out.
    
    Returns:
        Discretized F0 array where:
        - 0 = unvoiced (f0 == 0)
        - 1 to n_voiced_bins = voiced bins based on log-quantiles
    """
    f0_processed = f0.copy()
    
    # Apply sample-and-hold filtering if requested
    if hold_length is not None and hold_length > 1:
        f0_processed = sample_and_hold_f0(f0_processed, hold_length)
    
    # Extract non-zero (voiced) F0 values
    voiced_f0 = f0_processed[f0_processed > 0]
    
    if len(voiced_f0) == 0:
        # All frames are unvoiced
        return np.zeros_like(f0, dtype=int)
    
    # Calculate quantiles in log space for voiced frames
    # Omit the last quantile (100th percentile) to avoid outlier-dominated top bin
    log_f0 = np.log(voiced_f0)
    quantile_points = np.linspace(0, 1, n_voiced_bins + 1)[:-1]  # Remove last point
    log_quantiles = np.quantile(log_f0, quantile_points)
    # Add infinity as the upper bound to capture all remaining values
    log_quantiles = np.concatenate([log_quantiles, [np.inf]])
    
    # Convert back to linear space for bin edges
    # Note: we include -inf as the lower bound to catch any edge cases
    bin_edges = np.concatenate(([0, np.finfo(float).eps], np.exp(log_quantiles[1:-1]), [np.inf]))
    
    # Digitize the original F0 values
    discretized = np.digitize(f0_processed, bin_edges)
    
    # Ensure unvoiced frames (f0 == 0) are mapped to bin 0
    discretized[f0_processed == 0] = 0
    
    # Shift voiced bins to start from 1 (since bin 0 is reserved for unvoiced)
    discretized[f0_processed > 0] = discretized[f0_processed > 0] - 1
    
    return discretized

def discretize_f0_with_deltas(f0, n_delta_bins=4, smooth_period=50, hold_length=None):
    """
    Discretize F0 into a smoothed baseline and discrete delta bins.
    
    Args:
        f0: F0 array where 0 indicates unvoiced frames
        n_delta_bins: Number of bins for delta values (total bins = 2*n_delta_bins + 1)
        smooth_period: Period for heavy smoothing of baseline F0 (in frames)
        hold_length: Optional sample-and-hold length before smoothing
    
    Returns:
        tuple: (smoothed_f0, delta_bins) where:
        - smoothed_f0: Heavily smoothed F0 baseline at original frame rate
        - delta_bins: Discretized deltas from baseline where:
            * 0 = unvoiced (original f0 == 0)
            * 1 to n_delta_bins = negative deltas (pitch going down)
            * n_delta_bins + 1 = no change (delta ≈ 0)
            * n_delta_bins + 2 to 2*n_delta_bins + 1 = positive deltas (pitch going up)
    """
    f0_processed = f0.copy()
    
    # Apply sample-and-hold filtering if requested
    if hold_length is not None and hold_length > 1:
        f0_processed = sample_and_hold_f0(f0_processed, hold_length)
    
    # Create heavily smoothed baseline
    smoothed_f0 = smooth_f0_baseline(f0_processed, smooth_period)
    
    # Calculate deltas in log space (only for voiced frames)
    voiced_mask = (f0_processed > 0) & (smoothed_f0 > 0)
    deltas = np.zeros_like(f0_processed)
    
    if np.any(voiced_mask):
        # Compute log deltas for voiced frames
        log_deltas = np.log(f0_processed[voiced_mask]) - np.log(smoothed_f0[voiced_mask])
        deltas[voiced_mask] = log_deltas
    
    # Discretize the deltas
    delta_bins = discretize_deltas(deltas, voiced_mask, n_delta_bins)
    
    return smoothed_f0, delta_bins


def smooth_f0_baseline(f0, smooth_period):
    """
    Create a heavily smoothed F0 baseline, treating all voiced frames as continuous
    and ignoring unvoiced gaps for smoothing purposes.
    
    Args:
        f0: F0 array where 0 indicates unvoiced frames
        smooth_period: Smoothing window size in frames
    
    Returns:
        Smoothed F0 array maintaining voiced/unvoiced structure
    """
    result = np.zeros_like(f0)
    voiced_mask = f0 > 0
    
    if not np.any(voiced_mask):
        return result
    
    # Extract all voiced F0 values as a continuous sequence
    voiced_f0 = f0[voiced_mask]
    voiced_indices = np.where(voiced_mask)[0]
    
    if len(voiced_f0) <= smooth_period:
        # Too few voiced frames for meaningful smoothing
        smoothed_voiced = np.full_like(voiced_f0, np.median(voiced_f0))
    else:
        # Apply heavy smoothing in log space across all voiced frames
        log_voiced = np.log(voiced_f0)
        smoothed_log = ndimage.uniform_filter1d(log_voiced, size=smooth_period, mode='nearest')
        smoothed_voiced = np.exp(smoothed_log)
    
    # Map the smoothed values back to their original positions
    result[voiced_indices] = smoothed_voiced
    
    return result


def discretize_deltas(deltas, voiced_mask, n_delta_bins):
    """
    Discretize delta values into symmetric bins around zero.
    
    Args:
        deltas: Log-space delta values
        voiced_mask: Boolean mask for voiced frames
        n_delta_bins: Number of bins on each side of zero
    
    Returns:
        Discretized delta bins
    """
    delta_bins = np.zeros(len(deltas), dtype=int)
    
    if not np.any(voiced_mask):
        return delta_bins
    
    voiced_deltas = deltas[voiced_mask]
    
    if len(voiced_deltas) == 0:
        return delta_bins
    
    # Calculate symmetric quantiles for positive and negative deltas
    abs_deltas = np.abs(voiced_deltas)
    
    if len(abs_deltas) == 0 or np.all(abs_deltas == 0):
        # All deltas are zero, everything goes to center bin
        delta_bins[voiced_mask] = n_delta_bins + 1
        return delta_bins
    
    # Create symmetric bins based on absolute delta magnitudes
    # Omit the last quantile to avoid outlier-dominated bins
    quantile_points = np.linspace(0, 1, n_delta_bins + 1)[:-1]
    magnitude_thresholds = np.quantile(abs_deltas[abs_deltas > 0], quantile_points[1:])  # Skip 0th quantile
    
    # Create bin edges: [..., -thresh2, -thresh1, 0, +thresh1, +thresh2, ...]
    neg_edges = -magnitude_thresholds[::-1]  # Reverse for negative side
    pos_edges = magnitude_thresholds
    
    bin_edges = np.concatenate([[-np.inf], neg_edges, [0], pos_edges, [np.inf]])
    
    # Digitize the voiced deltas
    voiced_bins = np.digitize(voiced_deltas, bin_edges)
    delta_bins[voiced_mask] = voiced_bins
    
    # Unvoiced frames stay at 0
    return delta_bins


def get_voiced_segments(voiced_mask):
    """Find start and end indices of voiced segments."""
    segments = []
    start = None
    
    for i, is_voiced in enumerate(voiced_mask):
        if is_voiced and start is None:
            start = i
        elif not is_voiced and start is not None:
            segments.append((start, i))
            start = None
    
    if start is not None:  # Handle case where audio ends on voiced segment
        segments.append((start, len(voiced_mask)))
    
    return segments


def sample_and_hold_f0(f0, hold_length):
    """
    Apply sample-and-hold filtering to F0 values to smooth brief pitch changes.
    
    Args:
        f0: F0 array where 0 indicates unvoiced frames
        hold_length: Minimum length (in frames) for a pitch value to be maintained
    
    Returns:
        Smoothed F0 array
    """
    if hold_length <= 1:
        return f0
    
    result = f0.copy()
    voiced_mask = f0 > 0
    voiced_segments = get_voiced_segments(voiced_mask)
    
    # Apply sample-and-hold within each voiced segment
    for seg_start, seg_end in voiced_segments:
        segment_f0 = f0[seg_start:seg_end]
        smoothed_segment = apply_sample_hold_segment(segment_f0, hold_length)
        result[seg_start:seg_end] = smoothed_segment
    
    return result


def apply_sample_hold_segment(segment_f0, hold_length):
    """Apply sample-and-hold to a single voiced segment."""
    if len(segment_f0) <= hold_length:
        return np.full_like(segment_f0, np.median(segment_f0))
    
    result = segment_f0.copy()
    i = 0
    
    while i < len(segment_f0):
        current_val = segment_f0[i]
        hold_end = min(i + hold_length, len(segment_f0))
        
        # Look ahead to see if we should extend the hold
        while hold_end < len(segment_f0):
            next_segment = segment_f0[hold_end:hold_end + hold_length]
            if len(next_segment) == 0:
                break
            # If the next segment is similar enough, continue holding
            if np.abs(np.log(np.median(next_segment)) - np.log(current_val)) < 0.1:
                hold_end = min(hold_end + hold_length, len(segment_f0))
            else:
                break
        
        result[i:hold_end] = current_val
        i = hold_end
    
    return result