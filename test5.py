
# %%
from svc_helper.pitch.rmvpe import RMVPEModel
import matplotlib.pyplot as plt
import librosa
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

rmvpe_model = RMVPEModel()
# %%
data, rate = librosa.load('tests/test_2_03.wav',
    sr=RMVPEModel.expected_sample_rate)
pitch, hidden = rmvpe_model.extract_pitch(data, return_hidden=True)

def fake_bin_curve(hidden, voiced_thred = 0.05):
    # make fake bin curve lerping in unvoiced segments - 
    # this way large pitch jumps across unvoiced segments are not penalized
    bins = np.argmax(hidden, axis=1)
    maxheights = np.max(hidden, axis=1)

    vuv = maxheights >= voiced_thred

    if not len(maxheights[vuv]): # Completely unvoiced
        return np.zeros((hidden.shape[0], 1))

    interpolator = interp1d(np.arange(0, hidden.shape[0])[vuv], bins[vuv],
        kind='linear', bounds_error=False, fill_value='extrapolate')
    interpolated = interpolator(np.arange(0, hidden.shape[0]))
    bins[~vuv] = interpolated[~vuv]
    return bins, vuv

def gather_peaks(hidden, 
        distance = 30, 
        min_log_height = -7,
        likely_voiced_log_thred = -3,
        octave_height = 60,
        octave_eps = 2):
    num_bins = hidden.shape[1]
    # Use log scale to find peaks
    log_hidden = np.log(hidden)

    peak_vals = np.zeros((hidden.shape[0], num_bins * 2 // distance))
    peak_counts = np.zeros((hidden.shape[0], 1))

    fake_bins, vuv = fake_bin_curve(hidden)

    for i, timestep in enumerate(hidden):
        peaks, properties = find_peaks(log_hidden[i], height=min_log_height, distance=distance)
        primary_peak = np.argmax(log_hidden[i])

        likely_voiced = log_hidden[i, primary_peak] >= likely_voiced_log_thred
        if likely_voiced:
            # The most likely failure mode is an octave up or down
            
            # We always assume +1 octave and -1 octave are possible
            peaks = [
                primary_peak,
                primary_peak + octave_height,
                primary_peak - octave_height
            ]
            peak_vals[i, 0:len(peaks)] = peaks
            peak_counts[i] = len(peaks)
        else:
            peaks = [np.round(fake_bins[i])]
            peak_vals[i, 0:len(peaks)] = peaks
            peak_counts[i] = len(peaks)

    return peak_vals, peak_counts, vuv

def decode_f0_center_path(
    hidden,
    peak_vals, peak_counts, vuv,
    pmf_coef = 10, # rewards probability mass from original distribution
    # prevents collapsing to wrong octave in case of equally likely paths 
    # (i.e. no obvious octave artifacts)
    delta_coef = 1, octave_coef = 60,
    octave_height = 60, octave_eps = 2,
    eps = 1e-9):
    T = peak_vals.shape[0]
    P = int(np.max(peak_counts).item())

    dp_cost = np.full((T, P), np.inf)
    backptr = -np.ones((T, P), dtype=int)

    if T <= 1:
        return peak_vals

    # Base
    dp_cost[0, :] = 0

    # Forward pass
    for t in range(1, T):
        this_peak_vals = peak_vals[t][0:int(peak_counts[t][0])]
        for i,p in enumerate(this_peak_vals):
            p = int(p)
            node_cost = -np.log(hidden[t, p] * pmf_coef + eps)
            # if t == 700:
            #     print(p, index_to_f0(p), hidden[t, p], node_cost)

            best_cost = np.inf
            best_prev = -1

            prev_peak_vals = peak_vals[t - 1][0:int(peak_counts[t - 1][0])]
            for j,p_prev in enumerate(prev_peak_vals):
                delta_cost = np.abs(p - p_prev) * delta_coef
                if (np.abs(np.abs(p - p_prev) - octave_height)) <= octave_eps:
                    octave_cost = octave_coef
                else:
                    octave_cost = 0
                cost = node_cost + delta_cost + octave_cost
                if cost < best_cost:
                    best_cost = cost
                    best_prev = j

            # if t % 100 == 0:
            #     print(t, prev_peak_vals[best_prev].astype(int), 
            #         index_to_f0(prev_peak_vals[best_prev].astype(int)), best_cost)

            dp_cost[t, i] = best_cost
            backptr[t, i] = best_prev

    # Backward
    path_values = []
    i = np.argmin(dp_cost[-1, :])
    for t in reversed(range(T)):
        path_values.append(peak_vals[t, i])
        i = backptr[t, i]
    path_values.reverse()
    return (path_values * vuv).astype(int)

def index_to_f0(index: int):
    cents_mapping = 20 * np.arange(360) + 1997.3794084376191
    f0 = 10 * (2 ** (cents_mapping[index] / 1200)) + 10
    return f0

def decode_f0_mass(
    center_path : np.ndarray, 
    hidden : np.ndarray,
    octave_height = 60,
    return_confidence=False,
    return_subharmonic_confidence=False,
    return_inharmonic_confidence=False,
    return_pitch_normalized_hidden=False, # <-- New argument
    smooth_extras=False):
    """
    Decodes F0 and optionally extracts features from hidden states based on a center path.

    Args:
        center_path (np.ndarray): The Viterbi-decoded path of center bins.
        hidden (np.ndarray): The RMVPE hidden states (salience map).
        octave_height (int): The number of bins in an octave.
        return_confidence (bool): Whether to return the salience around the F0.
        return_subharmonic_confidence (bool): Whether to return the salience at the subharmonic.
        return_inharmonic_confidence (bool): Whether to return the salience outside the F0.
        return_pitch_normalized_hidden (bool): Whether to return the hidden states circularly
                                               shifted to align the F0.
        smooth_extras (bool): Whether to apply a Gaussian filter to the extra features.

    Returns:
        tuple[np.ndarray, dict]: A tuple containing the F0 curve and a dictionary of extra features.
    """
    # normal mass averaging from RMVPE

    cents_mapping = 20 * np.arange(360) + 1997.3794084376191
    cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    todo_salience = []
    todo_cents_mapping = []

    vuv = center_path != 0
    center_path = np.clip(center_path - 4, 0, 359)
    starts = np.clip(center_path - 4, 0, 359)
    ends = np.clip(center_path + 5, 0, 359)

    for idx in range(hidden.shape[0]):
        if vuv[idx] == False: # unvoiced
            todo_salience.append(np.zeros(9))
            todo_cents_mapping.append(np.zeros(9))
        else:
            todo_salience.append(hidden[idx, starts[idx] : ends[idx]])
            todo_cents_mapping.append(cents_mapping[starts[idx] : ends[idx]])
    todo_salience = np.array(todo_salience)  # 帧长，9
    todo_cents_mapping = np.array(todo_cents_mapping)  # 帧长，9

    confidence = np.sum(todo_salience, 1)

    product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
    weight_sum = np.sum(todo_salience, 1) + 1e-6  # 帧长
    divided = product_sum / weight_sum  # 帧长

    f0 = 10 * (2 ** (divided / 1200)) + 10
    f0[vuv == False] = 0

    extras = {}

    smoothing_sigma = 3
    if return_subharmonic_confidence:
        subharmonic_path = np.clip(center_path - octave_height, 0, 359)
        subharmonic_salience = np.zeros((hidden.shape[0], 9))
        subharmonic_starts = np.clip(subharmonic_path - 4, 0, 359)
        subharmonic_ends = np.clip(subharmonic_path + 5, 0, 359)
        if subharmonic_path[0] < 0:
            subharmonic_starts[0] = 0
        for idx in range(hidden.shape[0]):
            if vuv[idx] == False: # unvoiced
                subharmonic_salience[idx] = np.zeros(9)
            else:
                subharmonic_salience[idx] = hidden[idx, subharmonic_starts[idx] : subharmonic_ends[idx]]
        subharmonic_confidence = np.sum(subharmonic_salience, 1)
        if smooth_extras:
            subharmonic_confidence = gaussian_filter1d(subharmonic_confidence, smoothing_sigma)
        extras['subharmonic_confidence'] = subharmonic_confidence

    if return_inharmonic_confidence:
        inharmonic_confidence = np.sum(hidden, 1) - (confidence) 
        if smooth_extras:
            inharmonic_confidence = gaussian_filter1d(inharmonic_confidence, smoothing_sigma)
        extras['inharmonic_confidence'] = inharmonic_confidence

    if return_confidence:
        if smooth_extras:
            confidence = gaussian_filter1d(confidence, smoothing_sigma)
        extras['confidence'] = confidence

    if return_pitch_normalized_hidden:
        num_bins = hidden.shape[1]
        center_bin_target = num_bins // 2
        
        # Initialize with zeros. Unvoiced frames will remain as zero vectors.
        normalized_hidden = np.zeros_like(hidden)

        for i in range(hidden.shape[0]):
            if vuv[i]:
                # Calculate the circular shift needed to move the center_path bin to the target center
                shift = center_bin_target - center_path[i]
                normalized_hidden[i, :] = np.roll(hidden[i, :], shift)
            else: # don't normalize unvoiced
                normalized_hidden[i, :] = hidden[i, :]
        
        extras['pitch_normalized_hidden'] = normalized_hidden

    extras['vuv'] = vuv

    return f0, extras

# --- Main execution and plotting ---


# %%
peak_vals, peak_counts, vuv = gather_peaks(hidden)
path = decode_f0_center_path(hidden, peak_vals, peak_counts, vuv)

# peak_vals: (2447, 24)
# hidden: (2447, 360)
peak_viz = np.zeros(hidden.shape)
for i in range(peak_vals.shape[0]):
    peak_viz[i, peak_vals[i].astype(int)] = hidden[i, peak_vals[i].astype(int)]
plt.imshow(peak_viz.T, aspect='auto', cmap='inferno', interpolation='none')
plt.colorbar()
plt.title('Peak Viz')
plt.show()

# Call the function with the new flag set to True
pitch2, extras = decode_f0_mass(path, hidden,
    return_confidence=True,
    return_subharmonic_confidence=True,
    return_inharmonic_confidence=True,
    return_pitch_normalized_hidden=True, # <-- Enable the new feature
    smooth_extras=True)

# Plot 1: Original vs. Refined Pitch
plt.figure(figsize = (12, 6), dpi=150)

plt.subplot(2, 1, 1)
plt.plot(pitch2, label='Refined Pitch (pitch2)')
plt.plot(pitch, label='Original RMVPE Pitch', alpha=0.7)
plt.title('Original vs. Refined F0')
plt.ylabel('Frequency (Hz)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(extras['subharmonic_confidence'], label='Subharmonic Confidence')
plt.plot(extras['confidence'], label='Confidence')
plt.plot(extras['inharmonic_confidence'], label='Inharmonic Confidence')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Heatmap of Pitch-Normalized Hidden States
# pitch_normalized_hidden = extras.get('pitch_normalized_hidden')
# if pitch_normalized_hidden is not None:
#     plt.subplot(2, 1, 2)
#     # Use log scale for better visibility of low-probability bins
#     log_normalized_hidden = np.log(pitch_normalized_hidden.T + 1e-9)
    
#     # We transpose the matrix so time is on the x-axis
#     plt.imshow(log_normalized_hidden, aspect='auto', origin='lower',
#                cmap='magma')
    
#     num_bins = pitch_normalized_hidden.shape[1]
#     center_bin = num_bins // 2
    
#     # Add a line to indicate the center (where the F0 is now aligned)
#     plt.axhline(y=center_bin, color='cyan', linestyle='--', linewidth=1, label=f'F0 Center (Bin {center_bin})')
#     # Add lines for octaves relative to the F0
#     plt.axhline(y=center_bin + 60, color='lime', linestyle=':', linewidth=0.8, label='Octave +1')
#     plt.axhline(y=center_bin - 60, color='lime', linestyle=':', linewidth=0.8, label='Octave -1')
    
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Pitch-Normalized Hidden States (Log-Salience)')
#     plt.xlabel('Time Frames')
#     plt.ylabel('Relative Pitch Bins')
#     plt.legend(fontsize='small')

plt.tight_layout()
plt.show()
# %%
