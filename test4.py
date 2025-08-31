# %%
from svc_helper.pitch.rmvpe import RMVPEModel
import matplotlib.pyplot as plt
import librosa
import numpy as np

rmvpe_model = RMVPEModel()
data, rate = librosa.load('tests/modes.wav',
    sr=RMVPEModel.expected_sample_rate)
pitch, hidden = rmvpe_model.extract_pitch(data, return_hidden=True)

# %%
from scipy.interpolate import interp1d

# idea: make fake bin curve lerping in unvoiced segments
def fake_bin_curve(hidden, voiced_thred = 0.05):
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

bin_curve, vuv= fake_bin_curve(hidden)
plt.plot(bin_curve)
plt.show()

# %%
from scipy.signal import find_peaks
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
            peaks = [peak for peak in peaks if 
                np.abs(np.abs(peak - primary_peak) % octave_height) <= octave_eps]
            peaks.append(primary_peak)
            peak_vals[i, 0:len(peaks)] = peaks
            peak_counts[i] = len(peaks)
        else:
            peaks = [np.round(fake_bins[i])]
            peak_vals[i, 0:len(peaks)] = peaks
            peak_counts[i] = len(peaks)

    return peak_vals, peak_counts

peak_vals, peak_counts = gather_peaks(hidden)
plt.plot(peak_counts)
plt.show()

heatmap = np.zeros((hidden.shape[1], hidden.shape[0]))
for i in range(hidden.shape[0]):
    for j in range(int(peak_counts[i][:])):
        heatmap[int(peak_vals[i, j]), i] = 1
plt.figure(figsize = (10, 2), dpi=300)
plt.imshow(heatmap)
plt.show()

# %%
maxheights = np.argmax(hidden, axis=1)
plt.plot(maxheights)
plt.show()
# %%
