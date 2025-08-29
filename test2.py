# %%

from svc_helper.pitch.utils import discretize_f0_with_deltas, discretize_f0_log
from svc_helper.pitch.rmvpe import RMVPEModel
import matplotlib.pyplot as plt
import librosa
import numpy as np

rmvpe_model = RMVPEModel()
data, rate = librosa.load('tests/modes.wav',
    sr=RMVPEModel.expected_sample_rate)
pitch, hidden = rmvpe_model.extract_pitch(data, return_hidden=True)
hidden = np.log(hidden)

# %%
from scipy.signal import find_peaks
def gather_peaks_maxheights(hidden, 
    num_bins = 320, distance=25):
    all_peaks = np.ndarray((hidden.shape[0], num_bins * 2 // distance))
    num_peaks = np.ndarray((hidden.shape[0], 1))
    maxheights = np.ndarray((hidden.shape[0], 1))
    
    # 1. find peaks
    for i, timestep in enumerate(hidden):
        peaks, properties = find_peaks(timestep, height=0.01, distance=distance)
        all_peaks[i, 0:len(peaks)] = peaks
        num_peaks[i] = len(peaks)
        if len(peaks) == 0:
            maxheights[i] = 0
        else:
            maxheight = np.max(properties['peak_heights'])
            maxheights[i] = maxheight

    return all_peaks, num_peaks, maxheights

def score_path(path,
    delta_penalty = 0.1,
    length_reward = 0.1):
    score = 0
    for i in range(len(path) - 1):
        diff = path[i + 1] - path[i]
        score -= delta_penalty * diff
        score += length_reward
    return score

def decode2(hidden, 
    voiced_threshold = 0.5,
    num_bins = 360): # semi-greedy decoding based on multiple paths

    all_peaks, num_peaks, maxheights = gather_peaks_maxheights(
        hidden, num_bins=num_bins)
    output_pitch = np.ndarray((hidden.shape[0], 1))

    in_voiced = False
    current_paths = []
    num_paths = 0
    len_path = 0
    for i in range(len(hidden)):
        if maxheights[i] < voiced_threshold: # end path or unvoiced
            in_voiced = False

            if len(current_paths) > 0:
                scores = [score_path(path) for path in current_paths]
                best_path = current_paths[np.argmax(scores)]
                output_pitch[i - len(best_path) + 1:] = best_path

            num_paths = 0
            len_path = 0
            current_paths = []
            cur_num_peaks = 0
            continue
        if not in_voiced: # begin new path
            in_voiced = True
            cur_num_peaks = int(num_peaks[i].item())
            num_paths += cur_num_peaks
            start_peaks = all_peaks[i, 0:cur_num_peaks]
            len_path = 1
            for s in start_peaks:
                current_paths.append([int(s)])
        else:
            prev_num_peaks = cur_num_peaks
            cur_num_peaks = int(num_peaks[i].item())
            prev_peaks = all_peaks[i - 1, 0:prev_num_peaks]
            cur_peaks = all_peaks[i, 0:cur_num_peaks]

            peak_dists = np.abs(prev_peaks[:, np.newaxis] - cur_peaks[np.newaxis, :])
            novel_peaks = set(range(cur_num_peaks))
            for i, path in enumerate(current_paths):
                new_peak = np.argmin(peak_dists[i])
                novel_peaks.discard(new_peak)
                path.append(int(cur_peaks[new_peak]))
            for novel_peak in novel_peaks:
                current_paths.append([int(cur_peaks[novel_peak])])
            len_path += 1
    return output_pitch

output_pitch = decode2(hidden)
plt.plot(output_pitch)

# %%
