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

# %%
from scipy.signal import find_peaks
def gather_peaks_maxheights(hidden, 
    num_bins = 320, distance=25, min_height = 0.015):
    all_peaks = np.zeros((hidden.shape[0], num_bins * 2 // distance))
    num_peaks = np.zeros((hidden.shape[0], 1))
    maxheights = np.zeros((hidden.shape[0], 1))
    
    # 1. find peaks
    for i, timestep in enumerate(hidden):
        peaks, properties = find_peaks(timestep, height=min_height, distance=distance)
        all_peaks[i, 0:len(peaks)] = peaks
        num_peaks[i] = len(peaks)
        if len(peaks) == 0:
            maxheights[i] = 0
        else:
            maxheight = np.max(timestep)
            maxheights[i] = maxheight

    return all_peaks, num_peaks, maxheights

all_peaks, num_peaks, maxheights = gather_peaks_maxheights(hidden)
# print(maxheights)
# vuv_gt = (pitch != 0) * 100
# vuv = (maxheights >= 0.05) * 100
# plt.plot(vuv)
# plt.plot(vuv_gt)
# plt.legend(['vuv', 'vuv_gt'])
# plt.show()

# 90 - octave ambig.
# 122 - wrong octave confidence
timestep_to_look = 130
#plt.plot(np.max(hidden, axis=1))
plt.plot(hidden[timestep_to_look])
print(all_peaks[timestep_to_look, 0:10])
plt.show()

# %%

def score_path(path,
    jump_penalty = 0.1,
    length_reward = 0.1,
    jump_threshold = 20,
    print_jumps=False):
    score = 0
    for i in range(len(path) - 1):
        diff = path[i + 1] - path[i]
        if np.abs(diff) > jump_threshold: # penalize only large jumps
            if print_jumps:
                print(f'jump at {i}')
            score -= jump_penalty * diff
        score += length_reward
    return score

def decode2(hidden, 
    voiced_threshold = 0.05,
    num_bins = 360): # semi-greedy decoding based on multiple paths

    all_peaks, num_peaks, maxheights = gather_peaks_maxheights(
        hidden, num_bins=num_bins)
    output_path = np.zeros((hidden.shape[0], 1))

    in_voiced = False
    current_paths = []
    num_paths = 0
    for i in range(len(hidden)):
        # if i == 325:
        #     print('here')
        #     import ipdb; ipdb.set_trace()
        if maxheights[i] < voiced_threshold: # end path or unvoiced
            if in_voiced:
                print('ending path at ', i)
                scores = [score_path(path) for path in current_paths]
                # if i == 285:
                    #scores = [score_path(path, print_jumps=True) for path in current_paths]
                    # score_path(current_paths[0], print_jumps=True)
                    # plt.plot(current_paths[0])
                    # for j,path in enumerate(current_paths):
                    #     plt.plot(path)
                    # plt.legend([f'path {j}' for j in range(len(current_paths))])
                    # plt.show()
                    # print(scores)
                best_path = current_paths[np.argmax(scores)]
                output_path[i - len(best_path) + 1: i+1] = np.array(best_path)[:, np.newaxis]
            in_voiced = False

            num_paths = 0
            current_paths.clear()
            cur_num_peaks = 0
            continue
        if not in_voiced: # begin new path
            in_voiced = True
            cur_num_peaks = int(num_peaks[i].item())
            num_paths += cur_num_peaks
            start_peaks = all_peaks[i, 0:cur_num_peaks]
            for s in start_peaks:
                current_paths.append([int(s)])
        else:
            if i == 122:
                import ipdb; ipdb.set_trace()
            cur_num_peaks = int(num_peaks[i].item())
            prev_peaks = np.array([path[-1] for path in current_paths])
            cur_peaks = all_peaks[i, 0:cur_num_peaks]

            peak_dists = np.abs(prev_peaks[:, np.newaxis] - cur_peaks[np.newaxis, :])
            novel_peaks = set(range(cur_num_peaks))

            for i, path in enumerate(current_paths):
                # print(i)
                # print(peak_dists.shape)
                new_peak = np.argmin(peak_dists[i])
                novel_peaks.discard(new_peak)
                path.append(int(cur_peaks[new_peak]))
            for novel_peak in novel_peaks:
                current_paths.append([int(cur_peaks[novel_peak])])
    return output_path

# gt = pitch
pred = decode2(hidden)
plt.plot(pred)
plt.plot(pitch)
plt.legend(['pred', 'gt'])
plt.show()

# %%
