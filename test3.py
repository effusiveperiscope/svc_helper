# %%
from svc_helper.pitch.rmvpe import RMVPEModel
from scipy.stats import entropy
import matplotlib.pyplot as plt
import librosa
import numpy as np

rmvpe_model = RMVPEModel()
data, rate = librosa.load('tests/modes.wav',
    sr=RMVPEModel.expected_sample_rate)
pitch, hidden = rmvpe_model.extract_pitch(data, return_hidden=True)
# %%
class RMVPEDecode:
    """
    Rationale: Almost all pitch decoding errors in RMVPE are "octave down".
    """
    def __init__(self, 
        hidden, 
        voice_thred = 0.4,
        trigger_edge = 48):
        self.hidden = hidden
        self.voice_thred = voice_thred

        self.in_voiced = False
        self.in_dip = False
        self.voiced_segment_start = 0
        self.pitch_segment_start = 0
        self.trigger_edge = trigger_edge
        self.octave_bins = 60

        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def vuv_mask(self):
        vuv_mask = np.zeros((self.hidden.shape[0], 1))
        confidence = self.hidden.max(axis=1)
        vuv_mask[:, 0] = confidence >= self.voice_thred
        return vuv_mask, confidence

    def find_center_path(self):
        vuv_mask, confidence = self.vuv_mask()
        decoded = np.argmax(self.hidden, axis=1)

        # Correct for octave down errors
        for timestep in range(hidden.shape[0]):
            t = timestep

            if not vuv_mask[t]:
                self.in_voiced = False
                continue

            if not self.in_voiced: # new voiced segment
                if self.in_dip:
                    decoded[self.pitch_segment_start:t] += self.octave_bins
                self.voiced_segment_start = t
                self.pitch_segment_start = t

            self.in_voiced = True

            if t > 0:
                edge = decoded[t] - decoded[t-1]
                print(edge, t)
                if edge <= -self.trigger_edge: # falling edge
                    self.pitch_segment_start = t
                    self.in_dip = True
                elif edge >= self.trigger_edge: # rising edge
                    # raise octave of segment prior
                    if self.in_dip:
                        decoded[self.pitch_segment_start:t] += self.octave_bins
                    self.pitch_segment_start = t
                    self.in_dip = False
        if self.in_dip:
            decoded[self.pitch_segment_start:] += self.octave_bins

        return decoded * vuv_mask[:, 0], confidence

    def find_scaled(self):
        decoded, confidence = self.find_center_path()
        cents = (decoded * 20 + 1997.3794084376191)
        f0 = 10 * (2 ** (cents / 1200))
        f0[f0 == f0.min()] = 0
        return f0, confidence
                    

# %%
decoded, confidence = RMVPEDecode(hidden).find_scaled()
plt.figure(figsize=(20,5))
plt.xticks(np.arange(0, hidden.shape[0], 50))
# plt.plot(confidence)
plt.plot(hidden.argmax(axis=1))
plt.show()
# plt.figure(figsize=(20,5))
# plt.plot(hidden.max(axis=1))
#plt.plot(pitch)
# plt.plot(entropy(hidden, axis=1))
# plt.show()

# %%
