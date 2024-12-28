# %%
from svc_helper.pitch.rmvpe import RMVPEModel
import librosa

rmvpe_model_160 = RMVPEModel(hop_length=160)
rmvpe_model_200 = RMVPEModel(hop_length=200)

data, rate = librosa.load('tests/test_speech.wav', sr=16000)

pitch_160 = rmvpe_model_160.extract_pitch(data)
pitch_200 = rmvpe_model_200.extract_pitch(data)

# %%
import matplotlib.pyplot as plt

plt.plot(pitch_160)
plt.plot(pitch_200)
plt.show()
# %%
