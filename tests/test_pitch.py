from svc_helper.pitch.rmvpe import RMVPEModel
import numpy as np
import torch
import librosa

def test_pitch():
    rmvpe_model = RMVPEModel()

    data, rate = librosa.load('tests/test_speech.wav',
        sr=RMVPEModel.expected_sample_rate)
    pitch = rmvpe_model.extract_pitch(data)

    #print('pitch shape:',pitch.shape)
    #print('pitch mean:',pitch[pitch.nonzero()].mean())
    pitch = rmvpe_model.extract_pitch(data)