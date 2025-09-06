from svc_helper.pitch.rmvpe import RMVPEModel
from svc_helper.pitch.utils import f0_quantilize, nonzero_mean
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
    pitch, hidden = rmvpe_model.extract_pitch(data, return_hidden=True)

    print(nonzero_mean(pitch))
    # print(f0_quantilize(pitch))

    print(hidden.shape)

    pitch2, extras = rmvpe_model.extract_pitch2(data, 
        return_confidence=True,
        return_subharmonic_confidence=True,
        return_inharmonic_confidence=True,
        smooth_extras=True)
    
    print(pitch2.shape, extras["confidence"].shape, extras["subharmonic_confidence"].shape, extras["inharmonic_confidence"].shape)
    print(pitch2 - pitch)