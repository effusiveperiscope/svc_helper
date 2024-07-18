from svc_helper.sfeatures.models import RVCHubertModel
import numpy as np
import torch
import librosa

def test_sfeatures():
    rvc_model = RVCHubertModel()

    data, rate = librosa.load('tests/test_speech.wav',
        sr=RVCHubertModel.expected_sample_rate)
    feat = rvc_model.extract_features(torch.from_numpy(data))

    print(feat.shape)